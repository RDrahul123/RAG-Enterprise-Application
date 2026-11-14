"""LLM providers.

Three interchangeable backends, in order of preference:

* ``local``     – any Hugging Face instruct model via transformers, on CPU, free.
* ``hf_api``    – Hugging Face Inference API free tier (needs HF_TOKEN).
* ``extractive``– no LLM at all: selects the best supporting sentences from the
  retrieved chunks. Always available, never hallucinates, and doubles as the
  control group in the eval harness.

``RAG_LLM_PROVIDER=auto`` picks the best option that will actually work on this
machine without surprising you with a multi-gigabyte download.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path

from app.config import Settings
from app.observability import log_event


@dataclass
class GenerationResult:
    text: str
    provider: str
    model: str
    meta: dict = field(default_factory=dict)


class BaseProvider:
    name = "base"
    model = ""

    @property
    def ready(self) -> bool:
        return True

    def generate(self, system: str, user: str) -> GenerationResult:  # pragma: no cover
        raise NotImplementedError


class ExtractiveProvider(BaseProvider):
    """No LLM. Picks sentences with the highest query-term overlap."""

    name = "extractive"
    model = "none"

    def generate(self, system: str, user: str) -> GenerationResult:  # pragma: no cover
        raise NotImplementedError("Extractive answers are built in app.generation.rag")


class LocalProvider(BaseProvider):
    """transformers + a causal LM loaded lazily on first call."""

    name = "local"

    def __init__(self, model_name: str, max_new_tokens: int = 512, temperature: float = 0.1) -> None:
        self.model = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._pipe = None

    @property
    def ready(self) -> bool:
        return importlib.util.find_spec("transformers") is not None

    def _load(self):
        if self._pipe is None:
            from transformers import pipeline

            log_event("llm_loading", provider="local", model=self.model)
            kwargs = {"model": self.model, "device_map": None}
            try:
                # transformers >= 4.56 renamed torch_dtype -> dtype
                self._pipe = pipeline("text-generation", dtype="auto", **kwargs)
            except TypeError:
                self._pipe = pipeline("text-generation", torch_dtype="auto", **kwargs)
        return self._pipe

    def generate(self, system: str, user: str) -> GenerationResult:
        pipe = self._load()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        do_sample = self.temperature > 0
        kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "return_full_text": False,
        }
        if do_sample:
            kwargs["temperature"] = self.temperature
        out = pipe(messages, **kwargs)
        text = out[0]["generated_text"]
        if isinstance(text, list):  # some pipelines return chat messages
            text = text[-1].get("content", "")
        return GenerationResult(text=str(text).strip(), provider=self.name, model=self.model)


class HFAPIProvider(BaseProvider):
    """Hugging Face Inference API, free tier."""

    name = "hf_api"

    def __init__(self, model_name: str, token: str | None, max_new_tokens: int = 512, temperature: float = 0.1) -> None:
        self.model = model_name
        self.token = token
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._client = None

    @property
    def ready(self) -> bool:
        return bool(self.token)

    def _load(self):
        if self._client is None:
            from huggingface_hub import InferenceClient

            self._client = InferenceClient(model=self.model, token=self.token)
        return self._client

    def generate(self, system: str, user: str) -> GenerationResult:
        client = self._load()
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self.max_new_tokens,
            # Some hosted APIs reject temperature=0 outright.
            temperature=max(self.temperature, 0.01),
        )
        return GenerationResult(
            text=response.choices[0].message.content.strip(),
            provider=self.name,
            model=self.model,
        )


def _model_is_cached(model_name: str) -> bool:
    """True if the model is already in the local HF cache (so loading is instant)."""
    try:
        cache = Path(os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"
        folder = "models--" + model_name.replace("/", "--")
        return (cache / folder).exists()
    except Exception:
        return False


def build_provider(settings: Settings) -> BaseProvider:
    desired = (settings.llm_provider or "auto").lower()

    if desired == "extractive":
        return ExtractiveProvider()
    if desired == "local":
        return LocalProvider(settings.llm_model, settings.max_new_tokens, settings.temperature)
    if desired == "hf_api":
        return HFAPIProvider(settings.llm_model, settings.hf_token, settings.max_new_tokens, settings.temperature)

    # auto: never trigger a surprise multi-GB download
    if _model_is_cached(settings.llm_model) and importlib.util.find_spec("transformers"):
        log_event("provider_selected", provider="local", reason="model_cached")
        return LocalProvider(settings.llm_model, settings.max_new_tokens, settings.temperature)
    if settings.hf_token:
        log_event("provider_selected", provider="hf_api", reason="token_present")
        return HFAPIProvider(settings.llm_model, settings.hf_token, settings.max_new_tokens, settings.temperature)
    log_event(
        "provider_selected",
        provider="extractive",
        reason="no_token_and_model_not_cached",
        hint="set HF_TOKEN or run `make pull-model`",
    )
    return ExtractiveProvider()


def provider_status(provider: BaseProvider) -> dict:
    return {
        "provider": provider.name,
        "model": provider.model,
        "ready": provider.ready,
    }
