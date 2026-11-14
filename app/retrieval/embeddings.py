"""Embedding backends.

Primary path is a real Hugging Face sentence-transformer (BAAI/bge-small-en-v1.5,
~130 MB, runs on CPU in milliseconds). If torch is unavailable — first run before
the model is cached, CI, a bare container — we fall back to deterministic feature
hashing so the application still starts and the tests still run.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache

import numpy as np

from app.config import Settings
from app.observability import log_event

_WORD = re.compile(r"[a-z0-9]+")


class BaseEmbedder:
    name: str = "base"
    dim: int = 0
    uses_query_prefix: bool = False

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def encode_query(self, text: str) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


class HashingEmbedder(BaseEmbedder):
    """Feature hashing over word uni/bi-grams. Zero downloads, fully deterministic."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"
        self.uses_query_prefix = False

    def _embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        tokens = _WORD.findall(text.lower())
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            slot = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[slot] += sign
        return vector

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._normalize(np.stack([self._embed_one(t) for t in texts]))

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_documents([text])[0]


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str, batch_size: int = 32, query_prefix: str = "") -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.batch_size = batch_size
        self.query_prefix = query_prefix if "bge" in model_name.lower() else ""
        self.uses_query_prefix = bool(self.query_prefix)
        self._model = SentenceTransformer(model_name)
        getter = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension"
        )
        self.dim = int(getter())
        log_event("embedder_loaded", model=model_name, dim=self.dim)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        # bge v1.5 expects the instruction on queries only, never on passages.
        return self.encode_documents([self.query_prefix + text])[0]


@lru_cache(maxsize=1)
def get_embedder(settings: Settings, prefer_real: bool = True) -> BaseEmbedder:
    if prefer_real:
        try:
            return SentenceTransformerEmbedder(
                settings.embedding_model,
                batch_size=settings.embedding_batch_size,
                query_prefix=settings.query_prefix,
            )
        except Exception as exc:
            log_event(
                "embedder_fallback",
                requested=settings.embedding_model,
                reason=f"{type(exc).__name__}: {exc}",
            )
    return HashingEmbedder()
