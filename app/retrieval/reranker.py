"""Cross-encoder reranking.

Dense + BM25 give recall; a cross-encoder gives precision. It sees the query and
the passage together, so it can tell that "the refund window is 30 days" actually
answers "how long do I have to return something?".
"""

from __future__ import annotations

import re

from app.config import Settings
from app.observability import log_event

_WORD = re.compile(r"[a-z0-9]+")


class BaseReranker:
    name: str = "base"

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        raise NotImplementedError


class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self.name = model_name
        self._model = CrossEncoder(model_name, max_length=512)
        log_event("reranker_loaded", model=model_name)

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        pairs = [(query, text) for text in texts]
        scores = self._model.predict(pairs, show_progress_bar=False)
        return [float(s) for s in scores]


class LexicalReranker(BaseReranker):
    """Stopword-filtered term coverage. The graceful-degradation path."""

    name = "lexical-overlap"

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        from app.retrieval.bm25 import STOPWORDS

        def stem(token: str) -> str:
            if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
                return token[:-1]
            return token

        terms = {stem(t) for t in _WORD.findall(query.lower()) if t not in STOPWORDS and len(t) > 2}
        if not terms:
            return [0.0 for _ in texts]
        scores: list[float] = []
        for text in texts:
            tokens = {stem(t) for t in _WORD.findall(text.lower())}
            hit = len(terms & tokens)
            scores.append(hit / len(terms))
        return scores


def get_reranker(settings: Settings, prefer_real: bool = True) -> BaseReranker:
    if prefer_real:
        try:
            return CrossEncoderReranker(settings.reranker_model)
        except Exception as exc:
            log_event(
                "reranker_fallback",
                requested=settings.reranker_model,
                reason=f"{type(exc).__name__}: {exc}",
            )
    return LexicalReranker()
