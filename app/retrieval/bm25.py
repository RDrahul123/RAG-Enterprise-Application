"""Lexical retrieval (BM25) — the half of hybrid search that catches exact
identifiers, product codes and names that dense embeddings tend to smear.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.models import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have
    how i if in into is it its may might of on or our so that the their them then
    there these they this to was we were what when where which who why will with
    would you your""".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self._corpus = [tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._corpus) if self._corpus else None

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        if self._bm25 is None:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    # -- persistence (rebuildable cache, not a source of truth) ----------
    def save(self, path: Path) -> None:
        payload = {"chunk_ids": [c.chunk_id for c in self.chunks]}
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, chunks: list[Chunk], path: Path | None = None) -> "BM25Index":
        return cls(chunks)
