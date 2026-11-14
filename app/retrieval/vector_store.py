"""Dense vector index.

FAISS ``IndexFlatIP`` over L2-normalised vectors == exact cosine similarity with
zero tuning and sub-millisecond queries at portfolio scale. A numpy brute-force
implementation mirrors the same interface so the app degrades gracefully when
faiss-cpu is not installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:  # pragma: no cover - exercised implicitly
    import faiss

    _HAS_FAISS = True
except Exception:  # pragma: no cover
    faiss = None
    _HAS_FAISS = False


class VectorStore:
    def __init__(self, dim: int, index_path: Path | None = None) -> None:
        self.dim = dim
        self.index_path = index_path
        self._matrix: np.ndarray | None = None
        self._index = faiss.IndexFlatIP(dim) if _HAS_FAISS else None

    # -- lifecycle ------------------------------------------------------
    @property
    def backend(self) -> str:
        return "faiss" if self._index is not None else "numpy"

    @property
    def size(self) -> int:
        if self._index is not None:
            return int(self._index.ntotal)
        return 0 if self._matrix is None else int(self._matrix.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Expected (n, {self.dim}) vectors, got {vectors.shape}")
        if self._index is not None:
            self._index.add(vectors)
        self._matrix = vectors if self._matrix is None else np.vstack([self._matrix, vectors])

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        k = max(1, min(k, self.size))
        if self.size == 0:
            return []
        q = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        if self._index is not None:
            scores, indices = self._index.search(q, k)
            return [
                (int(i), float(s))
                for i, s in zip(indices[0], scores[0])
                if i >= 0
            ]
        # numpy fallback: exact inner product
        assert self._matrix is not None
        scores = (self._matrix @ q[0]).astype(np.float32)
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order]

    def save(self) -> None:
        if self.index_path is None:
            return
        if self._index is not None:
            faiss.write_index(self._index, str(self.index_path))
        else:
            np.save(self.index_path.with_suffix(".npy"), self._matrix)

    @classmethod
    def load(cls, dim: int, index_path: Path) -> "VectorStore":
        store = cls(dim=dim, index_path=index_path)
        if _HAS_FAISS and index_path.exists():
            store._index = faiss.read_index(str(index_path))
            if store._index.ntotal > 0:
                store._matrix = store._index.reconstruct_n(0, store._index.ntotal)
            return store
        npy_path = index_path.with_suffix(".npy")
        if npy_path.exists():
            store._matrix = np.load(npy_path)
            store._index = None
            return store
        raise FileNotFoundError(f"No vector index at {index_path}")
