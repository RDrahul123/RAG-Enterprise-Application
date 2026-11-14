"""Central configuration.

Every knob can be overridden with an environment variable (see ``.env.example``).
Keeping this in one place means the Dockerfile, the API and the eval harness all
agree on the same defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser().resolve()


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    # --- storage -------------------------------------------------------
    data_dir: Path = _path("RAG_DATA_DIR", ROOT / "data" / "docs")
    index_dir: Path = _path("RAG_INDEX_DIR", ROOT / "data" / "index")

    # --- embeddings ----------------------------------------------------
    # Set RAG_USE_REAL_MODELS=0 for fast, download-free runs (tests, CI).
    prefer_real: bool = os.getenv("RAG_USE_REAL_MODELS", "1").lower() not in {"0", "false", "no"}
    embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    embedding_batch_size: int = _int("RAG_EMBEDDING_BATCH", 32)
    # bge-*-v1.5 was trained with this instruction on the query side only.
    query_prefix: str = os.getenv(
        "RAG_QUERY_PREFIX",
        "Represent this sentence for searching relevant passages: ",
    )

    # --- reranking -----------------------------------------------------
    reranker_model: str = os.getenv(
        "RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # --- chunking ------------------------------------------------------
    chunk_tokens: int = _int("RAG_CHUNK_TOKENS", 320)
    chunk_overlap: int = _int("RAG_CHUNK_OVERLAP", 60)

    # --- retrieval -----------------------------------------------------
    dense_top_k: int = _int("RAG_DENSE_TOP_K", 24)
    bm25_top_k: int = _int("RAG_BM25_TOP_K", 24)
    rerank_top_k: int = _int("RAG_RERANK_TOP_K", 16)
    final_top_k: int = _int("RAG_FINAL_TOP_K", 6)
    rrf_k: int = _int("RAG_RRF_K", 60)
    # Cross-encoder scores are logits; below this we refuse instead of hallucinating.
    min_rerank_score: float = _float("RAG_MIN_RERANK_SCORE", -6.0)

    # --- generation ----------------------------------------------------
    # auto | extractive | local | hf_api
    llm_provider: str = os.getenv("RAG_LLM_PROVIDER", "auto")
    llm_model: str = os.getenv("RAG_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    hf_token: str | None = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    max_new_tokens: int = _int("RAG_MAX_NEW_TOKENS", 512)
    # 0 = greedy decoding: deterministic answers, reproducible evals.
    temperature: float = _float("RAG_TEMPERATURE", 0.0)

    @property
    def vectors_path(self) -> Path:
        return self.index_dir / "vectors.faiss"

    @property
    def chunks_path(self) -> Path:
        return self.index_dir / "chunks.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.index_dir / "manifest.json"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
