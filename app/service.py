"""Application service layer: holds the loaded index and LLM provider.

Keeping this out of the FastAPI module means the eval harness and the CLI use the
exact same code path as the HTTP API.
"""

from __future__ import annotations

import threading
from pathlib import Path

from app.config import Settings, get_settings
from app.generation.providers import build_provider, provider_status
from app.generation.rag import RagEngine
from app.ingestion.pipeline import IngestionPipeline
from app.models import AskResponse, HealthResponse, IngestResponse, StatsResponse
from app.observability import log_event, timed
from app.retrieval.embeddings import get_embedder
from app.retrieval.pipeline import RetrievalPipeline

ALLOWED_UPLOADS = {".pdf", ".md", ".markdown", ".txt"}


class RagService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._engine: RagEngine | None = None
        self._lock = threading.Lock()

    # -- lifecycle -------------------------------------------------------
    @property
    def engine(self) -> RagEngine:
        if self._engine is None:
            with self._lock:
                if self._engine is None:
                    with timed() as t:
                        pipeline = RetrievalPipeline.from_disk(
                            self.settings, prefer_real=self.settings.prefer_real
                        )
                        provider = build_provider(self.settings)
                        self._engine = RagEngine(pipeline, self.settings, provider)
                    log_event("engine_ready", ms=t["ms"], provider=provider.name)
        return self._engine

    def reset(self) -> None:
        with self._lock:
            self._engine = None

    @property
    def index_ready(self) -> bool:
        return self.settings.manifest_path.exists() and self.settings.chunks_path.exists()

    # -- operations ------------------------------------------------------
    def ingest(self, docs_dir: Path | None = None) -> IngestResponse:
        with timed() as t:
            embedder = get_embedder(self.settings, prefer_real=self.settings.prefer_real)
            stats = IngestionPipeline(self.settings, embedder).run(docs_dir)
        self.reset()
        return IngestResponse(
            documents=stats["documents"],
            pages=stats["pages"],
            chunks=stats["chunks"],
            embedder=stats["embedder"],
            dim=stats["dim"],
            elapsed_ms=t["ms"],
        )

    def save_uploads(self, files: list[tuple[str, bytes]]) -> list[str]:
        saved: list[str] = []
        for filename, payload in files:
            name = Path(filename).name
            if Path(name).suffix.lower() not in ALLOWED_UPLOADS:
                continue
            target = self.settings.data_dir / name
            target.write_bytes(payload)
            saved.append(name)
        log_event("uploads_saved", files=saved)
        return saved

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        doc_ids: list[str] | None = None,
        include_trace: bool = True,
    ) -> AskResponse:
        return self.engine.answer(question, top_k=top_k, doc_ids=doc_ids, include_trace=include_trace)

    def stats(self) -> StatsResponse:
        if not self.index_ready:
            return StatsResponse(
                indexed=False,
                documents=0,
                chunks=0,
                dim=0,
                embedder=self.settings.embedding_model,
                reranker=self.settings.reranker_model,
                llm_provider=self.settings.llm_provider,
                llm_model=self.settings.llm_model,
            )
        engine = self.engine
        info = engine.pipeline.describe()
        return StatsResponse(
            indexed=True,
            documents=info["documents"],
            chunks=info["chunks"],
            dim=info["dim"],
            embedder=info["embedder"],
            reranker=info["reranker"],
            llm_provider=engine.provider.name,
            llm_model=engine.provider.model,
            doc_list=info["doc_list"],
        )

    def health(self) -> HealthResponse:
        ready = self.index_ready
        provider = {"provider": self.settings.llm_provider, "model": self.settings.llm_model, "ready": True}
        if ready:
            try:
                provider = provider_status(self.engine.provider)
            except Exception as exc:
                return HealthResponse(
                    status="degraded",
                    index_ready=ready,
                    provider="unknown",
                    provider_ready=False,
                    detail={"error": f"{type(exc).__name__}: {exc}"},
                )
        return HealthResponse(
            status="ok" if ready else "degraded",
            index_ready=ready,
            provider=provider["provider"],
            provider_ready=bool(provider.get("ready")),
            detail={"model": provider.get("model"), "hint": "POST /api/ingest to build the index"},
        )


_service: RagService | None = None
_service_lock = threading.Lock()


def get_service() -> RagService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = RagService()
    return _service
