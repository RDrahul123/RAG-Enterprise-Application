"""FastAPI application.

Run:  uvicorn app.main:app --reload --port 8000
Docs: http://localhost:8000/docs        UI: http://localhost:8000
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.config import ROOT
from app.models import AskRequest, AskResponse, HealthResponse, IngestResponse, StatsResponse
from app.observability import metrics
from app.service import ALLOWED_UPLOADS, RagService, get_service

app = FastAPI(
    title="Enterprise RAG",
    version=__version__,
    description=(
        "Document intelligence with hybrid retrieval, cross-encoder reranking "
        "and page-level citations."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- API -------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    return get_service().health()


@app.get("/api/stats", response_model=StatsResponse, tags=["ops"])
def stats() -> StatsResponse:
    return get_service().stats()


@app.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
def prometheus_metrics() -> str:
    return metrics.render()


@app.post("/api/ask", response_model=AskResponse, tags=["rag"])
def ask(request: AskRequest) -> AskResponse:
    service: RagService = get_service()
    if not service.index_ready:
        raise HTTPException(status_code=409, detail="No index built. POST /api/ingest first.")
    return service.ask(
        request.question,
        top_k=request.top_k,
        doc_ids=request.doc_ids,
        include_trace=request.include_trace,
    )


@app.post("/api/ingest", response_model=IngestResponse, tags=["ingest"])
def ingest() -> IngestResponse:
    service: RagService = get_service()
    try:
        return service.ingest()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/upload", response_model=IngestResponse, tags=["ingest"])
async def upload(files: list[UploadFile] = File(...)) -> IngestResponse:
    service: RagService = get_service()
    payloads: list[tuple[str, bytes]] = []
    for file in files:
        if Path(file.filename or "").suffix.lower() not in ALLOWED_UPLOADS:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type: {file.filename}. Allowed: {sorted(ALLOWED_UPLOADS)}",
            )
        payloads.append((file.filename or "upload", await file.read()))

    saved = service.save_uploads(payloads)
    if not saved:
        raise HTTPException(status_code=400, detail="No files saved.")
    try:
        return service.ingest()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# --- UI --------------------------------------------------------------------
UI_DIR = ROOT / "ui"
if UI_DIR.exists():
    app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")
