"""HTTP API. Start with: uvicorn app:app --reload --port 8000"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import answer
from rag import DOCS_DIR, EMBED_MODEL, FINAL_TOP_K, RERANK_MODEL, RAG

app = FastAPI(title="Enterprise RAG")
rag = RAG()  # loads the models and the saved index once, at startup


class Question(BaseModel):
    question: str = Field(min_length=2)
    top_k: int = FINAL_TOP_K


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "indexed": bool(rag.chunks), "chunks": len(rag.chunks)}


@app.get("/api/stats")
def stats() -> dict:
    return {
        "indexed": bool(rag.chunks),
        "documents": len({chunk.doc for chunk in rag.chunks}),
        "chunks": len(rag.chunks),
        "embedder": EMBED_MODEL,
        "reranker": RERANK_MODEL,
        "llm": "local" if answer.USE_LLM else "extractive",
    }


@app.post("/api/ask")
def ask(body: Question) -> dict:
    if not rag.chunks:
        raise HTTPException(status_code=409, detail="No index. Run: python ingest.py")
    return answer.answer(rag, body.question, body.top_k)


@app.post("/api/ingest")
def ingest() -> dict:
    try:
        return rag.ingest()
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)) -> dict:
    for file in files:
        name = Path(file.filename or "").name
        if Path(name).suffix.lower() not in {".pdf", ".md", ".txt"}:
            raise HTTPException(status_code=415, detail=f"Unsupported file: {name}")
        (DOCS_DIR / name).write_bytes(await file.read())
    return rag.ingest()


app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "ui"), html=True), name="ui")
