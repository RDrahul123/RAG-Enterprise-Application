"""Ingestion pipeline: documents -> chunks -> embeddings -> FAISS + BM25 on disk."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.config import Settings
from app.ingestion.chunker import chunk_document
from app.ingestion.loaders import load_corpus
from app.models import Chunk, Document
from app.observability import log_event, timed


def _write_jsonl(path: Path, chunks: list[Chunk]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(chunk.model_dump_json() + "\n")


def read_chunks(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(Chunk.model_validate_json(line))
    return chunks


class IngestionPipeline:
    def __init__(self, settings: Settings, embedder) -> None:
        self.settings = settings
        self.embedder = embedder

    def run(self, docs_dir: Path | None = None) -> dict:
        from app.retrieval.bm25 import BM25Index
        from app.retrieval.vector_store import VectorStore

        docs_dir = docs_dir or self.settings.data_dir
        settings = self.settings

        with timed() as load_t:
            documents: list[Document] = load_corpus(docs_dir)

        if not documents:
            raise ValueError(f"No supported documents found in {docs_dir}")

        with timed() as chunk_t:
            chunks: list[Chunk] = []
            for document in documents:
                chunks.extend(
                    chunk_document(document, settings.chunk_tokens, settings.chunk_overlap)
                )

        with timed() as embed_t:
            vectors = self.embedder.encode_documents([c.text for c in chunks])

        with timed() as index_t:
            settings.index_dir.mkdir(parents=True, exist_ok=True)
            store = VectorStore(dim=int(vectors.shape[1]), index_path=settings.vectors_path)
            store.add(vectors)
            store.save()

            _write_jsonl(settings.chunks_path, chunks)

            manifest = {
                "documents": len(documents),
                "pages": sum(len(d.pages) for d in documents),
                "chunks": len(chunks),
                "dim": int(vectors.shape[1]),
                "embedder": self.embedder.name,
                "chunk_tokens": settings.chunk_tokens,
                "chunk_overlap": settings.chunk_overlap,
                "doc_list": [
                    {
                        "doc_id": d.doc_id,
                        "title": d.title,
                        "source_path": d.source_path,
                        "pages": len(d.pages),
                        "chunks": sum(1 for c in chunks if c.doc_id == d.doc_id),
                    }
                    for d in documents
                ],
            }
            settings.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            # BM25 is cheap to rebuild; we persist it only to fail fast on corruption.
            BM25Index(chunks).save(settings.index_dir / "bm25.json")

        stats = {
            "documents": len(documents),
            "pages": sum(len(d.pages) for d in documents),
            "chunks": len(chunks),
            "embedder": self.embedder.name,
            "dim": int(vectors.shape[1]),
            "load_ms": load_t["ms"],
            "chunk_ms": chunk_t["ms"],
            "embed_ms": embed_t["ms"],
            "index_ms": index_t["ms"],
            "vectors": np.asarray(vectors).shape[0],
        }
        log_event("ingest_complete", **stats)
        return stats
