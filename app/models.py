"""Pydantic schemas shared by ingestion, retrieval, generation and the API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Page(BaseModel):
    """A single page (or logical section) of a source document."""

    page_number: int
    text: str


class Document(BaseModel):
    doc_id: str
    title: str
    source_path: str
    doc_type: Literal["pdf", "markdown", "text"] = "pdf"
    pages: list[Page] = Field(default_factory=list)

    @property
    def char_count(self) -> int:
        return sum(len(p.text) for p in self.pages)


class Chunk(BaseModel):
    """The unit that actually gets embedded and cited."""

    chunk_id: str
    doc_id: str
    doc_title: str
    source_path: str
    page_number: int
    position: int  # order of the chunk inside its page
    text: str
    token_count: int


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    dense_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rerank_rank: int | None = None


class Citation(BaseModel):
    index: int
    doc_id: str
    doc_title: str
    source_path: str
    page_number: int
    snippet: str
    chunk_id: str


class StageTrace(BaseModel):
    name: str
    ms: float
    detail: dict[str, Any] = Field(default_factory=dict)


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    doc_ids: list[str] | None = None
    include_trace: bool = True


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool = False
    grounding_score: float = 0.0
    provider: str = ""
    model: str = ""
    latency_ms: float = 0.0
    trace: list[StageTrace] = Field(default_factory=list)
    retrieved: list[ScoredChunk] = Field(default_factory=list)


class IngestResponse(BaseModel):
    documents: int
    pages: int
    chunks: int
    embedder: str
    dim: int
    elapsed_ms: float


class DocumentSummary(BaseModel):
    doc_id: str
    title: str
    source_path: str
    pages: int
    chunks: int


class StatsResponse(BaseModel):
    indexed: bool
    documents: int
    chunks: int
    dim: int
    embedder: str
    reranker: str
    llm_provider: str
    llm_model: str
    doc_list: list[DocumentSummary] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    index_ready: bool
    provider: str
    provider_ready: bool
    detail: dict[str, Any] = Field(default_factory=dict)
