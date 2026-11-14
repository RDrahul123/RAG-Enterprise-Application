"""Hybrid retrieval: dense + BM25 -> Reciprocal Rank Fusion -> cross-encoder rerank.

Each stage exists for a reason, and the eval harness measures the marginal gain of
each one (see ``evals/run_eval.py``).
"""

from __future__ import annotations

import json

from app.config import Settings
from app.ingestion.pipeline import read_chunks
from app.models import Chunk, ScoredChunk, StageTrace
from app.observability import log_event, timed
from app.retrieval.bm25 import BM25Index
from app.retrieval.embeddings import BaseEmbedder, get_embedder
from app.retrieval.reranker import BaseReranker, get_reranker
from app.retrieval.vector_store import VectorStore


class RetrievalPipeline:
    def __init__(
        self,
        settings: Settings,
        embedder: BaseEmbedder,
        store: VectorStore,
        bm25: BM25Index,
        reranker: BaseReranker,
        chunks: list[Chunk],
        manifest: dict,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.bm25 = bm25
        self.reranker = reranker
        self.chunks = chunks
        self.manifest = manifest

    # -- construction ---------------------------------------------------
    @classmethod
    def from_disk(cls, settings: Settings, prefer_real: bool | None = None) -> "RetrievalPipeline":
        if prefer_real is None:
            prefer_real = settings.prefer_real
        if not settings.manifest_path.exists() or not settings.chunks_path.exists():
            raise FileNotFoundError(
                "No index found. Run `make ingest` (or POST /api/ingest) first."
            )
        manifest = json.loads(settings.manifest_path.read_text(encoding="utf-8"))
        chunks = read_chunks(settings.chunks_path)
        store = VectorStore.load(dim=int(manifest["dim"]), index_path=settings.vectors_path)
        if store.size != len(chunks):
            raise RuntimeError(
                f"Index/chunk mismatch: {store.size} vectors vs {len(chunks)} chunks. Re-ingest."
            )
        embedder = get_embedder(settings, prefer_real=prefer_real)
        return cls(
            settings=settings,
            embedder=embedder,
            store=store,
            bm25=BM25Index(chunks),
            reranker=get_reranker(settings, prefer_real=prefer_real),
            chunks=chunks,
            manifest=manifest,
        )

    # -- retrieval ------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        doc_ids: list[str] | None = None,
    ) -> tuple[list[ScoredChunk], list[StageTrace]]:
        settings = self.settings
        top_k = top_k or settings.final_top_k
        allowed = set(doc_ids) if doc_ids else None
        trace: list[StageTrace] = []

        def permitted(index: int) -> bool:
            return allowed is None or self.chunks[index].doc_id in allowed

        # 1. Dense -------------------------------------------------------
        with timed() as t:
            query_vector = self.embedder.encode_query(query)
            dense_hits = [
                (i, s) for i, s in self.store.search(query_vector, settings.dense_top_k * 3)
                if permitted(i)
            ][: settings.dense_top_k]
        dense_rank = {i: r + 1 for r, (i, _) in enumerate(dense_hits)}
        dense_score = {i: s for i, s in dense_hits}
        trace.append(StageTrace(name="dense", ms=t["ms"], detail={"candidates": len(dense_hits)}))

        # 2. BM25 --------------------------------------------------------
        with timed() as t:
            bm25_hits = [
                (i, s) for i, s in self.bm25.search(query, settings.bm25_top_k * 3)
                if permitted(i)
            ][: settings.bm25_top_k]
        bm25_rank = {i: r + 1 for r, (i, _) in enumerate(bm25_hits)}
        bm25_score = {i: s for i, s in bm25_hits}
        trace.append(StageTrace(name="bm25", ms=t["ms"], detail={"candidates": len(bm25_hits)}))

        # 3. Reciprocal Rank Fusion --------------------------------------
        with timed() as t:
            fused: dict[int, float] = {}
            for i in dense_rank:
                fused[i] = fused.get(i, 0.0) + 1.0 / (settings.rrf_k + dense_rank[i])
            for i in bm25_rank:
                fused[i] = fused.get(i, 0.0) + 1.0 / (settings.rrf_k + bm25_rank[i])
            fused_order = sorted(fused, key=lambda i: fused[i], reverse=True)
            rerank_pool = fused_order[: settings.rerank_top_k]
        trace.append(
            StageTrace(
                name="fusion",
                ms=t["ms"],
                detail={
                    "unique_candidates": len(fused),
                    "overlap": len(set(dense_rank) & set(bm25_rank)),
                    "rerank_pool": len(rerank_pool),
                },
            )
        )

        # 4. Cross-encoder rerank ----------------------------------------
        with timed() as t:
            rerank_scores = self.reranker.rerank(query, [self.chunks[i].text for i in rerank_pool])
            scored = sorted(zip(rerank_pool, rerank_scores), key=lambda p: p[1], reverse=True)
            rerank_rank = {i: r + 1 for r, (i, _) in enumerate(scored)}
        trace.append(StageTrace(name="rerank", ms=t["ms"], detail={"pool": len(rerank_pool)}))

        # 5. Assemble ----------------------------------------------------
        results: list[ScoredChunk] = []
        for i, score in scored[:top_k]:
            results.append(
                ScoredChunk(
                    chunk=self.chunks[i],
                    score=float(score),
                    dense_score=dense_score.get(i),
                    bm25_score=bm25_score.get(i),
                    rerank_score=float(score),
                    dense_rank=dense_rank.get(i),
                    bm25_rank=bm25_rank.get(i),
                    rerank_rank=rerank_rank.get(i),
                )
            )

        log_event(
            "retrieval",
            query=query[:120],
            dense=len(dense_hits),
            bm25=len(bm25_hits),
            fused=len(fused),
            final=len(results),
            top_score=round(results[0].score, 3) if results else None,
        )
        return results, trace

    # -- introspection ---------------------------------------------------
    def describe(self) -> dict:
        return {
            "documents": int(self.manifest.get("documents", 0)),
            "chunks": len(self.chunks),
            "dim": int(self.manifest.get("dim", 0)),
            "embedder": self.embedder.name,
            "reranker": self.reranker.name,
            "vector_backend": self.store.backend,
            "doc_list": self.manifest.get("doc_list", []),
        }
