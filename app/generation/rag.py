"""Grounded answer generation with enforced citations.

The contract this module enforces:

* No retrieved context above the relevance threshold -> refuse, don't hallucinate.
* Every claim should carry a ``[n]`` marker that maps to a real chunk and page.
* ``grounding_score`` reports the fraction of answer sentences that cite a source,
  so a reviewer can see at a glance whether the answer is actually supported.
"""

from __future__ import annotations

import re

from app.config import Settings
from app.generation.providers import BaseProvider, ExtractiveProvider
from app.models import AskResponse, Citation, ScoredChunk, StageTrace
from app.observability import log_event, metrics, timed
from app.retrieval.pipeline import RetrievalPipeline

_CITE = re.compile(r"\[(\d+)\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_TERM = re.compile(r"[a-z0-9]+")

SYSTEM_PROMPT = (
    "You are an enterprise document assistant. Answer strictly from the provided "
    "sources. Rules:\n"
    "1. Cite the source number in square brackets after every factual claim, e.g. [2].\n"
    "2. Use only the sources. Never use outside knowledge.\n"
    "3. If the sources do not contain the answer, say exactly: "
    "\"I don't have enough information in the provided documents to answer that.\"\n"
    "4. Be concise and specific. Prefer numbers, names and dates from the sources.\n"
    "5. Do not invent source numbers that are not in the list."
)

REFUSAL = "I don't have enough information in the provided documents to answer that."

_STOP = frozenset(
    """a an and are as at be by can could did do does for from had has have how i if
    in into is it its may might of on or our so that the their them then there these
    they this to was we were what when where which who why will with would you your""".split()
)


def _truncate(text: str, limit: int = 1200) -> str:
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


def build_prompt(question: str, results: list[ScoredChunk]) -> str:
    blocks: list[str] = []
    for i, result in enumerate(results, start=1):
        chunk = result.chunk
        blocks.append(
            f"[{i}] {chunk.doc_title} — page {chunk.page_number} ({chunk.source_path})\n"
            f"{_truncate(chunk.text)}"
        )
    sources = "\n\n".join(blocks)
    return (
        f"SOURCES:\n{sources}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer using only the sources above and cite them with [n]."
    )


def extractive_answer(question: str, results: list[ScoredChunk], max_sentences: int = 3) -> str:
    """Deterministic, zero-hallucination baseline used when no LLM is configured."""
    terms = {t for t in _TERM.findall(question.lower()) if t not in _STOP and len(t) > 2}
    candidates: list[tuple[float, int, str]] = []

    for i, result in enumerate(results, start=1):
        for sentence in _SENTENCE.split(result.chunk.text):
            sentence = sentence.strip()
            if len(sentence) < 30:
                continue
            tokens = set(_TERM.findall(sentence.lower()))
            overlap = len(terms & tokens) / max(1, len(terms))
            if overlap > 0:
                candidates.append((overlap + result.score / 100.0, i, sentence))

    if not candidates:
        return REFUSAL

    seen: set[str] = set()
    chosen: list[tuple[int, str]] = []
    for _, index, sentence in sorted(candidates, key=lambda c: c[0], reverse=True):
        key = sentence[:80].lower()
        if key in seen:
            continue
        seen.add(key)
        chosen.append((index, sentence))
        if len(chosen) >= max_sentences:
            break

    preamble = "Most relevant passages from the indexed documents:\n\n"
    body = "\n".join(f"- {sentence} [{index}]" for index, sentence in chosen)
    return preamble + body


def grounding_score(answer: str, max_index: int) -> float:
    sentences = [s for s in _SENTENCE.split(answer) if len(s.strip()) > 20]
    if not sentences:
        return 0.0
    cited = 0
    for sentence in sentences:
        if any(1 <= int(n) <= max_index for n in _CITE.findall(sentence)):
            cited += 1
    return round(cited / len(sentences), 3)


class RagEngine:
    def __init__(self, pipeline: RetrievalPipeline, settings: Settings, provider: BaseProvider) -> None:
        self.pipeline = pipeline
        self.settings = settings
        self.provider = provider

    @property
    def _guard_active(self) -> bool:
        return "cross-encoder" in self.pipeline.reranker.name

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        doc_ids: list[str] | None = None,
        include_trace: bool = True,
    ) -> AskResponse:
        settings = self.settings
        trace: list[StageTrace] = []

        results, retrieval_trace = self.pipeline.retrieve(question, top_k=top_k, doc_ids=doc_ids)
        trace.extend(retrieval_trace)

        best = results[0].score if results else None
        refuse = not results or (self._guard_active and best is not None and best < settings.min_rerank_score)

        if refuse:
            metrics.inc("requests_total", outcome="refused")
            log_event("answer_refused", query=question[:120], best_score=best)
            trace.append(
                StageTrace(
                    name="guard",
                    ms=0.0,
                    detail={"refused": True, "best_score": best, "threshold": settings.min_rerank_score},
                )
            )
            return AskResponse(
                answer=REFUSAL,
                citations=[],
                grounded=False,
                grounding_score=0.0,
                provider=self.provider.name,
                model=self.provider.model,
                latency_ms=round(sum(t.ms for t in trace), 2),
                trace=trace if include_trace else [],
                retrieved=results,
            )

        with timed() as gen_t:
            if isinstance(self.provider, ExtractiveProvider) or not self.provider.ready:
                text = extractive_answer(question, results)
                provider_name, model_name = "extractive", "none"
            else:
                prompt = build_prompt(question, results)
                result = self.provider.generate(SYSTEM_PROMPT, prompt)
                text = result.text
                provider_name, model_name = result.provider, result.model

        trace.append(StageTrace(name="generate", ms=gen_t["ms"], detail={"provider": provider_name}))

        citations, cited_indices = self._build_citations(text, results)
        score = grounding_score(text, len(results))
        grounded = bool(citations) and score >= 0.5 and REFUSAL not in text

        latency = round(sum(t.ms for t in trace), 2)
        metrics.inc("requests_total", outcome="answered")
        metrics.observe_ms("request_latency", latency, provider=provider_name)
        log_event(
            "answer",
            query=question[:120],
            provider=provider_name,
            citations=len(citations),
            grounding=score,
            latency_ms=latency,
        )

        return AskResponse(
            answer=text.strip(),
            citations=citations,
            grounded=grounded,
            grounding_score=score,
            provider=provider_name,
            model=model_name,
            latency_ms=latency,
            trace=trace if include_trace else [],
            retrieved=results,
        )

    @staticmethod
    def _build_citations(text: str, results: list[ScoredChunk]) -> tuple[list[Citation], list[int]]:
        citations: list[Citation] = []
        seen: set[int] = set()
        for match in _CITE.findall(text):
            index = int(match)
            if index in seen or not 1 <= index <= len(results):
                continue
            seen.add(index)
            chunk = results[index - 1].chunk
            citations.append(
                Citation(
                    index=index,
                    doc_id=chunk.doc_id,
                    doc_title=chunk.doc_title,
                    source_path=chunk.source_path,
                    page_number=chunk.page_number,
                    snippet=_truncate(chunk.text, 320),
                    chunk_id=chunk.chunk_id,
                )
            )
        return citations, sorted(seen)
