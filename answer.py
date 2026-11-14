"""Generation: turn retrieved chunks into an answer that cites its sources.

Two ways to answer:

* ``llm_answer``        – a small local Hugging Face model (Qwen2.5-1.5B).
* ``extractive_answer`` – no model: quote the most relevant sentences.

Both produce ``[n]`` markers. Everything after that (validating the markers,
mapping them to pages, scoring how well grounded the answer is) is identical,
and happens in Python rather than being trusted to the model.
"""

from __future__ import annotations

import os
import re
import time

from rag import FINAL_TOP_K, MIN_SCORE, RAG, STOPWORDS, Hit, SENTENCE, stem

LLM_MODEL = os.getenv("RAG_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
USE_LLM = os.getenv("RAG_USE_LLM", "1") != "0"

REFUSAL = "I don't have enough information in the provided documents to answer that."

SYSTEM_PROMPT = (
    "You are an enterprise document assistant. Answer strictly from the numbered sources.\n"
    "Rules:\n"
    "1. Every sentence that states a fact must end with the number of the source that contains "
    "it, written in square brackets, for example [2].\n"
    "2. Cite the source that actually contains the fact. Do not default to [1].\n"
    "3. Never use outside knowledge.\n"
    f"4. If the sources do not contain the answer, reply exactly: \"{REFUSAL}\"\n"
    "5. Answer in one to three short sentences. No preamble and do not restate the question."
)

CITATION = re.compile(r"\[(\d+)\]")
# Models often write "Source 3" or "(Source 3)" instead of "[3]".
PAREN_SOURCE = re.compile(r"\(\s*source\s*\[?(\d{1,2})\]?\s*\)", re.IGNORECASE)
SOURCE_WORD = re.compile(r"\bsource\s*\[?(\d{1,2})\]?", re.IGNORECASE)
WORD = re.compile(r"[a-z0-9]+")
# "It is 30 days. [1]" must count as one cited sentence, not two fragments.
TRAILING_CITATION = re.compile(r"([.!?])\s+(?=\[\d+\])")


# --- prompt ---------------------------------------------------------------
def build_prompt(question: str, hits: list[Hit]) -> str:
    sources = []
    for number, hit in enumerate(hits, start=1):
        text = hit.chunk.text[:1200]
        sources.append(f"[{number}] {hit.chunk.doc}, page {hit.chunk.page}\n{text}")
    return (
        "SOURCES:\n\n" + "\n\n".join(sources) + f"\n\nQUESTION: {question}\n\n"
        "Answer using only the sources above and cite them with [n]."
    )


# --- the two answer styles ------------------------------------------------
_llm = None


def llm_answer(question: str, hits: list[Hit]) -> str:
    """Ask the local model. Raises if transformers/the model is unavailable."""
    global _llm
    if _llm is None:
        from transformers import pipeline

        _llm = pipeline("text-generation", model=LLM_MODEL, dtype="auto")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(question, hits)},
    ]
    output = _llm(messages, max_new_tokens=512, do_sample=False, return_full_text=False)
    return output[0]["generated_text"].strip()


def extractive_answer(question: str, hits: list[Hit]) -> str:
    """No model: pick the sentences that share the most words with the question."""
    terms = {stem(w) for w in WORD.findall(question.lower()) if w not in STOPWORDS and len(w) > 2}
    candidates: list[tuple[int, int, str]] = []

    for number, hit in enumerate(hits, start=1):
        for sentence in SENTENCE.split(hit.chunk.text):
            sentence = sentence.strip()
            if len(sentence) < 30:
                continue
            words = {stem(w) for w in WORD.findall(sentence.lower())}
            overlap = len(terms & words)
            if overlap:
                candidates.append((overlap, number, sentence))

    if not candidates:
        return REFUSAL

    best = sorted(candidates, reverse=True)[:3]
    return "Most relevant passages:\n\n" + "\n".join(f"- {text} [{number}]" for _, number, text in best)


# --- citations ------------------------------------------------------------
def normalize_citations(text: str) -> str:
    """Rewrite "Source 3" / "(Source 3)" to "[3]" so citations are consistent."""
    text = PAREN_SOURCE.sub(r"[\1]", text)
    return SOURCE_WORD.sub(r"[\1]", text)


def parse_citations(text: str, hits: list[Hit]) -> list[dict]:
    """Keep only [n] markers that point at a real retrieved chunk."""
    found: list[dict] = []
    seen: set[int] = set()
    for marker in CITATION.findall(text):
        number = int(marker)
        if number in seen or not 1 <= number <= len(hits):
            continue
        seen.add(number)
        chunk = hits[number - 1].chunk
        found.append(
            {
                "index": number,
                "doc": chunk.doc,
                "page": chunk.page,
                "snippet": chunk.text[:300],
            }
        )
    return found


def grounding_score(text: str, n_hits: int) -> float:
    """Fraction of answer sentences that cite a valid source."""
    text = TRAILING_CITATION.sub(r"\1", text)
    sentences = [s for s in SENTENCE.split(text) if len(s.strip()) > 20]
    if not sentences:
        return 0.0
    cited = sum(1 for s in sentences if any(1 <= int(n) <= n_hits for n in CITATION.findall(s)))
    return round(cited / len(sentences), 2)


# --- the one function callers need ----------------------------------------
def answer(rag: RAG, question: str, top_k: int = FINAL_TOP_K) -> dict:
    started = time.perf_counter()
    hits = rag.search(question, top_k)
    retrieval_ms = round((time.perf_counter() - started) * 1000)

    if not hits or hits[0].score < MIN_SCORE:
        return {
            "answer": REFUSAL,
            "citations": [],
            "grounded": False,
            "grounding_score": 0.0,
            "provider": "refused",
            "sources": [],
            "retrieval_ms": retrieval_ms,
            "generation_ms": 0,
        }

    started = time.perf_counter()
    text, provider = "", "extractive"
    if USE_LLM:
        try:
            text, provider = llm_answer(question, hits), "local"
        except Exception:  # model not downloaded / no memory: fall back
            text = ""
    if not text:
        text, provider = extractive_answer(question, hits), "extractive"
    text = normalize_citations(text)
    generation_ms = round((time.perf_counter() - started) * 1000)

    citations = parse_citations(text, hits)
    score = grounding_score(text, len(hits))

    return {
        "answer": text,
        "citations": citations,
        "grounded": bool(citations) and score >= 0.5 and REFUSAL not in text,
        "grounding_score": score,
        "provider": provider,
        "sources": [
            {
                "doc": hit.chunk.doc,
                "page": hit.chunk.page,
                "text": hit.chunk.text[:300],
                "score": round(hit.score, 2),
                "dense_rank": hit.dense_rank,
                "bm25_rank": hit.bm25_rank,
            }
            for hit in hits
        ],
        "retrieval_ms": retrieval_ms,
        "generation_ms": generation_ms,
    }
