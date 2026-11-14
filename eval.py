"""Evaluate retrieval and answers against eval_set.jsonl.

    python eval.py

Reports what each retrieval stage contributes (the ablation), then how often the
answers are correct, cited, and pointed at the right page.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import answer
from rag import FINAL_TOP_K, RAG

EVAL_SET = Path(__file__).parent / "eval_set.jsonl"
MODES = ["dense", "bm25", "hybrid", "rerank"]


def load_eval_set() -> list[dict]:
    return [json.loads(line) for line in EVAL_SET.read_text().splitlines() if line.strip()]


def ranked(rag: RAG, question: str, mode: str, k: int) -> list[int]:
    """Return chunk indices for one configuration."""
    if mode == "rerank":
        return [rag.chunks.index(hit.chunk) for hit in rag.search(question, k)]

    dense = [index for index, _ in rag.dense_hits(question)]
    if mode == "dense":
        return dense[:k]

    lexical = [index for index, _ in rag.bm25_hits(question)]
    if mode == "bm25":
        return lexical[:k]

    return [index for index, _ in rag.fuse(rag.dense_hits(question), rag.bm25_hits(question))][:k]


def is_hit(chunk, row: dict) -> bool:
    return chunk.doc == row["doc"] and chunk.page == row["page"]


def retrieval_table(rag: RAG, rows: list[dict], k: int) -> dict:
    table = {}
    for mode in MODES:
        hits, reciprocal, timings = 0, 0.0, []
        for row in rows:
            started = time.perf_counter()
            indices = ranked(rag, row["question"], mode, k)
            timings.append((time.perf_counter() - started) * 1000)
            rank = next((r for r, i in enumerate(indices, 1) if is_hit(rag.chunks[i], row)), None)
            if rank:
                hits += 1
                reciprocal += 1 / rank
        table[mode] = {
            "recall": round(hits / len(rows), 3),
            "mrr": round(reciprocal / len(rows), 3),
            "ms": round(sum(timings) / len(timings), 1),
        }
    return table


def answer_table(rag: RAG, rows: list[dict], k: int) -> dict:
    correct = cited_page = grounded = 0
    timings = []
    for row in rows:
        result = answer.answer(rag, row["question"], k)
        timings.append(result["retrieval_ms"] + result["generation_ms"])
        text = result["answer"].lower().replace(",", "")
        if any(e.lower().replace(",", "") in text for e in row["expect"]):
            correct += 1
        if any(c["doc"] == row["doc"] and c["page"] == row["page"] for c in result["citations"]):
            cited_page += 1
        if result["grounded"]:
            grounded += 1
    return {
        "answer_accuracy": round(correct / len(rows), 3),
        "grounded_rate": round(grounded / len(rows), 3),
        "citation_page_accuracy": round(cited_page / len(rows), 3),
        "mean_ms": round(sum(timings) / len(timings)),
    }


def main() -> None:
    rag = RAG()
    rows = load_eval_set()

    retrieval = retrieval_table(rag, rows, FINAL_TOP_K)
    print(f"Retrieval (n={len(rows)}, k={FINAL_TOP_K})\n")
    print(f"{'mode':<10}{'recall':>9}{'MRR':>9}{'latency':>11}")
    print("-" * 39)
    for mode, row in retrieval.items():
        print(f"{mode:<10}{row['recall']:>9.3f}{row['mrr']:>9.3f}{row['ms']:>9.1f}ms")

    answers = answer_table(rag, rows, FINAL_TOP_K)
    print(f"\nAnswers ({'local LLM' if answer.USE_LLM else 'extractive'})\n")
    for key, value in answers.items():
        print(f"  {key:<24}{value}")

    print("\nwrote results.json")
    Path("results.json").write_text(json.dumps({"retrieval": retrieval, "answers": answers}, indent=2))


if __name__ == "__main__":
    main()
