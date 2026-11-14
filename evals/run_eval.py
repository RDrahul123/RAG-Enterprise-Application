"""Retrieval + answer evaluation with ablations.

    python evals/run_eval.py --k 5

Produces the table a reviewer actually wants to see: what each stage buys you.
Writes ``evals/results.json`` so the numbers in the README stay reproducible.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.retrieval.pipeline import RetrievalPipeline  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden_set.jsonl"
MODES = ["dense", "bm25", "hybrid", "hybrid+rerank"]


def load_golden(limit: int | None = None) -> list[dict]:
    items = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    return items[:limit] if limit else items


def _rrf_fuse(dense: list[tuple[int, float]], bm25: list[tuple[int, float]], k: int) -> list[int]:
    scores: dict[int, float] = {}
    for rank, (idx, _) in enumerate(dense, start=1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (60 + rank)
    for rank, (idx, _) in enumerate(bm25, start=1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (60 + rank)
    return [i for i, _ in sorted(scores.items(), key=lambda p: p[1], reverse=True)][:k]


def retrieve(pipeline: RetrievalPipeline, question: str, mode: str, k: int) -> list[int]:
    """Return chunk indices for a given ablation mode."""
    s = pipeline.settings
    if mode == "hybrid+rerank":
        results, _ = pipeline.retrieve(question, top_k=k)
        ids = [r.chunk.chunk_id for r in results]
        return [next(i for i, c in enumerate(pipeline.chunks) if c.chunk_id == cid) for cid in ids]

    dense = pipeline.store.search(pipeline.embedder.encode_query(question), s.dense_top_k)
    bm25 = pipeline.bm25.search(question, s.bm25_top_k)

    if mode == "dense":
        return [i for i, _ in dense[:k]]
    if mode == "bm25":
        return [i for i, _ in bm25[:k]]
    if mode == "hybrid":
        return _rrf_fuse(dense, bm25, k)
    raise ValueError(mode)


def is_hit(chunk, item: dict) -> bool:
    return chunk.source_path == item["doc"] and chunk.page_number == item["page"]


def keyword_hit(answer: str, expectations: list[str]) -> bool:
    lowered = answer.lower().replace(",", "").replace(" ", "")
    return any(e.lower().replace(",", "").replace(" ", "") in lowered for e in expectations)


def evaluate(
    pipeline: RetrievalPipeline,
    golden: list[dict],
    k: int,
    run_answers: bool,
    provider_override: str | None = None,
) -> dict:
    table: dict[str, dict] = {}

    for mode in MODES:
        hits = 0
        reciprocal = 0.0
        latencies: list[float] = []

        for item in golden:
            start = time.perf_counter()
            indices = retrieve(pipeline, item["question"], mode, k)
            latencies.append((time.perf_counter() - start) * 1000)

            rank = next(
                (r for r, i in enumerate(indices, start=1) if is_hit(pipeline.chunks[i], item)),
                None,
            )
            if rank is not None:
                hits += 1
                reciprocal += 1.0 / rank

        n = len(golden)
        table[mode] = {
            "recall_at_k": round(hits / n, 4) if n else 0.0,
            "mrr": round(reciprocal / n, 4) if n else 0.0,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        }

    answer_stats = {}
    if run_answers:
        from app.generation.providers import build_provider
        from app.generation.rag import RagEngine

        eval_settings = (
            replace(pipeline.settings, llm_provider=provider_override)
            if provider_override
            else pipeline.settings
        )
        engine = RagEngine(pipeline, eval_settings, build_provider(eval_settings))
        correct = 0
        grounded = 0
        cite_correct = 0
        latencies = []

        for item in golden:
            response = engine.answer(item["question"], top_k=k, include_trace=False)
            latencies.append(response.latency_ms)
            if keyword_hit(response.answer, item["expect"]):
                correct += 1
            if response.grounded:
                grounded += 1
            if any(
                c.source_path == item["doc"] and c.page_number == item["page"]
                for c in response.citations
            ):
                cite_correct += 1

        n = len(golden)
        answer_stats = {
            "answer_accuracy": round(correct / n, 4),
            "grounded_rate": round(grounded / n, 4),
            "citation_page_accuracy": round(cite_correct / n, 4),
            "avg_answer_latency_ms": round(sum(latencies) / n, 2),
            "provider": engine.provider.name,
            "model": engine.provider.model,
        }

    return {"k": k, "n": len(golden), "retrieval": table, "answers": answer_stats}


def render(report: dict) -> str:
    lines = [
        f"Retrieval ablation (n={report['n']} questions, k={report['k']})",
        "",
        f"{'mode':<16}{'recall@k':>10}{'MRR':>10}{'latency':>12}",
        "-" * 48,
    ]
    for mode, rows in report["retrieval"].items():
        lines.append(
            f"{mode:<16}{rows['recall_at_k']:>10.3f}{rows['mrr']:>10.3f}"
            f"{rows['avg_latency_ms']:>10.1f}ms"
        )
    if report["answers"]:
        a = report["answers"]
        lines += [
            "",
            f"Answers ({a['provider']} / {a['model']})",
            "-" * 48,
            f"  answer accuracy        {a['answer_accuracy']:.3f}",
            f"  grounded rate          {a['grounded_rate']:.3f}",
            f"  citation page accuracy {a['citation_page_accuracy']:.3f}",
            f"  mean answer latency    {a['avg_answer_latency_ms']:.0f} ms",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation suite.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-answers", action="store_true", help="Retrieval metrics only (fast).")
    parser.add_argument(
        "--provider",
        default="extractive",
        choices=["extractive", "local", "hf_api", "auto"],
        help="LLM backend for the answer phase. Defaults to extractive for reproducibility.",
    )
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results.json")
    args = parser.parse_args()

    settings = get_settings()
    pipeline = RetrievalPipeline.from_disk(settings)
    golden = load_golden(args.limit)

    report = evaluate(
        pipeline,
        golden,
        args.k,
        run_answers=not args.no_answers,
        provider_override=args.provider,
    )
    print(render(report))

    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
