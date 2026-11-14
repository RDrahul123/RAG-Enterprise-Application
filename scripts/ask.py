"""Ask a question from the terminal.

    python scripts/ask.py "How many days of annual leave do employees get?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.service import RagService  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the RAG index a question.")
    parser.add_argument("question", help="Natural-language question.")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--retrieved", action="store_true", help="Show retrieved chunks.")
    parser.add_argument("--trace", action="store_true", help="Show per-stage timings.")
    args = parser.parse_args()

    service = RagService()
    if not service.index_ready:
        raise SystemExit("No index found. Run `python scripts/ingest.py` first.")

    response = service.ask(args.question, top_k=args.top_k)

    print(f"\nQ: {args.question}\n")
    print(f"A: {response.answer}\n")
    print(
        f"provider={response.provider} model={response.model} "
        f"grounded={response.grounded} score={response.grounding_score} "
        f"latency={response.latency_ms:.0f}ms"
    )

    if response.citations:
        print("\nSources:")
        for citation in response.citations:
            print(f"  [{citation.index}] {citation.doc_title} — p.{citation.page_number}")

    if args.retrieved:
        print("\nRetrieved:")
        for result in response.retrieved:
            chunk = result.chunk
            print(
                f"  {result.rerank_score if result.rerank_score is not None else result.score:7.3f}  "
                f"dense#{result.dense_rank or '-'} bm25#{result.bm25_rank or '-'}  "
                f"{chunk.doc_title} p.{chunk.page_number}"
            )

    if args.trace:
        print("\nTrace:")
        for stage in response.trace:
            print(f"  {stage.name:10s} {stage.ms:8.1f} ms  {stage.detail}")


if __name__ == "__main__":
    main()
