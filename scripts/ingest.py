"""Build (or rebuild) the index from everything in data/docs.

    python scripts/ingest.py [--docs-dir PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.service import RagService  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the vector index.")
    parser.add_argument("--docs-dir", type=Path, default=None, help="Override the docs directory.")
    args = parser.parse_args()

    settings = get_settings()
    service = RagService(settings)
    result = service.ingest(args.docs_dir)

    print("\n=== Ingestion complete ===")
    print(f"  documents : {result.documents}")
    print(f"  pages     : {result.pages}")
    print(f"  chunks    : {result.chunks}")
    print(f"  embedder  : {result.embedder} (dim {result.dim})")
    print(f"  elapsed   : {result.elapsed_ms:.0f} ms")
    print(f"  index     : {settings.index_dir}")


if __name__ == "__main__":
    main()
