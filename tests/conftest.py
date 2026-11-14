"""Shared fixtures. Models load once for the whole test session.

The corpus and the index are prepared at *import* time, not inside a fixture,
because ``app.py`` builds its ``RAG`` object when the module is imported — which
happens before any fixture runs. On a fresh clone `data/docs` is empty and
`data/index` does not exist (both are gitignored), so this is what makes the
suite self-contained.
"""

from __future__ import annotations

import pytest

import rag


def prepare_corpus() -> None:
    if not any(rag.DOCS_DIR.glob("*.pdf")):
        import make_corpus

        make_corpus.main()
    if not (rag.INDEX_DIR / "chunks.json").exists():
        rag.RAG().ingest()


prepare_corpus()


@pytest.fixture(scope="session")
def store() -> rag.RAG:
    return rag.RAG()
