"""Shared fixtures. Models load once for the whole test session."""

from __future__ import annotations

import pytest

import rag


@pytest.fixture(scope="session")
def store() -> rag.RAG:
    if not any(rag.DOCS_DIR.glob("*.pdf")):
        import make_corpus

        make_corpus.main()

    instance = rag.RAG()
    if not instance.chunks:
        instance.ingest()
    return instance
