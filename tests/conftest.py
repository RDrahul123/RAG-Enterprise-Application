"""Shared test fixtures.

Tests run with ``prefer_real=False`` (hashed embeddings + lexical reranker) so the
suite needs zero model downloads and finishes in a couple of seconds. The HTTP and
retrieval code paths exercised are identical to production.
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.service import RagService

LEAVE_POLICY = """Acme Leave Policy

Full-time employees receive 25 days of paid annual leave each calendar year.
Leave accrues monthly. A maximum of four unused days may be carried into the
following year. Employees must request leave at least fourteen days in advance.
Sick leave is separate and provides eight paid days per year.
"""

API_GUIDE = """# Atlas API Guide

Access tokens expire after 3,600 seconds. Clients must request a new token.
The standard rate limit is 600 requests per minute per key.
Rate limited requests receive HTTP 429 and a Retry-After header.
Pagination uses a cursor and allows a maximum page size of 100 records.
"""


@pytest.fixture()
def settings(tmp_path) -> Settings:
    docs = tmp_path / "docs"
    docs.mkdir()
    index = tmp_path / "index"
    index.mkdir()

    (docs / "leave_policy.txt").write_text(LEAVE_POLICY, encoding="utf-8")
    (docs / "api_guide.md").write_text(API_GUIDE, encoding="utf-8")

    return Settings(
        data_dir=docs,
        index_dir=index,
        prefer_real=False,
        llm_provider="extractive",
        chunk_tokens=60,
        chunk_overlap=15,
    )


@pytest.fixture()
def service(settings: Settings) -> RagService:
    svc = RagService(settings)
    svc.ingest()
    return svc
