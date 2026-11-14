from __future__ import annotations

from app.ingestion.chunker import approx_tokens, chunk_document, chunk_page
from app.models import Document, Page

TEXT = (
    "The refund window is thirty days from delivery. "
    "Customers must keep the original packaging. "
    "Shipping costs are refunded only for faulty items. "
    "Refunds are processed within five business days. "
    "Gift cards are excluded from refunds."
)


def test_approx_tokens_counts_words():
    assert approx_tokens("one two three") == 3
    assert approx_tokens("") == 0


def test_chunk_page_splits_when_budget_exceeded():
    chunks = chunk_page(TEXT, max_tokens=20, overlap_tokens=5)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


def test_chunk_page_covers_all_content():
    chunks = chunk_page(TEXT, max_tokens=12, overlap_tokens=0)
    combined = " ".join(chunks)
    for sentence in TEXT.split(". "):
        head = sentence.split()[0]
        assert head in combined


def test_empty_text_yields_no_chunks():
    assert chunk_page("   \n  ", max_tokens=50, overlap_tokens=0) == []


def test_chunk_document_keeps_page_numbers_and_unique_ids():
    document = Document(
        doc_id="doc-1",
        title="Policy",
        source_path="policy.pdf",
        pages=[
            Page(page_number=1, text=TEXT),
            Page(page_number=2, text=TEXT),
        ],
    )
    chunks = chunk_document(document, max_tokens=20, overlap_tokens=5)

    assert {c.page_number for c in chunks} == {1, 2}
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(c.doc_title == "Policy" for c in chunks)
    assert all(c.token_count > 0 for c in chunks)
