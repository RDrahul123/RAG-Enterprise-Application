from __future__ import annotations

import rag

POLICY = (
    "The refund window is thirty days from delivery. "
    "Customers must keep the original packaging. "
    "Shipping costs are refunded only for faulty items. "
    "Refunds are processed within five business days. "
    "Gift cards are excluded from refunds."
)


def test_chunk_page_splits_long_text():
    chunks = rag.chunk_page(POLICY, max_tokens=20, overlap=5)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


def test_chunk_page_keeps_all_sentences():
    combined = " ".join(rag.chunk_page(POLICY, max_tokens=12, overlap=0))
    for sentence in POLICY.split(". "):
        assert sentence.split()[0] in combined


def test_empty_page_produces_no_chunks():
    assert rag.chunk_page("   \n  ") == []


def test_tokenize_drops_stopwords():
    assert rag.tokenize("What is the refund window?") == ["refund", "window"]


def test_search_finds_the_right_page(store):
    hits = store.search("How many days of paid annual leave do employees receive?")
    assert hits
    assert hits[0].chunk.doc == "Northwind Employee Handbook"
    assert hits[0].chunk.page == 2


def test_dense_and_lexical_both_return_results(store):
    assert store.dense_hits("parental leave")
    # BM25 is exact-match, so query with words that actually appear in the text.
    assert store.bm25_hits("parental leave")
