"""Sentence-aware, page-scoped chunking.

Two rules that make retrieval and citations behave:

1. Never merge across pages -> a chunk always has one unambiguous page number.
2. Split on sentence boundaries where possible -> a chunk reads like prose, not
   like a bag of fragments, which matters a lot for reranking quality.
"""

from __future__ import annotations

import re

from app.models import Chunk, Document

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def approx_tokens(text: str) -> int:
    """Whitespace token count. Cheap, deterministic and close enough to BPE for budget control."""
    return len(text.split())


def _hard_split(sentence: str, max_tokens: int) -> list[str]:
    """Break a single monster sentence (tables, code blobs) on word boundaries."""
    words = sentence.split()
    if len(words) <= max_tokens:
        return [sentence]
    return [" ".join(words[i : i + max_tokens]) for i in range(0, len(words), max_tokens)]


def split_sentences(text: str, max_tokens: int) -> list[str]:
    sentences: list[str] = []
    for raw in _SENTENCE_SPLIT.split(text):
        piece = raw.strip()
        if not piece:
            continue
        sentences.extend(_hard_split(piece, max_tokens))
    return sentences


def chunk_page(
    text: str,
    max_tokens: int = 320,
    overlap_tokens: int = 60,
) -> list[str]:
    """Greedy sentence packing with a token overlap window."""
    sentences = split_sentences(text, max_tokens)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        tokens = approx_tokens(sentence)

        if current and current_tokens + tokens > max_tokens:
            chunks.append(" ".join(current))

            # Carry the tail of the window forward for context continuity.
            carry: list[str] = []
            carry_tokens = 0
            for previous in reversed(current):
                previous_tokens = approx_tokens(previous)
                if carry_tokens + previous_tokens > overlap_tokens:
                    break
                carry.insert(0, previous)
                carry_tokens += previous_tokens
            current = carry
            current_tokens = carry_tokens

        current.append(sentence)
        current_tokens += tokens

    if current:
        chunks.append(" ".join(current))

    # Drop stubs produced purely by the overlap window.
    return [c for c in chunks if approx_tokens(c) >= 12] or chunks


def chunk_document(document: Document, max_tokens: int, overlap_tokens: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in document.pages:
        for position, text in enumerate(chunk_page(page.text, max_tokens, overlap_tokens)):
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}:p{page.page_number}:c{position}",
                    doc_id=document.doc_id,
                    doc_title=document.title,
                    source_path=document.source_path,
                    page_number=page.page_number,
                    position=position,
                    text=text,
                    token_count=approx_tokens(text),
                )
            )
    return chunks
