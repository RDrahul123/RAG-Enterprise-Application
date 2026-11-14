"""Retrieval: load PDFs, split them into chunks, index them, search them.

The whole retrieval half of the project lives in this one file. Read it top to
bottom and you know how it works:

    load PDF -> chunk by page -> embed -> FAISS + BM25 -> fuse -> rerank
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

# --- settings -------------------------------------------------------------
ROOT = Path(__file__).parent
DOCS_DIR = Path(os.getenv("RAG_DOCS_DIR", ROOT / "data" / "docs"))
INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", ROOT / "data" / "index"))

EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

CHUNK_TOKENS = 320
CHUNK_OVERLAP = 60
TOP_DENSE = 24
TOP_BM25 = 24
RERANK_POOL = 16
FINAL_TOP_K = 5
MIN_SCORE = -6.0  # below this the best passage is considered irrelevant

WORD = re.compile(r"[a-z0-9]+")
SENTENCE = re.compile(r"(?<=[.!?])\s+")
STOPWORDS = set(
    "a an and are as at be by for from has have how in is it its of on or that the "
    "this to was were what when where which who why will with".split()
)


@dataclass
class Chunk:
    text: str
    doc: str
    page: int


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None


# --- 1. loading -----------------------------------------------------------
def read_pdf(path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, text)] with page numbers preserved for citations."""
    pages = []
    for number, page in enumerate(PdfReader(str(path)).pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((number, text))
    return pages


def read_document(path: Path) -> list[tuple[int, str]]:
    if path.suffix.lower() == ".pdf":
        return read_pdf(path)
    return [(1, path.read_text(encoding="utf-8", errors="ignore").strip())]


# --- 2. chunking ----------------------------------------------------------
def count_words(text: str) -> int:
    return len(text.split())


def split_sentences(text: str, limit: int) -> list[str]:
    """Split on sentence boundaries; hard-split any sentence that is too long."""
    sentences = []
    for sentence in SENTENCE.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        words = sentence.split()
        if len(words) <= limit:
            sentences.append(sentence)
        else:  # a table or a wall of text with no punctuation
            sentences += [" ".join(words[i : i + limit]) for i in range(0, len(words), limit)]
    return sentences


def chunk_page(text: str, max_tokens: int = CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Pack sentences into chunks, never crossing a page boundary."""
    chunks: list[str] = []
    current: list[str] = []

    for sentence in split_sentences(text, max_tokens):
        if current and count_words(" ".join(current)) + count_words(sentence) > max_tokens:
            chunks.append(" ".join(current))
            # carry the tail of the window forward for context
            tail: list[str] = []
            for previous in reversed(current):
                if count_words(" ".join(tail)) + count_words(previous) > overlap:
                    break
                tail.insert(0, previous)
            current = tail
        current.append(sentence)

    if current:
        chunks.append(" ".join(current))
    return chunks


# --- 3. text helpers ------------------------------------------------------
def tokenize(text: str) -> list[str]:
    return [w for w in WORD.findall(text.lower()) if w not in STOPWORDS and len(w) > 1]


def stem(word: str) -> str:
    """Crude plural stripping so 'password' matches 'passwords'."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


# --- 4. the index ---------------------------------------------------------
class RAG:
    """Holds the corpus, the two indexes and the two models."""

    def __init__(self) -> None:
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.reranker = CrossEncoder(RERANK_MODEL)
        self.chunks: list[Chunk] = []
        self.index: faiss.IndexFlatIP | None = None
        self.bm25: BM25Okapi | None = None
        self.load()

    # -- build ------------------------------------------------------------
    def ingest(self) -> dict:
        self.chunks = []
        documents = []
        for path in sorted(DOCS_DIR.iterdir()):
            if path.suffix.lower() not in {".pdf", ".md", ".txt"} or path.name.startswith("."):
                continue
            documents.append(path.stem)
            title = path.stem.replace("_", " ").replace("-", " ").title()
            for page, text in read_document(path):
                for piece in chunk_page(text):
                    self.chunks.append(Chunk(text=piece, doc=title, page=page))

        if not self.chunks:
            raise ValueError(f"No documents found in {DOCS_DIR}. Run: python make_corpus.py")

        vectors = self.embedder.encode(
            [c.text for c in self.chunks], normalize_embeddings=True, show_progress_bar=False
        )
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self.bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])
        self.save()
        return {"documents": len(documents), "pages": len({(c.doc, c.page) for c in self.chunks}),
                "chunks": len(self.chunks)}

    # -- save / load --------------------------------------------------------
    def save(self) -> None:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(INDEX_DIR / "vectors.faiss"))
        (INDEX_DIR / "chunks.json").write_text(
            json.dumps([asdict(c) for c in self.chunks]), encoding="utf-8"
        )

    def load(self) -> bool:
        chunks_file = INDEX_DIR / "chunks.json"
        if not chunks_file.exists():
            return False
        self.chunks = [Chunk(**row) for row in json.loads(chunks_file.read_text())]
        self.index = faiss.read_index(str(INDEX_DIR / "vectors.faiss"))
        self.bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])
        return True

    # -- retrieval primitives ----------------------------------------------
    def dense_hits(self, question: str, k: int = TOP_DENSE) -> list[tuple[int, float]]:
        if not self.chunks:
            return []
        query = self.embedder.encode(QUERY_PREFIX + question, normalize_embeddings=True)
        scores, indices = self.index.search(np.array([query], dtype=np.float32), min(k, len(self.chunks)))
        return [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]

    def bm25_hits(self, question: str, k: int = TOP_BM25) -> list[tuple[int, float]]:
        if not self.chunks:
            return []
        scores = self.bm25.get_scores(tokenize(question))
        best = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in best if scores[i] > 0]

    @staticmethod
    def fuse(dense: list[tuple[int, float]], lexical: list[tuple[int, float]], k: int = 60):
        """Reciprocal Rank Fusion: merge two ranked lists using ranks only."""
        scores: dict[int, float] = {}
        for rank, (index, _) in enumerate(dense, start=1):
            scores[index] = scores.get(index, 0.0) + 1 / (k + rank)
        for rank, (index, _) in enumerate(lexical, start=1):
            scores[index] = scores.get(index, 0.0) + 1 / (k + rank)
        return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)

    # -- the one method callers need ----------------------------------------
    def search(self, question: str, top_k: int = FINAL_TOP_K) -> list[Hit]:
        dense = self.dense_hits(question)
        lexical = self.bm25_hits(question)
        fused = self.fuse(dense, lexical)[:RERANK_POOL]
        if not fused:
            return []

        candidates = [index for index, _ in fused]
        scores = self.reranker.predict([(question, self.chunks[i].text) for i in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)

        dense_rank = {index: rank + 1 for rank, (index, _) in enumerate(dense)}
        bm25_rank = {index: rank + 1 for rank, (index, _) in enumerate(lexical)}

        return [
            Hit(
                chunk=self.chunks[index],
                score=float(score),
                dense_rank=dense_rank.get(index),
                bm25_rank=bm25_rank.get(index),
            )
            for index, score in ranked[:top_k]
        ]
