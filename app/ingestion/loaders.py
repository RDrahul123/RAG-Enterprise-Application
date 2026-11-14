"""Document loaders: PDF, Markdown and plain text -> ``Document`` with pages.

Page numbers are preserved end to end. That is the whole reason citations can
point at "page 4" instead of "somewhere in this file".
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from app.models import Document, Page
from app.observability import log_event

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt"}


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return cleaned or "document"


def make_doc_id(path: Path) -> str:
    """Stable id from the relative path, so re-ingesting keeps citation links."""
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:10]
    return f"{_slug(path.stem)[:40]}-{digest}"


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path: Path) -> Document:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[Page] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - defensive
            log_event("pdf_page_failed", path=str(path), page=i, error=str(exc))
            raw = ""
        text = _clean(raw)
        if text:
            pages.append(Page(page_number=i, text=text))

    title = ""
    try:
        meta = reader.metadata or {}
        title = str(meta.get("/Title") or "").strip()
    except Exception:  # pragma: no cover - defensive
        title = ""

    return Document(
        doc_id=make_doc_id(path),
        title=title or path.stem.replace("_", " ").replace("-", " ").title(),
        source_path=path.name,
        doc_type="pdf",
        pages=pages,
    )


def load_markdown(path: Path) -> Document:
    text = _clean(path.read_text(encoding="utf-8", errors="ignore"))
    title = ""
    match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    if match:
        title = match.group(1).strip()
    return Document(
        doc_id=make_doc_id(path),
        title=title or path.stem.replace("_", " ").title(),
        source_path=path.name,
        doc_type="markdown",
        pages=[Page(page_number=1, text=text)] if text else [],
    )


def load_text(path: Path) -> Document:
    text = _clean(path.read_text(encoding="utf-8", errors="ignore"))
    return Document(
        doc_id=make_doc_id(path),
        title=path.stem.replace("_", " ").title(),
        source_path=path.name,
        doc_type="text",
        pages=[Page(page_number=1, text=text)] if text else [],
    )


def load_document(path: Path) -> Document:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in {".md", ".markdown"}:
        return load_markdown(path)
    if suffix == ".txt":
        return load_text(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def load_corpus(docs_dir: Path) -> list[Document]:
    """Load every supported file in ``docs_dir`` (non-recursive)."""
    documents: list[Document] = []
    for path in sorted(docs_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if path.name.startswith("."):
            continue
        try:
            doc = load_document(path)
        except Exception as exc:
            log_event("document_load_failed", path=str(path), error=str(exc))
            continue
        if doc.pages:
            documents.append(doc)
    return documents
