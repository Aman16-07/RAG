"""Document text extraction for the document RAG pipeline.

Supports PDF, DOCX, TXT, and Markdown files.  Uses ``pdfplumber`` for PDF
extraction and ``python-docx`` for DOCX files.
"""
from __future__ import annotations

from pathlib import Path

SUPPORTED_DOC_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def load_document(file_path: Path) -> str:
    """Extract text content from a document file.

    Raises ``ValueError`` for unsupported formats and ``FileNotFoundError``
    if *file_path* does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_DOC_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            f"Supported: {', '.join(sorted(SUPPORTED_DOC_EXTENSIONS))}"
        )

    if suffix == ".pdf":
        return _load_pdf(file_path)
    if suffix == ".docx":
        return _load_docx(file_path)
    return _load_text(file_path)


def _load_pdf(path: Path) -> str:
    """Extract text from a PDF using pdfplumber."""
    import pdfplumber

    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages)


def _load_docx(path: Path) -> str:
    """Extract text from a DOCX file using python-docx."""
    from docx import Document

    doc = Document(str(path))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _load_text(path: Path) -> str:
    """Read a plain-text or Markdown file with encoding safety."""
    return path.read_text(encoding="utf-8", errors="replace")
