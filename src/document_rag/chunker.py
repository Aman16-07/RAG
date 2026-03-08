"""Text chunking module for the document RAG pipeline.

Splits extracted document text into overlapping word-level chunks that
preserve semantic context for embedding and retrieval.  Each chunk carries
metadata identifying its source document and position.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TextChunk:
    """A single text chunk extracted from a document."""

    chunk_id: str
    document_source: str
    text_content: str
    chunk_index: int


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[TextChunk]:
    """Split *text* into overlapping word-level chunks.

    Parameters
    ----------
    text : str
        Full extracted document text.
    source : str
        Document source name used in chunk metadata.
    chunk_size : int
        Maximum number of words per chunk (approximates tokens).
    overlap : int
        Number of words to overlap between consecutive chunks.

    Returns
    -------
    list[TextChunk]
        Ordered list of text chunks with metadata.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[TextChunk] = []
    start = 0
    chunk_index = 0
    step = max(chunk_size - overlap, 1)

    while start < len(words):
        end = start + chunk_size
        chunk_content = " ".join(words[start:end])
        if chunk_content.strip():
            chunks.append(
                TextChunk(
                    chunk_id=f"{source}_chunk_{chunk_index}",
                    document_source=source,
                    text_content=chunk_content,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1
        start += step

    return chunks
