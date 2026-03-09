"""FAISS-backed document retrieval for the document RAG pipeline.

Manages a persistent FAISS inner-product index and associated chunk
metadata.  Supports incremental additions (new documents are appended
without rebuilding the whole index) and top-K similarity search.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from src.telemetry import get_tracer

_tracer = get_tracer("document-retrieval")


@dataclass
class DocumentSearchResult:
    """A single ranked search result from the document index."""

    rank: int
    score: float
    chunk_id: str
    document_source: str
    text_content: str


class DocumentIndex:
    """Persistent FAISS index with chunk metadata for document retrieval."""

    def __init__(self, index_path: Path, metadata_path: Path) -> None:
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: list[dict] = []
        self._load()

    # ── persistence ────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load the FAISS index and metadata from disk if they exist."""
        if self.index_path.exists() and self.metadata_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with self.metadata_path.open("r", encoding="utf-8") as fp:
                self.metadata = json.load(fp)

    def _save(self) -> None:
        """Persist the current FAISS index and metadata to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with self.metadata_path.open("w", encoding="utf-8") as fp:
            json.dump(self.metadata, fp, ensure_ascii=False, indent=2)

    # ── index operations ───────────────────────────────────────────────

    def add(self, embeddings: np.ndarray, chunk_metadata: list[dict]) -> None:
        """Append new embeddings and metadata to the index and persist."""
        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.metadata.extend(chunk_metadata)
        self._save()

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[DocumentSearchResult]:
        """Return the top-K chunks most similar to *query_embedding*."""
        if self.index is None or self.index.ntotal == 0:
            return []

        with _tracer.start_as_current_span("faiss_document_search") as span:
            k = min(top_k, self.index.ntotal)
            span.set_attribute("top_k", k)
            span.set_attribute("vector_index_size", self.index.ntotal)

            scores, indices = self.index.search(query_embedding, k)

            results: list[DocumentSearchResult] = []
            for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
                if idx < 0:
                    continue
                meta = self.metadata[idx]
                results.append(
                    DocumentSearchResult(
                        rank=rank,
                        score=float(score),
                        chunk_id=meta["chunk_id"],
                        document_source=meta["document_source"],
                        text_content=meta["text_content"],
                    )
                )
            span.set_attribute("results_count", len(results))
        return results

    # ── helpers ────────────────────────────────────────────────────────

    def get_indexed_documents(self) -> set[str]:
        """Return the set of already-indexed document source names."""
        return {m["document_source"] for m in self.metadata}

    @property
    def total_chunks(self) -> int:
        """Total number of chunk vectors in the index."""
        return self.index.ntotal if self.index else 0
