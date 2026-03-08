"""Embedding generation for the document RAG pipeline.

Uses SentenceTransformers to produce L2-normalised embeddings for text
chunks and queries.  Normalised vectors enable cosine similarity via
FAISS inner-product search (``IndexFlatIP``).
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class DocumentEmbedder:
    """Encode text chunks and queries using a SentenceTransformer model."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> None:
        self.model = SentenceTransformer(model_name, device=device)

    def embed_chunks(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Embed a batch of text chunks, returning L2-normalised vectors."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string, returning an L2-normalised 2-D vector."""
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )
        return np.asarray(embedding, dtype=np.float32)
