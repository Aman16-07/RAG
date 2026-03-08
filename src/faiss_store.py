"""FAISS vector store module for the car image RAG pipeline.

Provides functions to build, save, and load FAISS indices.  Two index types
are supported:

  - **flat** (``IndexFlatIP``) – brute-force inner-product search.  Best
    accuracy; ideal for datasets up to ~100k vectors.
  - **ivf** (``IndexIVFFlat``) – inverted-file index with configurable
    *nlist* clusters.  Offers sub-linear search time for large datasets at
    a small recall trade-off controlled by *nprobe*.

All indices use **inner product** as the distance metric.  Embeddings must
be L2-normalised before insertion so that inner product equals cosine
similarity.
"""
from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


def build_faiss_index(
    embeddings: np.ndarray,
    index_type: str = "flat",
    nlist: int = 100,
) -> faiss.Index:
    """Build a FAISS index using cosine similarity via inner product."""
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")

    dimension = embeddings.shape[1]
    index_type = index_type.lower()

    if index_type == "flat":
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        return index

    if index_type == "ivf":
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        if embeddings.shape[0] < nlist:
            raise ValueError(
                f"Need at least nlist ({nlist}) vectors to train IVF index; "
                f"found {embeddings.shape[0]}."
            )
        index.train(embeddings)
        index.add(embeddings)
        return index

    raise ValueError(f"Unsupported index_type: {index_type}")


def save_faiss_index(index: faiss.Index, index_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))


def load_faiss_index(index_path: Path) -> faiss.Index:
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index file not found: {index_path}")
    return faiss.read_index(str(index_path))

