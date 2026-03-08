"""Shared utility functions for the car image RAG pipeline.

Provides:
  - Image file discovery with extension filtering.
  - JSON I/O helpers.
  - L2 normalisation for embedding vectors.
  - Generic chunk iterator for batch processing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, TypeVar

import numpy as np

# Supported image formats for ingestion.
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def ensure_dir(path: Path) -> None:
    """Create *path* and any parents if they do not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def discover_images(dataset_dir: Path) -> list[Path]:
    """Recursively discover image files under *dataset_dir*.

    Returns a sorted list of :class:`Path` objects.
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    image_paths = [
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    image_paths.sort()
    return image_paths


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------


def save_json(data: object, path: Path) -> None:
    """Serialise *data* to a JSON file at *path*, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=True, indent=2)


def load_json(path: Path) -> object:
    """Deserialise and return the contents of a JSON file."""
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalise *vectors* row-wise.

    This allows cosine similarity to be computed with a simple inner product,
    which is the metric FAISS ``IndexFlatIP`` uses.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return vectors / norms


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def chunk_iterable(items: Iterable[T], chunk_size: int) -> Iterable[list[T]]:
    """Yield successive lists of at most *chunk_size* items.

    Useful for batching image ingestion and embedding generation so that
    memory usage stays bounded.
    """
    chunk: list[T] = []
    for item in items:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

