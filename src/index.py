"""FAISS index builder CLI for the car image RAG pipeline.

Reads the cached embedding matrix (``image_embeddings.npy``) and builds a
FAISS index that is persisted to disk.  Supports both flat (brute-force)
and IVF (approximate) index types.

Usage::

    python -m src.index --index-type flat --index-path faiss_index/car_images.index
    python -m src.index --index-type ivf --nlist 256 --index-path faiss_index/car_images.index
"""
from __future__ import annotations
import argparse
from pathlib import Path

import faiss
import numpy as np

try:  # Supports both `python -m src.index` and `python src/index.py`.from src.faiss_store import build_faiss_index, save_faiss_index
    from src.utils import ensure_dir
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    from faiss_store import build_faiss_index, save_faiss_index
    from utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and persist FAISS index.")
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        default=Path("embeddings/image_embeddings.npy"),
        help="Path to image embeddings (.npy).",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path("faiss_index/car_images.index"),
        help="Output path for FAISS index file.",
    )
    parser.add_argument(
        "--index-type",
        choices=["flat", "ivf"],
        default="flat",
        help="FAISS index type.",
    )
    parser.add_argument(
        "--nlist",
        type=int,
        default=100,
        help="Number of IVF clusters (used when --index-type ivf).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {args.embeddings_path}")

    embeddings = np.load(args.embeddings_path)
    embeddings = embeddings.astype(np.float32, copy=False)

    index = build_faiss_index(embeddings=embeddings, index_type=args.index_type, nlist=args.nlist)
    if isinstance(index, faiss.IndexIVF):
        index.nprobe = min(16, args.nlist)

    ensure_dir(args.index_path.parent)
    save_faiss_index(index, args.index_path)

    print(f"Index type: {args.index_type}")
    print(f"Vectors indexed: {index.ntotal}")
    print(f"Index saved to: {args.index_path}")


if __name__ == "__main__":
    main()
