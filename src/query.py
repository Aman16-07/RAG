"""Retrieval / query module for the car image RAG pipeline.

Provides:
  - ``RetrievalResult`` – data class representing a single search hit.
  - ``ImageRetriever``  – high-level API that loads a FAISS index + metadata,
    encodes text or image queries via CLIP, and returns the top-K most similar
    car images ranked by cosine similarity.

CLI usage::

    python -m src.query --text-query "red sports car" --top-k 5
    python -m src.query --image-query dataset/images/example.jpg --top-k 5
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss

try:  # Supports both `python -m src.query` and `python src/query.py`.
    from src.clip_backend import ClipConfig, ClipEncoder
    from src.faiss_store import load_faiss_index
    from src.utils import load_json
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    from clip_backend import ClipConfig, ClipEncoder
    from faiss_store import load_faiss_index
    from utils import load_json


@dataclass
class RetrievalResult:
    rank: int
    score: float
    file_path: str
    filename: str


class ImageRetriever:
    def __init__(
        self,
        index_path: Path,
        metadata_path: Path,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
        nprobe: int | None = None,
    ) -> None:
        self.index = load_faiss_index(index_path)
        metadata = load_json(metadata_path)
        if not isinstance(metadata, list):
            raise ValueError(f"Metadata at {metadata_path} is malformed.")
        self.metadata: list[dict[str, Any]] = metadata

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"Index vectors ({self.index.ntotal}) and metadata rows ({len(self.metadata)}) differ."
            )

        if isinstance(self.index, faiss.IndexIVF) and nprobe is not None:
            self.index.nprobe = nprobe

        self.encoder = ClipEncoder(
            ClipConfig(model_name=model_name, pretrained=pretrained, device=device)
        )

    def search_by_text(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        query_vector = self.encoder.encode_text(query)
        return self._search_vector(query_vector, top_k=top_k)

    def search_by_image(self, image_path: Path, top_k: int = 5) -> list[RetrievalResult]:
        query_vector = self.encoder.encode_query_image(image_path)
        return self._search_vector(query_vector, top_k=top_k)

    def _search_vector(self, vector, top_k: int) -> list[RetrievalResult]:
        scores, indices = self.index.search(vector, top_k)
        results: list[RetrievalResult] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0:
                continue
            row = self.metadata[idx]
            results.append(
                RetrievalResult(
                    rank=rank,
                    score=float(score),
                    file_path=str(row["file_path"]),
                    filename=str(row["filename"]),
                )
            )
        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search car images with CLIP + FAISS.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text-query", type=str, help="Text query for retrieval.")
    group.add_argument("--image-query", type=Path, help="Image path for image-to-image retrieval.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return.")
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path("faiss_index/car_images.index"),
        help="Path to FAISS index.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("embeddings/image_metadata.json"),
        help="Path to metadata JSON.",
    )
    parser.add_argument("--model-name", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (e.g., cuda, cpu). Defaults to auto.",
    )
    parser.add_argument(
        "--nprobe",
        type=int,
        default=None,
        help="IVF nprobe value for recall/latency tradeoff.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retriever = ImageRetriever(
        index_path=args.index_path,
        metadata_path=args.metadata_path,
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=args.device,
        nprobe=args.nprobe,
    )

    if args.text_query:
        results = retriever.search_by_text(args.text_query, top_k=args.top_k)
        print(f"Query (text): {args.text_query}")
    else:
        results = retriever.search_by_image(args.image_query, top_k=args.top_k)
        print(f"Query (image): {args.image_query}")

    if not results:
        print("No results found.")
        return

    print("-" * 70)
    print("Rank | Score    | Filename")
    print("-" * 70)
    for row in results:
        print(f"{row.rank:>4} | {row.score:>8.4f} | {row.filename}")
        print(f"      Path: {row.file_path}")
    print("-" * 70)


if __name__ == "__main__":
    main()
