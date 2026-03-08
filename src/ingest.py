"""Data ingestion module for the car image RAG pipeline.

Responsibilities:
  - Scan the dataset directory for supported image files.
  - Validate images using PIL (optional).
  - Extract metadata: filename, folder category, potential label.
  - Process images in configurable chunks for memory efficiency.
  - Persist a manifest JSON consumed by downstream embedding / indexing stages.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

try:  # Supports both `python -m src.ingest` and `python src/ingest.py`.
    from src.utils import chunk_iterable, discover_images, save_json
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    from utils import chunk_iterable, discover_images, save_json


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_images(image_paths: list[Path], chunk_size: int = 256) -> list[Path]:
    """Return only valid images that PIL can parse.

    Images are validated in chunks to allow progress reporting and to limit
    the number of open file handles at any time.
    """
    valid_paths: list[Path] = []
    for chunk in chunk_iterable(image_paths, chunk_size):
        for path in tqdm(chunk, desc="Validating images", leave=False):
            try:
                with Image.open(path) as image:
                    image.verify()
                valid_paths.append(path)
            except (OSError, UnidentifiedImageError):
                continue
    return valid_paths


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------


def _derive_label(filename: str) -> str:
    """Derive a human-friendly potential label from the filename.

    Examples:
        'red_sedan_001.jpg' -> 'red sedan 001'
        'car_003.jpg'       -> 'car 003'
    """
    stem = Path(filename).stem
    return stem.replace("_", " ").replace("-", " ")


def _derive_category(image_path: Path, dataset_root: Path) -> str:
    """Derive a folder category from the relative path.

    If the image lives directly in the dataset root the category is 'default'.
    Otherwise the first sub-directory name is used (e.g. 'SUV', 'sedan').
    """
    try:
        relative = image_path.resolve().relative_to(dataset_root.resolve())
    except ValueError:
        return "default"
    parts = relative.parts
    return parts[0] if len(parts) > 1 else "default"


def build_manifest(
    image_paths: list[Path],
    dataset_root: Path,
    chunk_size: int = 512,
) -> list[dict[str, str | int]]:
    """Build a manifest list containing metadata for every image.

    Each entry stores:
      - id:             sequential integer
      - file_path:      absolute path to the image
      - filename:       basename of the file
      - label:          human-friendly label derived from filename
      - category:       folder-level category (sub-directory name or 'default')
    """
    rows: list[dict[str, str | int]] = []
    idx = 0
    for chunk in chunk_iterable(image_paths, chunk_size):
        for path in chunk:
            rows.append(
                {
                    "id": idx,
                    "file_path": str(path.resolve()),
                    "filename": path.name,
                    "label": _derive_label(path.name),
                    "category": _derive_category(path, dataset_root),
                }
            )
            idx += 1
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest car images and build manifest.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset/images"),
        help="Root directory containing car images.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("embeddings/manifest.json"),
        help="Output manifest JSON path.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate image files using PIL before writing manifest.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Number of images processed per chunk during ingestion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Step 1 – discover image files
    image_paths = discover_images(args.dataset_dir)
    print(f"Discovered images: {len(image_paths)}")

    # Step 2 – optional validation (chunked)
    if args.validate:
        image_paths = validate_images(image_paths, chunk_size=args.chunk_size)
        print(f"Valid images after validation: {len(image_paths)}")

    # Step 3 – build manifest with enriched metadata
    manifest = build_manifest(
        image_paths,
        dataset_root=args.dataset_dir,
        chunk_size=args.chunk_size,
    )
    save_json(manifest, args.manifest_path)
    print(f"Manifest saved to: {args.manifest_path}  ({len(manifest)} entries)")


if __name__ == "__main__":
    main()
