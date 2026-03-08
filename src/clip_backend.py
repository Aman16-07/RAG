"""CLIP / OpenCLIP encoder backend for the car image RAG pipeline.

Provides:
  - ``ClipConfig`` – dataclass holding model selection and device preferences.
  - ``ClipEncoder`` – high-level API to encode images and text queries into
    L2-normalised embedding vectors suitable for inner-product similarity
    search with FAISS.

Optimisations:
  - Batch image encoding via a PyTorch ``DataLoader`` with configurable
    ``batch_size`` and ``num_workers``.
  - Automatic CUDA/GPU acceleration when available.
  - ``@torch.inference_mode()`` to disable gradient tracking.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:  # Supports both `python -m src.embed` and `python src/embed.py`.
    from src.utils import l2_normalize
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    from utils import l2_normalize

try:
    import open_clip
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise ImportError(
        "open-clip-torch is required. Install with: pip install open-clip-torch"
    ) from exc


class _ImagePathDataset(Dataset):
    def __init__(self, image_paths: list[Path], preprocess) -> None:
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.preprocess(image)
        return tensor, str(image_path)


@dataclass
class ClipConfig:
    model_name: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    device: str | None = None


class ClipEncoder:
    """Wrapper around OpenCLIP image and text encoders."""

    def __init__(self, config: ClipConfig) -> None:
        device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        model, _, preprocess = open_clip.create_model_and_transforms(
            config.model_name,
            pretrained=config.pretrained,
            device=self.device,
        )
        self.model = model.eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(config.model_name)

    @torch.inference_mode()
    def encode_images(
        self,
        image_paths: list[Path],
        batch_size: int = 64,
        num_workers: int = 0,
        show_progress: bool = True,
    ) -> tuple[np.ndarray, list[str]]:
        dataset = _ImagePathDataset(image_paths=image_paths, preprocess=self.preprocess)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
        )

        all_embeddings: list[np.ndarray] = []
        ordered_paths: list[str] = []

        progress = tqdm(loader, desc="Embedding images", disable=not show_progress)
        for images, batch_paths in progress:
            images = images.to(self.device, non_blocking=True)
            image_features = self.model.encode_image(images)
            image_features = image_features.float()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_embeddings.append(image_features.cpu().numpy())
            ordered_paths.extend(batch_paths)

        embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32, copy=False)
        embeddings = l2_normalize(embeddings)
        return embeddings, ordered_paths

    @torch.inference_mode()
    def encode_text(self, query: str) -> np.ndarray:
        tokens = self.tokenizer([query]).to(self.device)
        text_features = self.model.encode_text(tokens)
        text_features = text_features.float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        vector = text_features.cpu().numpy().astype(np.float32, copy=False)
        return l2_normalize(vector)

    @torch.inference_mode()
    def encode_query_image(self, image_path: Path) -> np.ndarray:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        features = self.model.encode_image(image_tensor)
        features = features.float()
        features = features / features.norm(dim=-1, keepdim=True)
        vector = features.cpu().numpy().astype(np.float32, copy=False)
        return l2_normalize(vector)
