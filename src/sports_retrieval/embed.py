"""CLIP embedding pipeline: images and text into one shared vector space.

Uses open_clip (ViT-B/32, OpenAI weights) for speed on CPU/MPS. All embeddings
are L2-normalized so cosine similarity reduces to inner product, matching a
FAISS IndexFlatIP.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

MODEL_NAME = "ViT-B-32-quickgelu"  # matches the activation OpenAI's weights were trained with
PRETRAINED = "openai"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class ClipEmbedder:
    model: torch.nn.Module
    preprocess: object
    tokenizer: object
    device: str


def load_clip(
    model_name: str = MODEL_NAME,
    pretrained: str = PRETRAINED,
    device: str | None = None,
) -> ClipEmbedder:
    device = device or pick_device()
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval().to(device)
    return ClipEmbedder(model=model, preprocess=preprocess, tokenizer=tokenizer, device=device)


@torch.no_grad()
def embed_images(
    embedder: ClipEmbedder,
    images: list[Path | Image.Image],
    batch_size: int = 32,
    show_progress: bool = True,
) -> tuple[np.ndarray, list[Path | Image.Image]]:
    """Embed images in batches. Accepts file paths (bulk indexing) or already-open
    PIL Images (e.g. a Gradio upload with nothing on disk). Skips unreadable
    entries rather than failing the whole run.

    Returns (embeddings, valid_images) -- valid_images excludes any skipped
    entries, so callers must zip against it rather than the original list.
    """
    embeddings: list[np.ndarray] = []
    valid_images: list[Path | Image.Image] = []

    batches = range(0, len(images), batch_size)
    if show_progress:
        batches = tqdm(batches, desc="Embedding images", unit="batch")

    for start in batches:
        batch = images[start : start + batch_size]
        tensors = []
        batch_valid = []
        for item in batch:
            try:
                img = item if isinstance(item, Image.Image) else Image.open(item)
                tensors.append(embedder.preprocess(img.convert("RGB")))
                batch_valid.append(item)
            except Exception as e:
                logger.warning("Skipping unreadable image %s: %s", item, e)

        if not tensors:
            continue

        stack = torch.stack(tensors).to(embedder.device)
        feats = embedder.model.encode_image(stack)
        feats = F.normalize(feats, dim=-1)
        embeddings.append(feats.cpu().numpy())
        valid_images.extend(batch_valid)

    if not embeddings:
        return np.empty((0, embedder.model.visual.output_dim), dtype=np.float32), []

    return np.concatenate(embeddings).astype(np.float32), valid_images


@torch.no_grad()
def embed_text(embedder: ClipEmbedder, texts: list[str]) -> np.ndarray:
    tokens = embedder.tokenizer(texts).to(embedder.device)
    feats = embedder.model.encode_text(tokens)
    feats = F.normalize(feats, dim=-1)
    return feats.cpu().numpy().astype(np.float32)
