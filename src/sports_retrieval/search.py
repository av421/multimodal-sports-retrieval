"""Unified search over the CLIP + FAISS index: query by image or by text.

SearchEngine loads the CLIP model once and keeps a process-isolated FAISS
index (see index.IndexProcess) alive for the caller's lifetime, so repeated
queries -- an eval loop, a Gradio app -- don't pay model-load cost per call.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import numpy as np
from PIL import Image

from sports_retrieval.data import ImageRecord
from sports_retrieval.embed import ClipEmbedder, embed_images, embed_text, load_clip
from sports_retrieval.index import IndexProcess, load_metadata


@dataclass(frozen=True)
class SearchResult:
    filepath: Path
    label: str
    split: str
    score: float


class SearchEngine:
    def __init__(
        self,
        index_path: Path,
        metadata_path: Path,
        embedder: ClipEmbedder | None = None,
    ) -> None:
        self.embedder = embedder or load_clip()
        self.metadata: list[ImageRecord] = load_metadata(metadata_path)
        self._index_process = IndexProcess(index_path)

    def search_text(self, text: str, k: int = 5) -> list[SearchResult]:
        query = embed_text(self.embedder, [text])
        return self._search(query, k)

    def search_image(self, image: Path | Image.Image, k: int = 5) -> list[SearchResult]:
        embeddings, valid = embed_images(self.embedder, [image], show_progress=False)
        if not valid:
            raise ValueError(f"Could not read query image: {image}")
        return self._search(embeddings, k)

    def _search(self, query: np.ndarray, k: int) -> list[SearchResult]:
        distances, indices = self._index_process.search(query, k)
        results = []
        for idx, score in zip(indices[0], distances[0], strict=True):
            if idx < 0:  # FAISS pads with -1 if k > ntotal
                continue
            record = self.metadata[idx]
            results.append(
                SearchResult(
                    filepath=record.filepath,
                    label=record.label,
                    split=record.split,
                    score=float(score),
                )
            )
        return results

    def close(self) -> None:
        self._index_process.close()

    def __enter__(self) -> "SearchEngine":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
