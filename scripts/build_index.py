"""Embed the full dataset with CLIP and build a FAISS index over it.

Writes artifacts/index.faiss and artifacts/metadata.csv (row i of the CSV
corresponds to row i of the index). Both are gitignored -- regenerate with:

    python scripts/build_index.py
    python scripts/build_index.py --max-per-class 40  # faster dev iteration
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from sports_retrieval.data import class_names, download_dataset, load_manifest
from sports_retrieval.embed import embed_images, load_clip
from sports_retrieval.index import build_index, save_index, save_metadata

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out-dir", type=Path, default=ARTIFACTS_DIR)
    args = parser.parse_args()

    print("Loading dataset manifest...")
    root = download_dataset()
    records = load_manifest(root, max_per_class=args.max_per_class)
    print(f"{len(records)} images across {len(class_names(records))} classes")

    print("Loading CLIP model...")
    embedder = load_clip()
    print(f"Using device: {embedder.device}")

    print("Embedding images...")
    t0 = time.time()
    paths = [r.filepath for r in records]
    embeddings, valid_paths = embed_images(embedder, paths, batch_size=args.batch_size)
    dt = time.time() - t0
    print(f"Embedded {len(valid_paths)}/{len(paths)} images in {dt:.1f}s")

    # embed_images can drop unreadable files; keep records aligned to valid_paths.
    valid_set = set(valid_paths)
    kept_records = [r for r in records if r.filepath in valid_set]
    assert len(kept_records) == len(valid_paths) == embeddings.shape[0]

    print("Building FAISS index...")
    index = build_index(embeddings)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.out_dir / "index.faiss"
    metadata_path = args.out_dir / "metadata.csv"
    save_index(index, index_path)
    save_metadata(kept_records, metadata_path)

    print(f"Saved index ({index.ntotal} vectors, dim={index.d}) to {index_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
