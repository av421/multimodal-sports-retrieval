"""Download the sports dataset and report the size of the resulting manifest.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --max-per-class 40
"""

from __future__ import annotations

import argparse
from collections import Counter

from sports_retrieval.data import class_names, download_dataset, load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="cap images per (class, split) for faster iteration; default: no cap",
    )
    args = parser.parse_args()

    print("Downloading dataset (or reusing cache)...")
    root = download_dataset()
    print(f"Dataset root: {root}")

    records = load_manifest(root, max_per_class=args.max_per_class)
    print(f"Total images: {len(records)}")
    print(f"Classes: {len(class_names(records))}")

    split_counts = Counter(r.split for r in records)
    for split, count in sorted(split_counts.items()):
        print(f"  {split}: {count}")


if __name__ == "__main__":
    main()
