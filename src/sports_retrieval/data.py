"""Download and load the Kaggle "100 Sports Image Classification" dataset.

CLIP is used frozen/pretrained here (no training on this data), so the usual
train/valid/test leakage concern doesn't apply the way it would for a
classifier: we pool all splits into one searchable index and keep `split` as
provenance metadata rather than a train/eval boundary.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import kagglehub

DATASET_SLUG = "gpiosenka/sports-classification"


@dataclass(frozen=True)
class ImageRecord:
    filepath: Path
    label: str
    split: str


def download_dataset() -> Path:
    """Download (or reuse the local kagglehub cache of) the dataset.

    Returns the root directory containing train/, valid/, test/, and sports.csv.
    """
    return Path(kagglehub.dataset_download(DATASET_SLUG))


def load_manifest(
    dataset_root: Path,
    max_per_class: int | None = None,
    splits: tuple[str, ...] = ("train", "valid", "test"),
    seed: int = 42,
) -> list[ImageRecord]:
    """Read sports.csv and build the list of images to embed.

    Args:
        dataset_root: path returned by download_dataset().
        max_per_class: if set, cap the number of images kept per (label, split)
            combination, chosen deterministically via a seeded shuffle. Use this
            to shrink the dataset for faster iteration on CPU/MPS.
        splits: which of train/valid/test to include.
        seed: shuffle seed for reproducible capping.
    """
    import random

    csv_path = dataset_root / "sports.csv"
    by_group: dict[tuple[str, str], list[ImageRecord]] = {}

    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            split = row["data set"]
            if split not in splits:
                continue
            record = ImageRecord(
                filepath=dataset_root / row["filepaths"],
                label=row["labels"],
                split=split,
            )
            by_group.setdefault((record.label, split), []).append(record)

    rng = random.Random(seed)
    records: list[ImageRecord] = []
    for group in by_group.values():
        if max_per_class is not None and len(group) > max_per_class:
            group = rng.sample(group, max_per_class)
        records.extend(group)

    records.sort(key=lambda r: (r.label, r.split, str(r.filepath)))
    return records


def class_names(records: list[ImageRecord]) -> list[str]:
    return sorted({r.label for r in records})
