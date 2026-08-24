import csv
from pathlib import Path

from sports_retrieval.data import class_names, load_manifest


def _make_fake_dataset(root: Path) -> None:
    rows = []
    for cls, n in [("archery", 5), ("baseball", 3)]:
        for split, count in [("train", n), ("valid", 2), ("test", 2)]:
            for i in range(count):
                rel = f"{split}/{cls}/{i:03d}.jpg"
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).touch()
                rows.append(
                    {
                        "class id": "0",
                        "filepaths": rel,
                        "labels": cls,
                        "data set": split,
                    }
                )

    with open(root / "sports.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class id", "filepaths", "labels", "data set"])
        writer.writeheader()
        writer.writerows(rows)


def test_load_manifest_no_cap(tmp_path: Path) -> None:
    _make_fake_dataset(tmp_path)
    records = load_manifest(tmp_path)
    assert len(records) == (5 + 2 + 2) + (3 + 2 + 2)
    assert class_names(records) == ["archery", "baseball"]


def test_load_manifest_caps_per_class_and_split(tmp_path: Path) -> None:
    _make_fake_dataset(tmp_path)
    records = load_manifest(tmp_path, max_per_class=2)
    # archery train (5->2) + valid (2) + test (2), baseball train (3->2) + valid (2) + test (2)
    assert len(records) == (2 + 2 + 2) + (2 + 2 + 2)
    for r in records:
        assert r.filepath.exists()


def test_load_manifest_filters_splits(tmp_path: Path) -> None:
    _make_fake_dataset(tmp_path)
    records = load_manifest(tmp_path, splits=("train",))
    assert all(r.split == "train" for r in records)
    assert len(records) == 5 + 3
