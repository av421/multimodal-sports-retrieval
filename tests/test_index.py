from pathlib import Path

import numpy as np

from sports_retrieval.data import ImageRecord
from sports_retrieval.index import (
    IndexProcess,
    build_index,
    load_index,
    load_metadata,
    save_index,
    save_metadata,
)


def _random_normalized(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.normal(size=(n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


def test_build_index_has_expected_shape() -> None:
    embeddings = _random_normalized(20, 8)
    index = build_index(embeddings)

    assert index.ntotal == 20
    assert index.d == 8


def test_save_and_load_index_roundtrip(tmp_path: Path) -> None:
    embeddings = _random_normalized(10, 8)
    index = build_index(embeddings)
    path = tmp_path / "sub" / "index.faiss"

    save_index(index, path)
    loaded = load_index(path)

    assert loaded.ntotal == index.ntotal
    assert loaded.d == index.d


def test_index_process_finds_exact_self_match(tmp_path: Path) -> None:
    # Runs in a subprocess (see IndexProcess docstring) -- this is the only
    # supported way to call .search() safely once torch has been imported
    # in this process (test_embed.py, earlier in the suite, already has).
    embeddings = _random_normalized(20, 8)
    index = build_index(embeddings)
    path = tmp_path / "index.faiss"
    save_index(index, path)

    with IndexProcess(path) as proc:
        distances, indices = proc.search(embeddings, k=1)

    assert list(indices[:, 0]) == list(range(20))
    np.testing.assert_allclose(distances[:, 0], 1.0, atol=1e-5)


def test_save_and_load_metadata_roundtrip(tmp_path: Path) -> None:
    records = [
        ImageRecord(filepath=Path("/data/train/archery/001.jpg"), label="archery", split="train"),
        ImageRecord(filepath=Path("/data/valid/baseball/002.jpg"), label="baseball", split="valid"),
    ]
    path = tmp_path / "sub" / "metadata.csv"

    save_metadata(records, path)
    loaded = load_metadata(path)

    assert loaded == records
