from pathlib import Path

import pytest
from PIL import Image

from sports_retrieval.data import ImageRecord
from sports_retrieval.embed import embed_images, load_clip
from sports_retrieval.index import build_index, save_index, save_metadata
from sports_retrieval.search import SearchEngine


@pytest.fixture(scope="module")
def built_index(tmp_path_factory) -> tuple[Path, Path]:
    """A tiny real index: a red square (labeled "red") and a blue square
    (labeled "blue"), embedded with the real CLIP model."""
    tmp_path = tmp_path_factory.mktemp("search_engine_fixture")
    red_path = tmp_path / "red.jpg"
    blue_path = tmp_path / "blue.jpg"
    Image.new("RGB", (64, 64), color=(255, 0, 0)).save(red_path)
    Image.new("RGB", (64, 64), color=(0, 0, 255)).save(blue_path)

    embedder = load_clip()
    embeddings, valid = embed_images(
        embedder, [red_path, blue_path], show_progress=False
    )
    records = [
        ImageRecord(filepath=red_path, label="red", split="train"),
        ImageRecord(filepath=blue_path, label="blue", split="train"),
    ]
    assert valid == [red_path, blue_path]

    index_path = tmp_path / "index.faiss"
    metadata_path = tmp_path / "metadata.csv"
    save_index(build_index(embeddings), index_path)
    save_metadata(records, metadata_path)
    return index_path, metadata_path


def test_search_text_ranks_matching_label_first(built_index: tuple[Path, Path]) -> None:
    index_path, metadata_path = built_index
    with SearchEngine(index_path, metadata_path) as engine:
        results = engine.search_text("a solid red square", k=2)

    assert len(results) == 2
    assert results[0].label == "red"
    assert results[0].score > results[1].score


def test_search_image_finds_exact_self_match(built_index: tuple[Path, Path]) -> None:
    index_path, metadata_path = built_index
    with SearchEngine(index_path, metadata_path) as engine:
        red_path = engine.metadata[0].filepath
        results = engine.search_image(red_path, k=2)

    assert results[0].filepath == red_path
    assert results[0].score == pytest.approx(1.0, abs=1e-5)


def test_search_image_accepts_pil_image_directly(built_index: tuple[Path, Path]) -> None:
    index_path, metadata_path = built_index
    with SearchEngine(index_path, metadata_path) as engine:
        img = Image.new("RGB", (64, 64), color=(0, 0, 255))
        results = engine.search_image(img, k=1)

    assert results[0].label == "blue"
