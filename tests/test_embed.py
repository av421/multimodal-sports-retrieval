from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sports_retrieval.embed import embed_images, embed_text, load_clip


@pytest.fixture(scope="module")
def embedder():
    return load_clip()


def _make_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (64, 64), color=color).save(path)


def test_embed_images_are_unit_normalized(tmp_path: Path, embedder) -> None:
    paths = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    _make_image(paths[0], (255, 0, 0))
    _make_image(paths[1], (0, 255, 0))

    embs, valid_paths = embed_images(embedder, paths, batch_size=8, show_progress=False)

    assert valid_paths == paths
    assert embs.shape == (2, embedder.model.visual.output_dim)
    np.testing.assert_allclose(np.linalg.norm(embs, axis=1), 1.0, atol=1e-5)


def test_embed_images_skips_unreadable_files(tmp_path: Path, embedder) -> None:
    good = tmp_path / "good.jpg"
    _make_image(good, (0, 0, 255))
    bad = tmp_path / "bad.jpg"
    bad.write_text("not an image")

    embs, valid_paths = embed_images(embedder, [good, bad], batch_size=8, show_progress=False)

    assert valid_paths == [good]
    assert embs.shape == (1, embedder.model.visual.output_dim)


def test_embed_text_is_unit_normalized(embedder) -> None:
    embs = embed_text(embedder, ["a dog running", "a cat sleeping"])
    assert embs.shape == (2, embedder.model.visual.output_dim)
    np.testing.assert_allclose(np.linalg.norm(embs, axis=1), 1.0, atol=1e-5)


def test_image_and_text_embeddings_are_comparable(tmp_path: Path, embedder) -> None:
    # A solid red square should be closer to "a red square" than to "a blue circle".
    img_path = tmp_path / "red.jpg"
    _make_image(img_path, (255, 0, 0))

    img_embs, _ = embed_images(embedder, [img_path], show_progress=False)
    text_embs = embed_text(embedder, ["a red square", "a blue circle"])

    sims = img_embs @ text_embs.T
    assert sims[0, 0] > sims[0, 1]
