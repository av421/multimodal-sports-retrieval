import sys
from pathlib import Path

import gradio as gr
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from app import run_search  # noqa: E402

from sports_retrieval.search import SearchResult  # noqa: E402


class FakeEngine:
    def __init__(self) -> None:
        self.image_calls = []
        self.text_calls = []

    def search_image(self, image, k):
        self.image_calls.append((image, k))
        return [SearchResult(filepath=Path("/x/golf.jpg"), label="golf", split="train", score=0.9)]

    def search_text(self, text, k):
        self.text_calls.append((text, k))
        return [
            SearchResult(filepath=Path("/x/archery.jpg"), label="archery", split="train", score=0.5)
        ]


def test_run_search_prefers_image_when_both_given() -> None:
    engine = FakeEngine()
    results = run_search(engine, image="fake_image", text="some text", k=3)

    assert engine.image_calls == [("fake_image", 3)]
    assert engine.text_calls == []
    assert results == [("/x/golf.jpg", "golf -- 0.900")]


def test_run_search_uses_text_when_no_image() -> None:
    engine = FakeEngine()
    results = run_search(engine, image=None, text="diving into a pool", k=5)

    assert engine.text_calls == [("diving into a pool", 5)]
    assert results == [("/x/archery.jpg", "archery -- 0.500")]


def test_run_search_ignores_blank_text() -> None:
    engine = FakeEngine()
    with pytest.raises(gr.Error):
        run_search(engine, image=None, text="   ", k=5)


def test_run_search_errors_with_no_input() -> None:
    engine = FakeEngine()
    with pytest.raises(gr.Error):
        run_search(engine, image=None, text=None, k=5)
