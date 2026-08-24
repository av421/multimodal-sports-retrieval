"""Gradio demo: search the sports image index by uploaded photo or by text.

Usage:
    python scripts/app.py
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import gradio as gr  # safe at module level -- gradio doesn't import torch

if TYPE_CHECKING:
    from sports_retrieval.search import SearchEngine

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "assets" / "examples"

DESCRIPTION = """
Search ~14,500 sports action photos (100 classes, from the Kaggle
"100 Sports Image Classification" dataset) by **uploading a photo** or
**typing a description**. Both go through the same CLIP model into one
shared embedding space, so an image query and a text query are answered
by the same nearest-neighbor search over a FAISS index.

Note: the bundled example photos are themselves part of the search index,
so an image query using one of them will trivially return itself as the
#1 result (score 1.000) -- upload your own photo to see genuine
out-of-distribution retrieval.
"""


def run_search(engine: SearchEngine, image, text, k: int) -> list[tuple[str, str]]:
    """Dispatch to image or text search; image takes priority if both are set.

    Standalone (not a closure) so this dispatch/validation logic is
    unit-testable against a fake engine, without needing Gradio or CLIP.
    """
    if image is not None:
        results = engine.search_image(image, k=k)
    elif text and text.strip():
        results = engine.search_text(text.strip(), k=k)
    else:
        raise gr.Error("Upload an image or type a text query first.")
    return [(str(r.filepath), f"{r.label} -- {r.score:.3f}") for r in results]


def build_demo(engine: SearchEngine) -> gr.Blocks:
    example_images = sorted(EXAMPLES_DIR.glob("*.jpg")) if EXAMPLES_DIR.is_dir() else []

    with gr.Blocks(title="Multimodal Sports Retrieval") as demo:
        gr.Markdown("# Multimodal Sports Retrieval")
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column():
                image_in = gr.Image(type="pil", label="Upload an image")
                text_in = gr.Textbox(
                    label="...or describe a scene",
                    placeholder='e.g. "diving into a pool", "swinging a golf club"',
                )
                k_in = gr.Slider(1, 10, value=5, step=1, label="Number of results (k)")
                search_btn = gr.Button("Search", variant="primary")
                if example_images:
                    gr.Examples(
                        examples=[[str(p), None, 5] for p in example_images],
                        inputs=[image_in, text_in, k_in],
                        label="Example images",
                    )
            with gr.Column():
                gallery = gr.Gallery(label="Top-k results", columns=5, object_fit="contain")

        handler = partial(run_search, engine)
        search_btn.click(handler, inputs=[image_in, text_in, k_in], outputs=gallery)
        text_in.submit(handler, inputs=[image_in, text_in, k_in], outputs=gallery)

    return demo


def main() -> None:
    # Imported here, not at module level: multiprocessing's spawn start method
    # re-executes this script's top-level code in the search subprocess (see
    # IndexProcess docstring). A module-level SearchEngine import/instantiation
    # would recursively spawn another SearchEngine in that subprocess.
    from sports_retrieval.data import download_dataset
    from sports_retrieval.search import SearchEngine

    engine = SearchEngine(
        index_path=ARTIFACTS_DIR / "index.faiss",
        metadata_path=ARTIFACTS_DIR / "metadata.csv",
    )
    demo = build_demo(engine)
    try:
        # Result images live in kagglehub's cache dir, outside Gradio's
        # default allowed (cwd/tmp) paths -- without this it refuses to
        # serve them.
        dataset_root = download_dataset()
        demo.launch(allowed_paths=[str(dataset_root)])
    finally:
        engine.close()


if __name__ == "__main__":
    main()
