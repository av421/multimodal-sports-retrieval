"""Evaluate retrieval quality against the dataset's own class labels.

Two protocols:
  - Text-to-image: for each class name (as a text query), precision@k and
    recall@k against images of that true class.
  - Image-to-image: leave-one-out -- each indexed image is queried against
    the index (with itself excluded), checking how often its neighbors
    share its true class.

Both reduce to the same underlying question -- of the top-k retrieved
items, how many share the query's true label -- so the metric helpers
below are shared between them.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sports_retrieval.data import ImageRecord
from sports_retrieval.embed import ClipEmbedder, embed_text
from sports_retrieval.index import IndexProcess

PROMPT_TEMPLATE = "a photo of {label}"


def precision_at_k(retrieved_labels: list[str], true_label: str, k: int) -> float:
    considered = retrieved_labels[:k]
    if not considered:
        return 0.0
    return sum(1 for label in considered if label == true_label) / len(considered)


def recall_at_k(retrieved_labels: list[str], true_label: str, k: int, total_relevant: int) -> float:
    if total_relevant == 0:
        return 0.0
    considered = retrieved_labels[:k]
    return sum(1 for label in considered if label == true_label) / total_relevant


@dataclass(frozen=True)
class TextToImageResult:
    per_class: pd.DataFrame  # columns: label, num_images, precision_at_k, recall_at_k
    mean_precision_at_k: float
    mean_recall_at_k: float


def evaluate_text_to_image(
    embedder: ClipEmbedder,
    index_process: IndexProcess,
    metadata: list[ImageRecord],
    class_names: list[str],
    k: int = 10,
    prompt_template: str = PROMPT_TEMPLATE,
) -> TextToImageResult:
    class_counts = Counter(r.label for r in metadata)

    prompts = [prompt_template.format(label=c) for c in class_names]
    queries = embed_text(embedder, prompts)
    _distances, indices = index_process.search(queries, k)

    rows = []
    for class_name, retrieved_idx in zip(class_names, indices, strict=True):
        retrieved_labels = [metadata[i].label for i in retrieved_idx if i >= 0]
        total_relevant = class_counts[class_name]
        rows.append(
            {
                "label": class_name,
                "num_images": total_relevant,
                "precision_at_k": precision_at_k(retrieved_labels, class_name, k),
                "recall_at_k": recall_at_k(retrieved_labels, class_name, k, total_relevant),
            }
        )

    per_class = pd.DataFrame(rows)
    return TextToImageResult(
        per_class=per_class,
        mean_precision_at_k=per_class["precision_at_k"].mean(),
        mean_recall_at_k=per_class["recall_at_k"].mean(),
    )


@dataclass(frozen=True)
class ImageToImageResult:
    mean_precision_at_k: float
    top1_accuracy: float
    confusions: pd.DataFrame  # columns: true_label, predicted_label, count -- sorted desc


def evaluate_image_to_image(
    index_process: IndexProcess,
    metadata: list[ImageRecord],
    k: int = 10,
) -> ImageToImageResult:
    all_vectors = index_process.reconstruct_all()
    # search for k+1: row 0 of each result is (almost always) the query
    # itself, which we then drop to get the true k leave-one-out neighbors.
    _distances, indices = index_process.search(all_vectors, k + 1)

    precisions = []
    top1_correct = 0
    confusion_counts: Counter[tuple[str, str]] = Counter()

    for query_idx, retrieved_idx in enumerate(indices):
        true_label = metadata[query_idx].label
        neighbors = [i for i in retrieved_idx if i >= 0 and i != query_idx][:k]
        neighbor_labels = [metadata[i].label for i in neighbors]

        precisions.append(precision_at_k(neighbor_labels, true_label, k))

        if neighbor_labels:
            top1_label = neighbor_labels[0]
            if top1_label == true_label:
                top1_correct += 1
            else:
                confusion_counts[(true_label, top1_label)] += 1

    confusion_rows = [
        {"true_label": t, "predicted_label": p, "count": c}
        for (t, p), c in confusion_counts.items()
    ]
    confusions = pd.DataFrame(confusion_rows).sort_values(
        "count", ascending=False, ignore_index=True
    )

    n = len(metadata)
    return ImageToImageResult(
        mean_precision_at_k=float(np.mean(precisions)) if precisions else 0.0,
        top1_accuracy=top1_correct / n if n else 0.0,
        confusions=confusions,
    )


def plot_confusion_heatmap(confusions: pd.DataFrame, path: Path, top_n_pairs: int = 25) -> None:
    """Heatmap over the classes involved in the top_n_pairs most-confused
    (true_label, predicted_label) pairs -- not the full 100x100 matrix,
    which is mostly empty and unreadable at this scale."""
    top_pairs = confusions.head(top_n_pairs)
    labels = sorted(set(top_pairs["true_label"]) | set(top_pairs["predicted_label"]))
    label_idx = {label: i for i, label in enumerate(labels)}

    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for _, row in top_pairs.iterrows():
        matrix[label_idx[row["true_label"]], label_idx[row["predicted_label"]]] = row["count"]

    fig_size = max(6, len(labels) * 0.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(matrix, cmap="Reds")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted (nearest neighbor's label)")
    ax.set_ylabel("True label")
    ax.set_title(f"Top {top_n_pairs} class confusions (leave-one-out, top-1)")
    fig.colorbar(im, ax=ax, label="count")
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
