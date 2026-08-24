"""Run text-to-image and image-to-image evaluation against the built index.

Writes CSVs + a confusion heatmap PNG to artifacts/eval/, and prints summary
numbers (copy these into the README).

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py -k 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def main() -> None:
    # See IndexProcess docstring: these imports must stay inside main(), not
    # at module level, so the spawned search subprocess never inherits torch.
    from sports_retrieval.embed import load_clip
    from sports_retrieval.eval import (
        evaluate_image_to_image,
        evaluate_text_to_image,
        plot_confusion_heatmap,
    )
    from sports_retrieval.index import IndexProcess, load_metadata

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=ARTIFACTS_DIR / "eval")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(ARTIFACTS_DIR / "metadata.csv")
    class_names = sorted({r.label for r in metadata})
    print(f"{len(metadata)} images, {len(class_names)} classes, k={args.k}")

    embedder = load_clip()
    print(f"CLIP loaded on device={embedder.device}")

    with IndexProcess(ARTIFACTS_DIR / "index.faiss") as index_process:
        print("\n--- Text-to-image ---")
        t2i = evaluate_text_to_image(embedder, index_process, metadata, class_names, k=args.k)
        print(f"Mean precision@{args.k}: {t2i.mean_precision_at_k:.3f}")
        print(f"Mean recall@{args.k}:    {t2i.mean_recall_at_k:.3f}")
        t2i.per_class.sort_values("precision_at_k").to_csv(
            args.out_dir / "text_to_image_per_class.csv", index=False
        )

        print("\n--- Image-to-image (leave-one-out) ---")
        i2i = evaluate_image_to_image(index_process, metadata, k=args.k)
        print(f"Mean precision@{args.k}: {i2i.mean_precision_at_k:.3f}")
        print(f"Top-1 accuracy:        {i2i.top1_accuracy:.3f}")
        i2i.confusions.to_csv(args.out_dir / "confusions.csv", index=False)

        print("\nTop 15 confused class pairs (true -> predicted, count):")
        for _, row in i2i.confusions.head(15).iterrows():
            print(f"  {row['true_label']} -> {row['predicted_label']}: {row['count']}")

        heatmap_path = args.out_dir / "confusion_heatmap.png"
        plot_confusion_heatmap(i2i.confusions, heatmap_path)
        print(f"\nSaved confusion heatmap to {heatmap_path}")

    print(f"\nSaved per-class results to {args.out_dir}")


if __name__ == "__main__":
    main()
