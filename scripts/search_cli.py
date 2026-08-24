"""Quick manual test of SearchEngine from the command line.

Usage:
    python scripts/search_cli.py --text "diving into a pool"
    python scripts/search_cli.py --image path/to/photo.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def main() -> None:
    # Imported here, not at module level: multiprocessing's spawn start method
    # (required on macOS -- see IndexProcess) re-executes this script's
    # top-level code in the worker subprocess to rebuild sys.modules["__main__"].
    # A module-level import of SearchEngine would pull torch into that
    # subprocess too, recreating the exact conflict IndexProcess exists to
    # avoid. Code inside main() only runs when __name__ == "__main__", which
    # is never true in the spawned child.
    from sports_retrieval.search import SearchEngine

    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str)
    group.add_argument("--image", type=Path)
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    with SearchEngine(
        index_path=ARTIFACTS_DIR / "index.faiss",
        metadata_path=ARTIFACTS_DIR / "metadata.csv",
    ) as engine:
        results = (
            engine.search_text(args.text, k=args.k)
            if args.text
            else engine.search_image(args.image, k=args.k)
        )

    for rank, r in enumerate(results, start=1):
        print(f"{rank}. [{r.score:.3f}] {r.label} ({r.split}) -- {r.filepath}")


if __name__ == "__main__":
    main()
