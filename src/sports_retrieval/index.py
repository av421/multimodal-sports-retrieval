"""Build, save, and load a FAISS index over CLIP image embeddings.

Embeddings are L2-normalized (see embed.py), so an IndexFlatIP (inner
product) is equivalent to exact cosine-similarity search. Flat/exact is
plenty fast at ~14k vectors -- no need for an approximate index.
"""

from __future__ import annotations

import csv
import multiprocessing as mp
from pathlib import Path
from types import TracebackType

import faiss
import numpy as np

from sports_retrieval.data import ImageRecord


def build_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index(index: faiss.Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: Path) -> faiss.Index:
    return faiss.read_index(str(path))


def save_metadata(records: list[ImageRecord], path: Path) -> None:
    """Write metadata in the same row order as the FAISS index, so row i of
    the index corresponds to row i of this file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label", "split"])
        for r in records:
            writer.writerow([str(r.filepath), r.label, r.split])


def load_metadata(path: Path) -> list[ImageRecord]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return [
            ImageRecord(filepath=Path(row["filepath"]), label=row["label"], split=row["split"])
            for row in reader
        ]


def _search_worker(index_path: str, conn: "mp.connection.Connection") -> None:
    import faiss  # imported only in this process, which never touches torch

    index = faiss.read_index(index_path)
    while True:
        msg = conn.recv()
        if msg is None:
            break
        cmd, payload = msg
        if cmd == "search":
            queries, k = payload
            conn.send(index.search(queries, k))
        elif cmd == "reconstruct_all":
            conn.send(index.reconstruct_n(0, index.ntotal))
    conn.close()


class IndexProcess:
    """Runs FAISS search in a dedicated subprocess.

    faiss and torch each bundle their own OpenMP runtime. Calling
    index.search() in the same process as a torch/MPS model reliably
    corrupts memory on macOS/Apple Silicon -- confirmed via repeated
    segfaults during development, not just the benign "duplicate libomp"
    warning, and not avoidable via KMP_DUPLICATE_LIB_OK or thread-count
    tweaks (both were tried and still crashed under load). Building the
    index (add/write, no search) is unaffected and does not need this.

    Isolating the search call in its own process, which never imports
    torch, sidesteps the conflict entirely.

    Caller constraint: on macOS, multiprocessing's required "spawn" start
    method re-executes the launching script's top-level code in the child
    (to rebuild sys.modules["__main__"] for unpickling) -- everything
    outside an `if __name__ == "__main__":` guard runs again there. If the
    outermost script (the one run as `python foo.py`) imports anything that
    imports torch at module level, the child inherits it too, recreating
    the exact conflict this class exists to avoid. Keep such imports inside
    main()/`if __name__ == "__main__":` in any script that constructs an
    IndexProcess (directly or via SearchEngine).
    """

    def __init__(self, index_path: Path) -> None:
        ctx = mp.get_context("spawn")
        self._parent_conn, child_conn = ctx.Pipe()
        self._process = ctx.Process(
            target=_search_worker, args=(str(index_path), child_conn), daemon=True
        )
        self._process.start()

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        self._parent_conn.send(("search", (queries, k)))
        return self._parent_conn.recv()

    def reconstruct_all(self) -> np.ndarray:
        """Return every vector stored in the index, in row order. Exact for
        IndexFlatIP -- it stores the raw vectors, no approximation involved."""
        self._parent_conn.send(("reconstruct_all", None))
        return self._parent_conn.recv()

    def close(self) -> None:
        if self._process.is_alive():
            self._parent_conn.send(None)
            self._process.join(timeout=5)

    def __enter__(self) -> "IndexProcess":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
