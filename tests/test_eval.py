from sports_retrieval.eval import precision_at_k, recall_at_k


def test_precision_at_k_counts_matches_in_top_k() -> None:
    retrieved = ["golf", "golf", "tennis", "golf", "archery"]
    assert precision_at_k(retrieved, "golf", k=5) == 3 / 5
    assert precision_at_k(retrieved, "golf", k=2) == 1.0
    assert precision_at_k(retrieved, "golf", k=3) == 2 / 3


def test_precision_at_k_empty_retrieval_is_zero() -> None:
    assert precision_at_k([], "golf", k=5) == 0.0


def test_recall_at_k_normalizes_by_total_relevant() -> None:
    retrieved = ["golf", "golf", "tennis", "golf", "archery"]
    # 3 golf images retrieved out of 10 total golf images in the pool
    assert recall_at_k(retrieved, "golf", k=5, total_relevant=10) == 3 / 10


def test_recall_at_k_zero_relevant_is_zero_not_division_error() -> None:
    assert recall_at_k(["golf"], "golf", k=5, total_relevant=0) == 0.0


def test_recall_at_k_respects_k_cutoff() -> None:
    retrieved = ["golf", "golf", "golf", "golf", "golf"]
    # only first 2 count toward recall when k=2, even though all 5 match
    assert recall_at_k(retrieved, "golf", k=2, total_relevant=5) == 2 / 5
