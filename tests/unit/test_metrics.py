import numpy as np

from meld.benchmarks.metrics import compute_compute_savings, compute_equivalence_gap, confusion_cosine_similarity


def test_confusion_cosine_similarity_matches_identical_inputs():
    confusion = np.array([[8, 2], [1, 9]])
    assert confusion_cosine_similarity(confusion, confusion) == 1.0
    assert compute_equivalence_gap(confusion, confusion) == 0.0


def test_compute_savings_percent_uses_full_vs_delta():
    assert compute_compute_savings(40.0, 100.0) == 60.0
