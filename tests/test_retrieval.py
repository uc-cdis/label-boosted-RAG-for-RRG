"""Ranking: Jaccard with empty sets, tie-breaking, exclusion."""

import numpy as np
import pytest

from labrag import retrieval


def test_jaccard_empty_sets_match_each_other_only():
    t = np.array([[0, 0, 0], [1, 1, 0]])
    c = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1]])
    np.testing.assert_allclose(retrieval.jaccard(t, c), [[1, 0, 0, 0], [0, 0.5, 1, 0]])


def test_cosine_ignores_norm():
    t = np.array([[1.0, 0.0]])
    c = np.array([[5.0, 0.0], [1.0, 1.0], [0.0, 2.0]])
    np.testing.assert_allclose(retrieval.cosine(t, c), [[1, 2**-0.5, 0]])


def test_top_k_overlap_then_similarity_then_row():
    sim = np.array([[0.9, 0.1, 0.5, 0.5, 0.8]])
    overlap = np.array([[0.5, 1.0, 1.0, 1.0, 0.5]])
    exclude = np.zeros_like(sim, dtype=bool)
    assert retrieval.top_k(5, sim, exclude, overlap).tolist() == [[2, 3, 1, 0, 4]]
    assert retrieval.top_k(3, sim, exclude).tolist() == [[0, 4, 2]]


def test_top_k_never_returns_excluded():
    sim = np.array([[0.9, 0.1, 0.5]])
    exclude = np.array([[True, False, False]])
    assert retrieval.top_k(2, sim, exclude).tolist() == [[2, 1]]
    with pytest.raises(ValueError):
        retrieval.top_k(3, sim, exclude)
