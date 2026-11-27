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


def test_retrieve_in_blocks_matches_one_block(monkeypatch):
    rng = np.random.default_rng(0)
    t, c = rng.normal(size=(7, 4)), rng.normal(size=(9, 4))
    ty, cy = rng.integers(0, 2, (7, 3)), rng.integers(0, 2, (9, 3))
    subject_t, subject_c = rng.integers(0, 3, 7), rng.integers(0, 3, 9)

    def exclude(rows):
        return subject_t[rows, None] == subject_c[None, :]

    sim, overlap = retrieval.cosine(t, c), retrieval.jaccard(ty, cy)
    for by_labels in (True, False):
        want = retrieval.top_k(
            2, sim, exclude(slice(None)), overlap if by_labels else None
        )
        monkeypatch.setattr(retrieval, "CHUNK", 3)
        top, top_sim, top_jac = retrieval.retrieve(2, t, c, exclude, ty, cy, by_labels)
        assert top.tolist() == want.tolist()
        # a block's shape can change the cosine's rounding (BLAS path), not the rank
        np.testing.assert_allclose(
            top_sim, np.take_along_axis(sim, want, 1), rtol=0, atol=1e-12
        )
        np.testing.assert_array_equal(top_jac, np.take_along_axis(overlap, want, 1))
    top, _, top_jac = retrieval.retrieve(2, t, c, exclude)
    assert top.tolist() == retrieval.top_k(2, sim, exclude(slice(None))).tolist()
    assert top_jac is None
