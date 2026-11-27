"""Ranking corpus studies for each target study (docs/revision-plan.md §3).

Image similarity: cosine of the embeddings. Label ranking: Jaccard of positive label
sets ("No Finding" included), ties by image similarity; two empty sets have Jaccard 1
(the preprint's implicit "Other"). Remaining ties go to the lower corpus row. Studies of
the target's patient are never returned.
"""

from collections.abc import Callable

import numpy as np


def cosine(targets: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """(n_targets, n_corpus) cosine similarity, computed in float64."""
    t = targets.astype(np.float64)
    c = corpus.astype(np.float64)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    return t @ c.T


def jaccard(targets: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """(n_targets, n_corpus) Jaccard overlap of 0/1 label rows; empty vs empty is 1."""
    t = targets.astype(np.int64)
    c = corpus.astype(np.int64)
    inter = t @ c.T
    union = t.sum(1)[:, None] + c.sum(1)[None, :] - inter
    return np.where(union == 0, 1.0, inter / np.maximum(union, 1))


# Targets per block: bounds the (targets, corpus) matrices in memory. Cosines can round
# differently with the block's shape, so every val and test set stays one block.
CHUNK = 8192


def retrieve(
    k: int,
    targets: np.ndarray,
    corpus: np.ndarray,
    exclude: Callable[[slice], np.ndarray],
    target_y: np.ndarray | None = None,
    corpus_y: np.ndarray | None = None,
    by_labels: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """top_k over blocks of CHUNK targets: (n_targets, k) corpus indices, their cosine
    and their Jaccard (None without labels).

    ``exclude(rows)`` gives the (block, n_corpus) mask for target rows ``rows``. With
    labels, the ranking is Jaccard then cosine if ``by_labels``, else cosine alone.
    """
    top, sims, jacs = [], [], []
    for start in range(0, len(targets), CHUNK):
        rows = slice(start, start + CHUNK)
        sim = cosine(targets[rows], corpus)
        overlap = None if target_y is None else jaccard(target_y[rows], corpus_y)
        t = top_k(k, sim, exclude(rows), overlap if by_labels else None)
        top.append(t)
        sims.append(np.take_along_axis(sim, t, 1))
        if overlap is not None:
            jacs.append(np.take_along_axis(overlap, t, 1))
    if not top:  # no targets
        top, sims = [np.zeros((0, k), dtype=np.int64)], [np.zeros((0, k))]
        jacs = [np.zeros((0, k))] if target_y is not None else []
    return (
        np.concatenate(top),
        np.concatenate(sims),
        np.concatenate(jacs) if jacs else None,
    )


def top_k(
    k: int,
    similarity: np.ndarray,
    exclude: np.ndarray,
    overlap: np.ndarray | None = None,
) -> np.ndarray:
    """(n_targets, k) corpus indices, best first.

    Ranked by ``overlap`` (if given) then ``similarity``, then corpus row. ``exclude``
    is a boolean (n_targets, n_corpus) mask of pairs never to return.
    """
    n_corpus = similarity.shape[1]
    rows = np.arange(n_corpus)
    out = np.empty((len(similarity), k), dtype=np.int64)
    for i in range(len(similarity)):
        keep = np.flatnonzero(~exclude[i])
        if len(keep) < k:
            raise ValueError(f"target {i}: only {len(keep)} corpus studies allowed")
        keys = [rows[keep], -similarity[i, keep]]  # np.lexsort: last key is primary
        if overlap is not None:
            keys.append(-overlap[i, keep])
        out[i] = keep[np.lexsort(keys)[:k]]
    return out
