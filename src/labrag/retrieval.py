"""Ranking corpus studies for each target study (docs/revision-plan.md §3).

- **Image similarity** (retrieval only): cosine similarity of the image embeddings.
- **Label ranking** (LaB-RAG, filter only): Jaccard overlap of the positive label sets,
  ties broken by image similarity. "No Finding" is one of the labels. A study with no
  positive label at all (CheXbert leaves No Finding blank when the report describes an
  abnormality outside its 14 labels) has the empty set, and two empty sets have
  Jaccard 1. That is the same ranking as the preprint's implicit "Other" label.

Remaining ties go to the lower corpus row, so the ranking is deterministic. Excluded
pairs (the same patient, which includes the target study itself) are never returned.
"""

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
