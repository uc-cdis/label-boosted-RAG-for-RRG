"""Attention head: shapes, learns a planted signal, same seed gives the same model."""

import numpy as np
import pytest
import torch

from labrag.classifiers import attention, quant


@pytest.fixture
def small(monkeypatch):
    params = attention.PARAMS | {
        "ffn": 32,
        "batch_size": 16,
        "max_epochs": 8,
        "lr": 3e-3,
    }
    monkeypatch.setattr(attention, "PARAMS", params)
    attention.deterministic()
    rng = np.random.default_rng(0)
    n, t, d, labels = 600, 6, 16, 3
    Y = rng.integers(0, 2, size=(n, labels))
    x = rng.normal(size=(n, t, d)).astype(np.float32)
    # Label j shows as a shift of channel j in one token (a different one per label).
    for j in range(labels):
        x[:, 1 + j, j] += 3.0 * Y[:, j]
    store = quant.empty("fp16", (n, t, d), torch.device("cpu"))
    store.parts["x"][:] = torch.from_numpy(x).half()
    role = np.array(["train"] * 500 + ["val"] * 100)
    return store, Y, role == "train", role == "val"


def test_forward_shape():
    head = attention.Head(n_labels=3, dim=16, heads=8, ffn=32, dropout=0.0)
    pooled, weights = head.attention(torch.randn(5, 7, 16))
    assert head(torch.randn(5, 7, 16)).shape == (5, 3)
    assert pooled.shape == (5, 3, 16) and weights.shape == (5, 8, 3, 7)
    torch.testing.assert_close(weights.sum(-1), torch.ones(5, 8, 3))


def test_learns_and_is_deterministic(small):
    store, Y, train, val = small
    model, history = attention.fit(store, Y, train, val, seed=1, log=lambda _: None)
    again, history_again = attention.fit(
        store, Y, train, val, seed=1, log=lambda _: None
    )
    assert history == history_again
    rows = torch.arange(len(store))
    P = attention.predict_proba(model, store, rows)
    np.testing.assert_array_equal(P, attention.predict_proba(again, store, rows))
    assert attention.mean_auroc(Y[val], P[val]) > 0.9
    assert max(h["val_auroc"] for h in history) == attention.mean_auroc(Y[val], P[val])


def test_out_of_fold_predicts_each_fold_from_the_other_folds(small):
    from labrag.stages import classify

    store, Y, train, val = small
    fold = np.where(train, np.arange(len(Y)) % 3, -1)

    def decide(P):
        return np.full(P.shape[1], 0.5), (P >= 0.5).astype(int)

    P, pred, summary = classify.out_of_fold(store, Y, fold, val, 0, decide)
    assert [s["fold"] for s in summary] == [0, 1, 2]
    assert sum(s["n_held_out"] for s in summary) == train.sum()
    assert np.isnan(P[~train]).all() and not np.isnan(P[train]).any()
    assert attention.mean_auroc(Y[train], P[train]) > 0.75  # planted signal, held out
    assert (pred[train] == (P[train] >= 0.5)).all()
    # Fold 0's rows come from the model trained on folds 1 and 2 only.
    model, _ = attention.fit(store, Y, train & (fold != 0), val, 0)
    rows = torch.as_tensor(np.flatnonzero(fold == 0))
    np.testing.assert_array_equal(
        P[fold == 0], attention.predict_proba(model, store, rows)
    )
