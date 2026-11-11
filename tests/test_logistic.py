"""Per-label logistic regression and the validation F1 threshold."""

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from labrag.classifiers import logistic


def _brute_force_threshold(y, p):
    def f1(t):
        yhat = p >= t
        tp = (yhat & (y == 1)).sum()
        return 2 * tp / (yhat.sum() + y.sum())

    candidates = np.unique(p)
    return candidates[np.argmax([f1(t) for t in candidates])]


def test_f1_threshold_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(50):
        y = rng.integers(0, 2, size=40)
        y[0] = 1  # at least one positive
        p = np.round(rng.random(40), 2)  # with ties
        assert logistic.f1_threshold(y, p) == _brute_force_threshold(y, p)


def test_f1_threshold_ignores_zero_precision_and_recall():
    # At t=0.9 precision = recall = 0, so F1 is 0/0. The preprint's np.argmax picked
    # that NaN point; the best threshold is 0.2 (F1 = 2/3).
    assert logistic.f1_threshold(np.array([1, 0]), np.array([0.2, 0.9])) == 0.2


def test_fit_and_predict_match_sklearn():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 8)).astype(np.float32)
    y = (X[:, 0] + rng.normal(size=300) > 0).astype(int)
    coef, intercept, _ = logistic.fit(X, y)
    ref = LogisticRegression(**logistic.PARAMS).fit(X.astype(np.float64), y)
    p = logistic.predict_proba(X, coef[None], np.array([intercept]))[:, 0]
    np.testing.assert_allclose(p, ref.predict_proba(X.astype(np.float64))[:, 1])


def test_fit_refuses_to_stop_before_converging(monkeypatch):
    monkeypatch.setitem(logistic.PARAMS, "max_iter", 1)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 8))
    with pytest.raises(ConvergenceWarning):
        logistic.fit(X, (X[:, 0] > 0).astype(int))
