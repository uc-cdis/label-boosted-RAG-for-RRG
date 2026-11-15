"""Per-label logistic regression on image embeddings (the preprint's classifier).

The preprint's objective (L2, C=1, no scaling, no class weights), but solved to
convergence: the preprint's saga at 500 iterations never converged, sklearn's default
tol stops ~30% short, and lbfgs stalls on float32. So lbfgs, float64, tol=1e-8; not
converging is an error.
"""

import warnings

import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

PARAMS = {
    "l1_ratio": 0.0,  # pure L2 (sklearn >= 1.8 spelling of penalty="l2")
    "C": 1.0,
    "solver": "lbfgs",
    "tol": 1e-8,
    "max_iter": 10_000,
}


def fit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Fit one label. Returns (coef of shape (dim,), intercept, iterations)."""
    model = LogisticRegression(**PARAMS)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(np.asarray(X, dtype=np.float64), y)
    return model.coef_[0], float(model.intercept_[0]), int(model.n_iter_[0])


def predict_proba(X: np.ndarray, coef: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    """Probabilities, shape (n, n_labels), from stacked coef (n_labels, dim)."""
    return expit(np.asarray(X, dtype=np.float64) @ coef.T + intercept)


def f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """The probability threshold with the highest F1 on (y, p); predict ``p >= t``.

    The preprint's "pr" threshold (argmax F1 over the precision-recall curve), with two
    fixes: points where precision + recall = 0 score F1 = 0 instead of NaN (np.argmax
    picks NaN), and predictions use ``>=``, the rule the curve's points are defined by
    (the preprint used ``>``, which drops the studies at the threshold itself).
    """
    precision, recall, thresholds = precision_recall_curve(y, p)
    precision, recall = precision[:-1], recall[:-1]  # last point has no threshold
    denom = precision + recall
    f1 = np.divide(
        2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0
    )
    return float(thresholds[np.argmax(f1)])
