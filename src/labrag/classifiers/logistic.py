"""Per-label logistic regression on image embeddings (the preprint's classifier).

The preprint's objective (L2, C=1, no scaling, no class weights), but solved to
convergence: the preprint's saga at 500 iterations never converged, sklearn's default
tol stops ~30% short, and lbfgs stalls on float32. So lbfgs, float64, tol=1e-8; not
converging is an error.

``fit_tuned`` is the tuned control: features standardized on train, C picked per label
by val AUROC from a grid, the same solver otherwise.
"""

import warnings

import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score

PARAMS = {
    "l1_ratio": 0.0,  # pure L2 (sklearn >= 1.8 spelling of penalty="l2")
    "C": 1.0,
    "solver": "lbfgs",
    "tol": 1e-8,
    "max_iter": 10_000,
}
C_GRID = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)
TUNED_PARAMS = {
    **PARAMS,
    "C": list(C_GRID),
    "scaling": "standardize each feature by its train mean and std",
    "select": "per label, the C with the best val AUROC",
}


def fit(
    X: np.ndarray, y: np.ndarray, C: float = PARAMS["C"]
) -> tuple[np.ndarray, float, int]:
    """Fit one label. Returns (coef of shape (dim,), intercept, iterations)."""
    model = LogisticRegression(**(PARAMS | {"C": C}))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(np.asarray(X, dtype=np.float64), y)
    return model.coef_[0], float(model.intercept_[0]), int(model.n_iter_[0])


def fit_tuned(
    X: np.ndarray, Y: np.ndarray, train: np.ndarray, val: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit every label of Y on rows ``train`` at each C of C_GRID; keep the C with the
    best AUROC on rows ``val``. Returns coef (labels, dim) and intercept (labels,) for
    the unstandardized X, and the chosen C and iterations per label."""
    X = np.asarray(X, dtype=np.float64)
    mean, std = X[train].mean(0), X[train].std(0)
    assert (std > 0).all(), "a feature is constant on train"
    Z = (X - mean) / std
    out = []
    for y in Y.T:
        assert 0 < y[val].sum() < val.sum(), "a label lacks a class on val"
        best = (-np.inf,)
        for C in C_GRID:
            coef, intercept, n_iter = fit(Z[train], y[train], C)
            auroc = roc_auc_score(y[val], Z[val] @ coef + intercept)
            if auroc > best[0]:
                best = (auroc, coef, intercept, C, n_iter)
        _, coef, intercept, C, n_iter = best
        # Fold the standardization into the weights: predict_proba then takes raw X.
        out.append((coef / std, intercept - coef @ (mean / std), C, n_iter))
    coef, intercept, C, n_iter = (np.array(v) for v in zip(*out))
    return coef, intercept, C, n_iter


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
