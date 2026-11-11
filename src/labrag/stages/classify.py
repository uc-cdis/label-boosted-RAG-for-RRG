"""Stage ``classify``: per-label image classifiers for one report section's labels.

Reads ``embeddings/<dataset>/<encoder>/`` and ``labels/<dataset>/<section>/<labeler>/``
and writes ``classifiers/<dataset>/<encoder>.<classifier>/<section>/<labeler>[.u0]/``:
- ``model.npz``: labels, coef (n_labels, dim), intercept, threshold; enough to predict
  on any embeddings from the same encoder (e.g. other datasets, J and §4.3);
- ``probs.csv``: study_id, role, one probability column per label;
- ``predictions.csv``: study_id, one 0/1 column per label (prob >= threshold), the
  same layout as the reference labels.csv;
- ``metrics.csv``: per label, counts, threshold, and AUROC/F1/precision/recall on val
  and on the official valid split.

Rows are the studies that have reference labels for this section: classifiers are fit
within each section, never on another section's labels. Roles come from the adapter
(``classifier_roles``): fit on ``train``, thresholds on ``val``; every other role is
predicted only. Uncertain labels follow the CheXpert policies (Irvin et al. 2019):
``ones`` (U-Ones, the preprint's default) maps positive or uncertain to 1; ``zeros``
(U-Zeros) maps only positive to 1 and adds ``.u0`` to the labeler directory. Negative
and not mentioned are always 0. No out-of-fold predictions yet (§2 "Leakage
controls"): nothing trains on these predictions so far.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from labrag import artifacts, paths
from labrag.classifiers import logistic
from labrag.data import DATASETS

CLASSIFIERS = ("lr",)
# CheXbert codes counted as positive, per uncertain-label policy.
POSITIVE = {"ones": (1, -1), "zeros": (1,)}
BINARIZE = {
    "ones": "positive or uncertain = 1, negative or blank = 0",
    "zeros": "positive = 1, uncertain or negative or blank = 0",
}
LABELER_SUFFIX = {"ones": "", "zeros": ".u0"}


def run(
    dataset: str,
    section: str,
    encoder: str = "raddino",
    labeler: str = "chexbert14",
    classifier: str = "lr",
    uncertain: str = "ones",
    overwrite: bool = False,
    cxr_root: Path | None = None,
) -> Path:
    cxr_root = cxr_root or paths.CXR_ROOT
    adapter = DATASETS[dataset]
    emb_dir = paths.SHARED_ROOT / "embeddings" / dataset / encoder
    lab_dir = paths.SHARED_ROOT / "labels" / dataset / section / labeler
    inputs = {
        "embeddings": artifacts.artifact_record(emb_dir),
        "labels": artifacts.artifact_record(lab_dir),
        "sources": [artifacts.file_record(cxr_root / adapter.SOURCE_CSV)],
        "splits": [
            artifacts.file_record(paths.SPLITS_DIR / f) for f in adapter.SPLIT_FILES
        ],
    }
    config = {
        "stage": "classify",
        "dataset": dataset,
        "section": section,
        "encoder": encoder,
        "labeler": labeler,
        "classifier": classifier,
        "params": logistic.PARAMS,
        "binarize": BINARIZE[uncertain],
        "fit": "train",
        "threshold": "max F1 on val; predict prob >= threshold",
    }

    def build(out: Path) -> dict:
        labels = pd.read_csv(lab_dir / "labels.csv")
        names = list(labels.columns.drop("study_id"))
        studies = adapter.load_studies(cxr_root).set_index("study_id")
        studies["role"] = adapter.classifier_roles(studies, paths.SPLITS_DIR)
        studies = studies.loc[labels.study_id]  # KeyError if a label has no study

        # The embedded image must be the one the adapter picks for the study today.
        embedded = pd.read_csv(emb_dir / "studies.csv").set_index("study_id")
        embedded["row"] = np.arange(len(embedded))
        embedded = embedded.loc[labels.study_id]
        assert (embedded.image_path == studies.image_path).all(), "image mismatch"
        X = np.load(emb_dir / "embeddings.npy")[embedded.row.to_numpy()]
        Y = labels[names].isin(POSITIVE[uncertain]).to_numpy(dtype=int)
        role = studies.role.to_numpy()
        train, val, official = role == "train", role == "val", role == "official_valid"

        fits = [logistic.fit(X[train], Y[train, j]) for j in range(len(names))]
        coef = np.stack([f[0] for f in fits])
        intercept = np.array([f[1] for f in fits])
        P = logistic.predict_proba(X, coef, intercept)
        threshold = np.array(
            [logistic.f1_threshold(Y[val, j], P[val, j]) for j in range(len(names))]
        )
        pred = (P >= threshold).astype(int)

        metrics = pd.DataFrame(
            {
                "label": names,
                "n_iter": [f[2] for f in fits],
                "train_pos": Y[train].sum(0),
                "threshold": threshold,
                **_scores("val", Y[val], P[val], pred[val]),
                **_scores("official_valid", Y[official], P[official], pred[official]),
            }
        )
        np.savez(
            out / "model.npz",
            labels=np.array(names),
            coef=coef,
            intercept=intercept,
            threshold=threshold,
        )
        probs = pd.DataFrame(P, columns=names)
        probs.insert(0, "role", role)
        probs.insert(0, "study_id", labels.study_id)
        probs.to_csv(out / "probs.csv", index=False)
        preds = pd.DataFrame(pred, columns=names)
        preds.insert(0, "study_id", labels.study_id)
        preds.to_csv(out / "predictions.csv", index=False)
        metrics.to_csv(out / "metrics.csv", index=False)
        print(metrics.to_string(index=False))
        return {
            "n_studies": len(labels),
            "n_by_role": pd.Series(role).value_counts().sort_index().to_dict(),
        }

    return artifacts.publish(
        f"classifiers/{dataset}/{encoder}.{classifier}/{section}/"
        f"{labeler}{LABELER_SUFFIX[uncertain]}",
        config,
        inputs,
        build,
        overwrite,
    )


def _scores(prefix: str, Y: np.ndarray, P: np.ndarray, pred: np.ndarray) -> dict:
    """Per-label positives, AUROC, F1, precision, recall (NaN when undefined)."""
    cols = {k: [] for k in ("pos", "auroc", "f1", "precision", "recall")}
    for y, p, yhat in zip(Y.T, P.T, pred.T):
        both = 0 < y.sum() < len(y)
        cols["pos"].append(int(y.sum()))
        cols["auroc"].append(roc_auc_score(y, p) if both else np.nan)
        cols["f1"].append(f1_score(y, yhat, zero_division=np.nan))
        cols["precision"].append(precision_score(y, yhat, zero_division=np.nan))
        cols["recall"].append(recall_score(y, yhat, zero_division=np.nan))
    return {f"{prefix}_{k}": v for k, v in cols.items()}
