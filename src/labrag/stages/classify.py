"""Stage ``classify``: per-label image classifiers for one report section's labels.

Writes ``classifiers/<dataset>/<encoder>.<classifier>/<section>/<labeler>[.u0]/``:
model.npz (coef, intercept, threshold per label), probs.csv (study_id, role, one column
per label), predictions.csv (0/1, the layout of labels.csv), metrics.csv (per label, on
val and official valid).

Fit within each section, on the adapter's ``train`` role; max-F1 thresholds on ``val``;
other roles predicted only. Classifiers: ``lr`` (per-label logistic regression on the
CLS embedding; CPU) and ``attn`` (per-label attention pooling over the patch tokens of
``embed --feature tokens``, trained jointly; 1 GPU), the latter written to
``<encoder>.attn-<tokens>-seed<seed>`` (tokens = the storage format they are read in). Uncertain labels (Irvin et al. 2019): ``zeros`` (U-Zeros,
default, ``.u0``) counts only positive as 1; ``ones`` (U-Ones, the preprint) also
uncertain. No out-of-fold predictions yet: nothing trains on them so far.

Pathologies win (``consistent``): No Finding is cleared wherever a label other than
Support Devices is predicted, as CheXbert's reference labels do (all but 40 of 46,759).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from labrag import artifacts, paths
from labrag.classifiers import attention, logistic, quant
from labrag.data import DATASETS

CLASSIFIERS = ("lr", "attn")
TOKEN_CHUNK = 256  # studies packed at a time while loading tokens
# CheXbert codes counted as positive, per uncertain-label policy.
POSITIVE = {"ones": (1, -1), "zeros": (1,)}
BINARIZE = {
    "ones": "positive or uncertain = 1, negative or blank = 0",
    "zeros": "positive = 1, uncertain or negative or blank = 0",
}
LABELER_SUFFIX = {"ones": "", "zeros": ".u0"}
NO_FINDING = "No Finding"
NOT_PATHOLOGY = ("No Finding", "Support Devices")
CONSISTENCY = "No Finding = 0 when any label other than Support Devices is predicted"


def run(
    dataset: str,
    section: str,
    encoder: str = "raddino",
    labeler: str = "chexbert14",
    classifier: str = "lr",
    uncertain: str = "zeros",
    tokens: str | None = None,
    seed: int | None = None,
    overwrite: bool = False,
    cxr_root: Path | None = None,
) -> Path:
    attn = classifier == "attn"
    if not attn and (tokens is not None or seed is not None):
        raise SystemExit("classify: --tokens and --seed are for --classifier attn")
    tokens = tokens or "fp16"
    seed = 0 if seed is None else seed
    cxr_root = cxr_root or paths.CXR_ROOT
    adapter = DATASETS[dataset]
    emb_dir = paths.SHARED_ROOT / "embeddings" / dataset / encoder
    if attn:
        emb_dir = emb_dir.with_name(f"{encoder}.tokens") / section
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
        "params": attention.PARAMS if attn else logistic.PARAMS,
        "binarize": BINARIZE[uncertain],
        "fit": "train",
        "threshold": "max F1 on val; predict prob >= threshold",
        "consistency": CONSISTENCY,
    }
    name = f"{encoder}.{classifier}"
    if attn:
        config |= {"tokens": quant.DESCRIPTION[tokens], "seed": seed}
        name += f"-{tokens}-seed{seed}"

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
        rows = embedded.row.to_numpy()
        Y = labels[names].isin(POSITIVE[uncertain]).to_numpy(dtype=int)
        role = studies.role.to_numpy()
        train, val, official = role == "train", role == "val", role == "official_valid"

        if attn:
            attention.deterministic()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            X = _load_tokens(emb_dir / "tokens.npy", rows, tokens, device)
            model, history = attention.fit(X, Y, train, val, seed)
            P = attention.predict_proba(model, X, torch.arange(len(X), device=device))
            extra = {}
        else:
            X = np.load(emb_dir / "embeddings.npy")[rows]
            fits = [logistic.fit(X[train], Y[train, j]) for j in range(len(names))]
            coef = np.stack([f[0] for f in fits])
            intercept = np.array([f[1] for f in fits])
            P = logistic.predict_proba(X, coef, intercept)
            extra = {"n_iter": [f[2] for f in fits]}
        threshold = np.array(
            [logistic.f1_threshold(Y[val, j], P[val, j]) for j in range(len(names))]
        )
        pred = consistent((P >= threshold).astype(int), names)

        metrics = pd.DataFrame(
            {
                "label": names,
                **extra,
                "train_pos": Y[train].sum(0),
                "threshold": threshold,
                **_scores("val", Y[val], P[val], pred[val]),
                **_scores("official_valid", Y[official], P[official], pred[official]),
            }
        )
        if attn:
            state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(
                {"labels": names, "state_dict": state, "threshold": threshold},
                out / "model.pt",
            )
            pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        else:
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
        summary = {
            "n_studies": len(labels),
            "n_by_role": pd.Series(role).value_counts().sort_index().to_dict(),
        }
        if attn:
            best = max(history, key=lambda h: h["val_auroc"])
            summary |= {
                "epochs": len(history),
                "best_epoch": best["epoch"],
                "val_mean_auroc": best["val_auroc"],
                "device": str(device),
            }
        return summary

    return artifacts.publish(
        f"classifiers/{dataset}/{name}/{section}/{labeler}{LABELER_SUFFIX[uncertain]}",
        config,
        inputs,
        build,
        overwrite,
    )


def _load_tokens(
    path: Path, rows: np.ndarray, fmt: str, device: torch.device
) -> quant.Packed:
    """Rows ``rows`` of a float16 tokens.npy, packed in format ``fmt`` on ``device``."""
    stored = np.load(path, mmap_mode="r")
    packed = quant.empty(fmt, (len(rows), *stored.shape[1:]), device)
    for start in range(0, len(rows), TOKEN_CHUNK):
        chunk = rows[start : start + TOKEN_CHUNK]
        x = torch.from_numpy(np.ascontiguousarray(stored[chunk])).to(device)
        for k, v in quant.pack(fmt, x).items():
            packed.parts[k][start : start + len(chunk)] = v
    return packed


def consistent(pred: np.ndarray, names: list[str]) -> np.ndarray:
    """Pathologies win over No Finding: clear No Finding where any pathology is 1."""
    pred = pred.copy()
    pathology = [j for j, n in enumerate(names) if n not in NOT_PATHOLOGY]
    pred[pred[:, pathology].any(axis=1), names.index(NO_FINDING)] = 0
    return pred


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
