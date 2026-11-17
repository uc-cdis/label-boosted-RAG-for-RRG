"""Stage ``classify``: per-label image classifiers for one report section's labels.

Writes ``classifiers/<dataset>/<encoder>.<classifier>/<section>/<labeler>[.u0]/``:
model.npz (coef, intercept, threshold per label; C for lr-tuned), probs.csv (study_id,
role, one column per label), predictions.csv (0/1, the layout of labels.csv),
metrics.csv (per label, on val and official valid).

Fit within each section, on the adapter's ``train`` role; max-F1 thresholds on ``val``;
other roles predicted only. Classifiers:

- ``lr``: per-label logistic regression on the CLS embedding (the preprint's; CPU).
- ``lr-tuned``: ``lr`` on standardized features with C picked per label on val (CPU).
- ``attn``: per-label attention pooling over the tokens of ``embed --feature tokens``
  (CLS + patches), trained jointly (1 GPU).
- ``attn-cls``: ``attn`` given only the CLS token, the control for the patch tokens.

The ``attn`` ones are written to ``<encoder>.<classifier>-<tokens>-seed<seed>`` (tokens =
the storage format they are read in) with model.pt instead of model.npz. A float16
token store can be read in any format; a packed store (embed's) only in its own.

Uncertain labels (Irvin et al. 2019): ``zeros`` (U-Zeros, default, ``.u0``) counts only
positive as 1; ``ones`` (U-Ones, the preprint) also uncertain.

``--folds K`` (attn only, name suffix ``-oofK``) also writes out-of-fold predictions for
the train role (oof_probs.csv, oof_predictions.csv): train patients split into K folds,
each predicted by the same training on the other folds (epoch and thresholds on val),
for wherever predicted labels stand in for reference ones on studies the classifier
trains on.

Pathologies win (``consistent``): No Finding is cleared wherever a label other than
Support Devices is predicted, as CheXbert's reference labels do (all but 40 of 46,759).
PadChest's ``normal`` is cleared wherever any other label is predicted (its labels pair
normal with another label in 155 of 108,746 studies, devices included).

A labels-only dataset (PadChest) takes no ``--section``: it reads
``labels/<dataset>/<labeler>/`` and writes ``classifiers/<dataset>/<name>/<labeler>/``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from labrag import artifacts, paths
from labrag.classifiers import attention, logistic, quant
from labrag.data import DATASETS, check_section, labels_only, padchest

CLASSIFIERS = ("lr", "lr-tuned", "attn", "attn-cls")
TOKEN_CHUNK = 256  # studies packed at a time while loading tokens
# CheXbert codes counted as positive, per uncertain-label policy.
POSITIVE = {"ones": (1, -1), "zeros": (1,)}
BINARIZE = {
    "ones": "positive or uncertain = 1, negative or blank = 0",
    "zeros": "positive = 1, uncertain or negative or blank = 0",
}
LABELER_SUFFIX = {"ones": "", "zeros": ".u0"}
PARAMS = {
    "lr": logistic.PARAMS,
    "lr-tuned": logistic.TUNED_PARAMS,
    "attn": attention.PARAMS,
    "attn-cls": attention.PARAMS,
}
NO_FINDING = "No Finding"
NOT_PATHOLOGY = ("No Finding", "Support Devices")
CONSISTENCY = "No Finding = 0 when any label other than Support Devices is predicted"
FOLD_SEED = 20261117
PADCHEST_CONSISTENCY = "normal = 0 when any other label is predicted"


def run(
    dataset: str,
    section: str | None = None,
    encoder: str = "raddino",
    labeler: str = "chexbert14",
    classifier: str = "lr",
    uncertain: str = "zeros",
    tokens: str | None = None,
    seed: int | None = None,
    folds: int | None = None,
    overwrite: bool = False,
    cxr_root: Path | None = None,
) -> Path:
    check_section("classify", dataset, section)
    release = labels_only(dataset)  # PadChest: 0/1 release labels, no section
    if release and uncertain != "zeros":
        raise SystemExit(f"classify: {dataset} labels have no uncertain value")
    attn = classifier in ("attn", "attn-cls")
    if not attn and (tokens is not None or seed is not None or folds is not None):
        raise SystemExit(
            "classify: --tokens, --seed and --folds are for the attn classifiers"
        )
    if folds is not None and folds < 2:
        raise SystemExit("classify: --folds needs at least 2")
    tokens = tokens or "fp16"
    seed = 0 if seed is None else seed
    cxr_root = cxr_root or paths.CXR_ROOT
    adapter = DATASETS[dataset]
    emb_dir = paths.SHARED_ROOT / "embeddings" / dataset / encoder
    if attn:
        emb_dir = emb_dir.with_name(f"{encoder}.tokens") / (section or "")
        stored = quant.stored_format(emb_dir)
        if stored not in (None, "fp16", tokens):
            raise SystemExit(
                f"classify: tokens are stored as {stored}: --tokens {stored}"
            )
    lab_dir = paths.SHARED_ROOT / "labels" / dataset / (section or "") / labeler
    inputs = {
        "embeddings": artifacts.artifact_record(emb_dir),
        "labels": artifacts.artifact_record(lab_dir),
        "sources": [artifacts.file_record(cxr_root / f) for f in adapter.SOURCE_FILES],
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
        "params": PARAMS[classifier],
        "binarize": (
            "release labels: 1 present, 0 absent" if release else BINARIZE[uncertain]
        ),
        "fit": "train",
        "threshold": "max F1 on val; predict prob >= threshold",
        "consistency": PADCHEST_CONSISTENCY if release else CONSISTENCY,
    }
    name = f"{encoder}.{classifier}"
    if attn:
        config |= {"tokens": quant.DESCRIPTION[tokens], "seed": seed}
        if classifier == "attn-cls":
            config["input"] = "CLS token only (token 0 of the tokens)"
        name += f"-{tokens}-seed{seed}"
    if folds:
        config["oof"] = {
            "folds": folds,
            "split": "train-role patients, seeded permutation, position mod folds",
            "fold_seed": FOLD_SEED,
            "fold_models": "the same training on the other folds; epoch, thresholds on val",
        }
        name += f"-oof{folds}"

    def build(out: Path) -> dict:
        labels = artifacts.read_table(lab_dir / "labels.csv")
        names = list(labels.columns.drop("study_id"))
        studies = adapter.load_studies(cxr_root).set_index("study_id")
        studies["role"] = adapter.classifier_roles(studies, paths.SPLITS_DIR)
        studies = studies.loc[labels.study_id]  # KeyError if a label has no study

        # The embedded image must be the one the adapter picks for the study today.
        embedded = artifacts.read_table(emb_dir / "studies.csv").set_index("study_id")
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
            keep = 1 if classifier == "attn-cls" else None
            X = _load_tokens(emb_dir, rows, tokens, device, keep)
            model, history = attention.fit(X, Y, train, val, seed)
            P = attention.predict_proba(model, X, torch.arange(len(X), device=device))
            extra = {}
        elif classifier == "lr-tuned":
            X = np.load(emb_dir / "embeddings.npy")[rows]
            coef, intercept, C, n_iter = logistic.fit_tuned(X, Y, train, val)
            P = logistic.predict_proba(X, coef, intercept)
            extra = {"C": C, "n_iter": n_iter}
        else:
            X = np.load(emb_dir / "embeddings.npy")[rows]
            fits = [logistic.fit(X[train], Y[train, j]) for j in range(len(names))]
            coef = np.stack([f[0] for f in fits])
            intercept = np.array([f[1] for f in fits])
            P = logistic.predict_proba(X, coef, intercept)
            extra = {"n_iter": [f[2] for f in fits]}

        def decide(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Max-F1 thresholds on val, and the consistent 0/1 predictions."""
            threshold = np.array(
                [logistic.f1_threshold(Y[val, j], P[val, j]) for j in range(len(names))]
            )
            pred = (P >= threshold).astype(int)
            if release:
                return threshold, consistent(pred, names, padchest.NO_FINDING, ())
            return threshold, consistent(pred, names)

        threshold, pred = decide(P)
        oof = None
        if folds:
            fold = fold_of(studies.subject_id.to_numpy(), train, folds)
            oof_P, oof_pred, oof = out_of_fold(X, Y, fold, val, seed, decide)
            held = fold >= 0
            ids = labels.study_id[held].to_numpy()
            probs_oof = pd.DataFrame(oof_P[held], columns=names)
            probs_oof.insert(0, "fold", fold[held])
            probs_oof.insert(0, "study_id", ids)
            probs_oof.to_csv(out / "oof_probs.csv", index=False)
            preds_oof = pd.DataFrame(oof_pred[held], columns=names)
            preds_oof.insert(0, "study_id", ids)
            preds_oof.to_csv(out / "oof_predictions.csv", index=False)

        metrics = pd.DataFrame(
            {
                "label": names,
                **extra,
                "train_pos": Y[train].sum(0),
                "threshold": threshold,
                **_scores("val", Y[val], P[val], pred[val]),
                **(
                    _scores("official_valid", Y[official], P[official], pred[official])
                    if official.any()
                    else {}
                ),
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
                **({"C": extra["C"]} if "C" in extra else {}),
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
        if oof is not None:
            summary["oof"] = oof
        return summary

    if release:
        relpath = f"classifiers/{dataset}/{name}/{labeler}"
    else:
        relpath = f"classifiers/{dataset}/{name}/{section}/{labeler}{LABELER_SUFFIX[uncertain]}"
    return artifacts.publish(
        relpath,
        config,
        inputs,
        build,
        overwrite,
    )


def _load_tokens(
    directory: Path,
    rows: np.ndarray,
    fmt: str,
    device: torch.device,
    keep: int | None = None,
) -> quant.Packed:
    """Rows ``rows`` of the tokens stored in ``directory`` (only their first ``keep``
    tokens if given) in format ``fmt`` on ``device``: a float16 store is packed here, a
    packed store (format ``fmt``) is read as is."""
    if quant.stored_format(directory) == "fp16":
        stored = np.load(directory / "tokens.npy", mmap_mode="r")
        _, t, d = stored.shape
        parts = None
    else:
        parts = quant.open_store(directory, fmt)
        t, d = parts["scale"].shape[1], quant.Packed(fmt, parts).dim
    packed = quant.empty(fmt, (len(rows), keep or t, d), device)
    for start in range(0, len(rows), TOKEN_CHUNK):
        chunk = rows[start : start + TOKEN_CHUNK]
        if parts is None:
            x = torch.from_numpy(np.ascontiguousarray(stored[chunk, :keep])).to(device)
            new = quant.pack(fmt, x)
        else:
            new = {
                k: quant.from_store(v[chunk, :keep], packed.parts[k]).to(device)
                for k, v in parts.items()
            }
        for k, v in new.items():
            packed.parts[k][start : start + len(chunk)] = v
    return packed


def out_of_fold(X, Y, fold, val, seed, decide):
    """Each fold's rows predicted by an attention head trained on the other folds'
    rows (fold >= 0 only): probabilities, ``decide``'s 0/1 predictions (thresholds on
    ``val`` from that fold's model), and one summary row per fold."""
    P = np.full(Y.shape, np.nan)
    pred = np.zeros(Y.shape, dtype=int)
    summary = []
    all_rows = torch.arange(len(X), device=next(iter(X.parts.values())).device)
    for k in range(fold.max() + 1):
        model, history = attention.fit(X, Y, (fold >= 0) & (fold != k), val, seed)
        Pk = attention.predict_proba(model, X, all_rows)
        held = fold == k
        P[held], pred[held] = Pk[held], decide(Pk)[1][held]
        best = max(history, key=lambda r: r["val_auroc"])
        summary.append(
            {
                "fold": k,
                "n_held_out": int(held.sum()),
                "best_epoch": best["epoch"],
                "val_mean_auroc": best["val_auroc"],
            }
        )
    return P, pred, summary


def fold_of(subject_id: np.ndarray, train: np.ndarray, folds: int) -> np.ndarray:
    """Each row's fold (0..folds-1) for train rows, by patient; -1 elsewhere."""
    patients = np.unique(subject_id[train])
    order = np.random.default_rng(FOLD_SEED).permutation(len(patients))
    fold_by_patient = dict(zip(patients[order], np.arange(len(patients)) % folds))
    return np.array(
        [fold_by_patient[s] if t else -1 for s, t in zip(subject_id, train)]
    )


def consistent(
    pred: np.ndarray,
    names: list[str],
    no_finding: str = NO_FINDING,
    not_pathology: tuple[str, ...] = NOT_PATHOLOGY,
) -> np.ndarray:
    """Pathologies win over ``no_finding``: clear it where any label other than itself
    and ``not_pathology`` is 1."""
    pred = pred.copy()
    keep = {no_finding, *not_pathology}
    pathology = [j for j, n in enumerate(names) if n not in keep]
    pred[pred[:, pathology].any(axis=1), names.index(no_finding)] = 0
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
