"""Stage ``classify``: per-label image classifiers for one or more sections' labels.

Writes ``classifiers/<dataset>/<encoder>.<classifier>/<section>/<labeler>[.u1]/``:
model.npz (coef, intercept, threshold per label; C for lr-tuned), metrics.csv (per
section and label, on val and official valid), and per section s of the classifier
probs.<s>.csv (study_id, role, one column per label) and predictions.<s>.csv (0/1, the
layout of labels.csv).

Its studies: those with labels in any of its sections (each must be embedded). Fit on
the adapter's ``train`` role; max-F1 thresholds per section on its ``val`` studies;
other roles predicted only. Classifiers:

- ``lr``: per-label logistic regression on the CLS embedding (the preprint's; CPU).
- ``lr-tuned``: ``lr`` on standardized features with C picked per label on val (CPU).
- ``attn``: per-label attention pooling over the tokens of ``embed --feature tokens``
  (CLS + patches), trained jointly (1 GPU).
- ``attn-cls``: ``attn`` given only the CLS token, the control for the patch tokens.

The ``attn`` ones are written to ``<encoder>.<classifier>-<tokens>-seed<seed>`` (tokens =
the storage format they are read in) with model.pt instead of model.npz. A float16
token store can be read in any format; a packed store (embed's) only in its own.

``--section`` with sections joined by ``-`` (``findings-impression``; attn only): one
head learns every section's labels, one query per label shared by the sections, one
readout per section, and each study's loss only over the sections it has labels for.

Uncertain labels (Irvin et al. 2019): ``zeros`` (U-Zeros, the default) counts only
positive as 1; ``ones`` (U-Ones, the preprint, labeler suffix ``.u1``) also uncertain.

``--folds K`` (attn, one section; name suffix ``-oofK``) also writes out-of-fold
predictions for the train role (oof_probs.<s>.csv, oof_predictions.<s>.csv): train
patients split into K folds, each predicted by the same training on the other folds
(epoch and thresholds on val), for wherever predicted labels stand in for reference
ones on studies the classifier trains on.

Pathologies win (``consistent``): No Finding is cleared wherever a label other than
Support Devices is predicted, as CheXbert's reference labels do (all but 40 of 46,759).
PadChest's ``normal`` is cleared wherever any other label is predicted (its labels pair
normal with another label in 155 of 108,746 studies, devices included).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from labrag import artifacts, paths
from labrag.classifiers import attention, logistic, quant
from labrag.data import DATASETS, labels_only, padchest, split_sections

CLASSIFIERS = ("lr", "lr-tuned", "attn", "attn-cls")
TOKEN_CHUNK = 256  # studies packed at a time while loading tokens
# CheXbert codes counted as positive, per uncertain-label policy.
POSITIVE = {"ones": (1, -1), "zeros": (1,)}
BINARIZE = {
    "ones": "positive or uncertain = 1, negative or blank = 0",
    "zeros": "positive = 1, uncertain or negative or blank = 0",
}
RELEASE_BINARIZE = "release labels: 1 present, 0 absent"
LABELER_SUFFIX = {"zeros": "", "ones": ".u1"}
# Multiclass classifiers (finetune --multiclass) predict CheXbert's own values.
MULTICLASS_SUFFIX = ".mc"
MULTICLASS_CLASSES = "positive 1, negative 0, uncertain -1, blank (unmentioned)"
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
    section: str,
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
    sections = split_sections("classify", dataset, section)
    release = labels_only(dataset)  # PadChest: 0/1 release labels
    if release and uncertain != "zeros":
        raise SystemExit(f"classify: {dataset} labels have no uncertain value")
    attn = classifier in ("attn", "attn-cls")
    if not attn and (tokens is not None or seed is not None or folds is not None):
        raise SystemExit(
            "classify: --tokens, --seed and --folds are for the attn classifiers"
        )
    if len(sections) > 1 and not attn:
        raise SystemExit("classify: several sections only for the attn classifiers")
    if folds is not None and len(sections) > 1:
        raise SystemExit("classify: --folds needs a single section")
    if folds is not None and folds < 2:
        raise SystemExit("classify: --folds needs at least 2")
    tokens = tokens or "fp16"
    seed = 0 if seed is None else seed
    cxr_root = cxr_root or paths.CXR_ROOT
    adapter = DATASETS[dataset]
    emb_dir = paths.SHARED_ROOT / "embeddings" / dataset / encoder
    if attn:
        emb_dir = emb_dir.with_name(f"{encoder}.tokens") / "frontal"
        stored = quant.stored_format(emb_dir)
        if stored not in (None, "fp16", tokens):
            raise SystemExit(
                f"classify: tokens are stored as {stored}: --tokens {stored}"
            )
    lab_dirs = {
        s: paths.SHARED_ROOT / "labels" / dataset / s / labeler for s in sections
    }
    inputs = {
        "embeddings": artifacts.artifact_record(emb_dir),
        "labels": {s: artifacts.artifact_record(d) for s, d in lab_dirs.items()},
        "sources": [artifacts.file_record(cxr_root / f) for f in adapter.SOURCE_FILES],
        "splits": [
            artifacts.file_record(paths.SPLITS_DIR / f) for f in adapter.SPLIT_FILES
        ],
    }
    config = {
        "stage": "classify",
        "dataset": dataset,
        "sections": sections,
        "encoder": encoder,
        "labeler": labeler,
        "classifier": classifier,
        "params": PARAMS[classifier],
        "binarize": RELEASE_BINARIZE if release else BINARIZE[uncertain],
        "fit": "train; each section's loss on the studies with its labels",
        "threshold": "max F1 per section on its val studies; predict prob >= threshold",
        "consistency": PADCHEST_CONSISTENCY if release else CONSISTENCY,
    }
    if len(sections) > 1:
        config["readouts"] = (
            "one query per label shared by the sections, a readout per section"
        )
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
        labels = {
            s: artifacts.read_table(d / "labels.csv").set_index("study_id")
            for s, d in lab_dirs.items()
        }
        names = list(labels[sections[0]].columns)
        assert all(list(t.columns) == names for t in labels.values()), "labels differ"
        n = len(names)
        blocks = {s: slice(i * n, (i + 1) * n) for i, s in enumerate(sections)}
        embedded = artifacts.read_table(emb_dir / "studies.csv")
        labelled = set().union(*(t.index for t in labels.values()))
        any_label = embedded.study_id.isin(labelled).to_numpy()
        assert any_label.sum() == len(labelled), "labelled studies not embedded"
        rows = np.flatnonzero(any_label)
        ids = embedded.study_id[any_label].reset_index(drop=True)
        studies = adapter.load_studies(cxr_root).set_index("study_id")
        studies["role"] = adapter.classifier_roles(studies, paths.SPLITS_DIR)
        studies = studies.loc[ids]
        # The embedded image must be the one the adapter picks for the study today.
        embedded_image = embedded.image_path.to_numpy()[rows]
        assert (embedded_image == studies.image_path.to_numpy()).all(), "image mismatch"
        role = studies.role.to_numpy()
        train, val = role == "train", role == "val"
        official = role == "official_valid"
        has = {s: ids.isin(t.index).to_numpy() for s, t in labels.items()}
        Y = np.zeros((len(ids), n * len(sections)), dtype=int)
        for s, t in labels.items():
            y = t.loc[ids[has[s]], names].isin(POSITIVE[uncertain])
            Y[has[s], blocks[s]] = y.to_numpy(dtype=int)
        mask = None
        if len(sections) > 1:
            mask = np.concatenate(
                [np.repeat(has[s][:, None], n, 1) for s in sections], 1
            )

        if attn:
            attention.deterministic()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            keep = 1 if classifier == "attn-cls" else None
            X = _load_tokens(emb_dir, rows, tokens, device, keep)
            model, history = attention.fit(
                X, Y, train, val, seed, mask=mask, readouts=len(sections)
            )
            P = attention.predict_proba(model, X, torch.arange(len(X), device=device))
            extra = {}
        elif classifier == "lr-tuned":
            X = np.load(emb_dir / "embeddings.npy")[rows]
            coef, intercept, C, n_iter = logistic.fit_tuned(X, Y, train, val)
            P = logistic.predict_proba(X, coef, intercept)
            extra = {"C": C, "n_iter": n_iter}
        else:
            X = np.load(emb_dir / "embeddings.npy")[rows]
            fits = [logistic.fit(X[train], Y[train, j]) for j in range(n)]
            coef = np.stack([f[0] for f in fits])
            intercept = np.array([f[1] for f in fits])
            P = logistic.predict_proba(X, coef, intercept)
            extra = {"n_iter": [f[2] for f in fits]}

        def decide(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Max-F1 thresholds per section on its val studies (sections x labels),
            and the consistent 0/1 predictions."""
            thresholds, pred = [], np.zeros(P.shape, dtype=int)
            for s, b in blocks.items():
                Yv, Pv = Y[val & has[s], b], P[val & has[s], b]
                t = np.array([logistic.f1_threshold(y, p) for y, p in zip(Yv.T, Pv.T)])
                pred[:, b] = predict_labels(P[:, b], t, names, release)
                thresholds.append(t)
            return np.stack(thresholds), pred

        threshold, pred = decide(P)
        oof = None
        if folds:
            fold = fold_of(studies.subject_id.to_numpy(), train, folds)
            oof_P, oof_pred, oof = out_of_fold(X, Y, fold, val, seed, decide)
            held = fold >= 0
            (s,) = sections
            probs_oof = pd.DataFrame(oof_P[held], columns=names)
            probs_oof.insert(0, "fold", fold[held])
            probs_oof.insert(0, "study_id", ids[held].to_numpy())
            probs_oof.to_csv(out / f"oof_probs.{s}.csv", index=False)
            preds_oof = pd.DataFrame(oof_pred[held], columns=names)
            preds_oof.insert(0, "study_id", ids[held].to_numpy())
            preds_oof.to_csv(out / f"oof_predictions.{s}.csv", index=False)

        metrics, by_section = [], {}
        for i, (s, b) in enumerate(blocks.items()):
            v, o = val & has[s], official & has[s]
            Ys, Ps, preds_s = Y[:, b], P[:, b], pred[:, b]
            by_section[s] = attention.mean_auroc(Ys[v], Ps[v])
            metrics.append(
                pd.DataFrame(
                    {
                        "section": s,
                        "label": names,
                        **extra,
                        "train_pos": Ys[train & has[s]].sum(0),
                        "threshold": threshold[i],
                        **_scores("val", Ys[v], Ps[v], preds_s[v]),
                        **(
                            _scores("official_valid", Ys[o], Ps[o], preds_s[o])
                            if o.any()
                            else {}
                        ),
                    }
                )
            )
            probs = pd.DataFrame(Ps, columns=names)
            probs.insert(0, "role", role)
            probs.insert(0, "study_id", ids)
            probs.to_csv(out / f"probs.{s}.csv", index=False)
            preds = pd.DataFrame(preds_s, columns=names)
            preds.insert(0, "study_id", ids)
            preds.to_csv(out / f"predictions.{s}.csv", index=False)
        metrics = pd.concat(metrics, ignore_index=True)
        metrics.to_csv(out / "metrics.csv", index=False)
        print(metrics.to_string(index=False))

        if attn:
            state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "labels": names,
                    "sections": sections,
                    "state_dict": state,
                    "threshold": threshold,
                },
                out / "model.pt",
            )
            pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        else:
            np.savez(
                out / "model.npz",
                labels=np.array(names),
                sections=np.array(sections),
                coef=coef,
                intercept=intercept,
                threshold=threshold,
                **({"C": extra["C"]} if "C" in extra else {}),
            )
        summary = {
            "n_studies": len(ids),
            "n_by_role": pd.Series(role).value_counts().sort_index().to_dict(),
            "n_labelled": {s: int(h.sum()) for s, h in has.items()},
            "val_mean_auroc_by_section": by_section,
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

    relpath = (
        f"classifiers/{dataset}/{name}/{section}/{labeler}{LABELER_SUFFIX[uncertain]}"
    )
    return artifacts.publish(relpath, config, inputs, build, overwrite)


def predict_labels(
    P: np.ndarray, threshold: np.ndarray, names: list[str], release: bool
) -> np.ndarray:
    """0/1 predictions at ``threshold``, made consistent by the dataset's rule."""
    pred = (P >= threshold).astype(int)
    if release:
        return consistent(pred, names, padchest.NO_FINDING, ())
    return consistent(pred, names)


def argmax_codes(
    P: np.ndarray, section: int, sections: int, names: list[str]
) -> np.ndarray:
    """A multiclass model's most probable CheXbert value (1, 0, -1, NaN for blank)
    per label for section number ``section`` of its outputs P (N, sections * classes
    * labels); No Finding is blank wherever a pathology is positive (CONSISTENCY)."""
    from labrag.classifiers.finetune import MULTICLASS_CODES

    classes = len(MULTICLASS_CODES)
    block = P.reshape(len(P), sections, classes, len(names))[:, section]
    codes = np.array([np.nan if c is None else c for c in MULTICLASS_CODES])
    out = codes[block.argmax(1)]
    pathology = [j for j, n in enumerate(names) if n not in NOT_PATHOLOGY]
    out[(out[:, pathology] == 1).any(axis=1), names.index(NO_FINDING)] = np.nan
    return out


def codes_positive(codes: np.ndarray) -> np.ndarray:
    """0/1: CheXbert value positive."""
    return (codes == 1).astype(int)


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
