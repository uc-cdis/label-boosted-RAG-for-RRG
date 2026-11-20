"""Stage ``predict``: a trained attention-head classifier applied to another dataset.

A ``finetune`` classifier (RAD-DINO fine-tuned with the head) is applied to any
dataset, its training datasets included, from the images: per section, max-F1
thresholds on the target's own val studies when the target has that section's labels
(metrics.csv: their val scores), else the classifier's pooled-val thresholds.

Writes ``predictions/<dataset>/<classifier>/`` for ``--classifier
<source>/<name>/<section>/<labeler>`` (a classify artifact under ``classifiers/``): per
section s of the classifier, probs.<s>.csv (study_id, role, one column per label) and
predictions.<s>.csv (0/1, the classifier's thresholds and its source's consistency
rule) for every study of the dataset's frontal token store, read in the format the
classifier was trained on.

A multiclass fine-tune (``chexbert14.mc``): predictions.<s>.csv holds the most
probable CheXbert value per label (1, 0, -1, blank; labels.csv's coding), and
probs.<s>.<class>.csv each class's probability; no thresholds.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from labrag import artifacts, paths, weights
from labrag.classifiers import attention, quant
from labrag.data import DATASETS
from labrag.stages import classify


def run(
    dataset: str,
    classifier: str,
    overwrite: bool = False,
    cxr_root: Path | None = None,
) -> Path:
    classifier = classifier.strip("/").removeprefix("classifiers/")
    cxr_root = cxr_root or paths.CXR_ROOT
    adapter = DATASETS[dataset]
    clf_dir = paths.SHARED_ROOT / "classifiers" / classifier
    clf_record = artifacts.artifact_record(clf_dir)
    clf_config = artifacts.read_manifest(clf_dir)["config"]
    if clf_config["stage"] == "finetune":
        return _finetuned(
            dataset, classifier, clf_dir, clf_record, clf_config, overwrite, cxr_root
        )
    if clf_config["stage"] != "classify":
        raise SystemExit("predict: --classifier must be a classify artifact (a model)")
    if clf_config["classifier"] not in ("attn", "attn-cls"):
        raise SystemExit("predict: only the attn classifiers (they read the tokens)")
    source = clf_config["dataset"]
    if source == dataset:
        raise SystemExit("predict: same dataset; classify already predicts it")
    fmt = {v: k for k, v in quant.DESCRIPTION.items()}[clf_config["tokens"]]
    encoder = clf_config["encoder"]
    emb_dir = paths.SHARED_ROOT / "embeddings" / dataset / f"{encoder}.tokens/frontal"
    inputs = {
        "classifier": clf_record,
        "embeddings": artifacts.artifact_record(emb_dir),
        "sources": [artifacts.file_record(cxr_root / f) for f in adapter.SOURCE_FILES],
        "splits": [
            artifacts.file_record(paths.SPLITS_DIR / f) for f in adapter.SPLIT_FILES
        ],
    }
    config = {
        "stage": "predict",
        "dataset": dataset,
        "classifier": classifier,
        "tokens": clf_config["tokens"],
        "threshold": "the classifier's (max F1 on its own val)",
        "consistency": clf_config["consistency"],
    }
    sections = clf_config["sections"]
    release = clf_config["binarize"] == classify.RELEASE_BINARIZE

    def build(out: Path) -> dict:
        studies = adapter.load_studies(cxr_root).set_index("study_id")
        studies["role"] = adapter.classifier_roles(studies, paths.SPLITS_DIR)
        embedded = artifacts.read_table(emb_dir / "studies.csv")
        studies = studies.loc[embedded.study_id]
        image = embedded.image_path.to_numpy()
        assert (image == studies.image_path).all(), "image mismatch"
        # Our own artifact (inputs are sha256-checked); it holds a numpy threshold.
        saved = torch.load(clf_dir / "model.pt", weights_only=False)
        names, threshold = saved["labels"], np.asarray(saved["threshold"])
        assert saved["sections"] == sections, "model.pt sections differ"
        params = clf_config["params"]

        attention.deterministic()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        keep = 1 if clf_config["classifier"] == "attn-cls" else None
        rows = np.arange(len(embedded))
        X = classify._load_tokens(emb_dir, rows, fmt, device, keep)
        model = attention.Head(
            len(names),
            X.dim,
            params["heads"],
            params["ffn"],
            params["dropout"],
            len(sections),
        ).to(device)
        model.load_state_dict(saved["state_dict"])
        P = attention.predict_proba(model, X, torch.arange(len(X), device=device))
        n, summary = len(names), {}
        for i, s in enumerate(sections):
            Ps = P[:, i * n : (i + 1) * n]
            pred = classify.predict_labels(Ps, threshold[i], names, release)
            probs = pd.DataFrame(Ps, columns=names)
            probs.insert(0, "role", studies.role.to_numpy())
            probs.insert(0, "study_id", embedded.study_id)
            probs.to_csv(out / f"probs.{s}.csv", index=False)
            preds = pd.DataFrame(pred, columns=names)
            preds.insert(0, "study_id", embedded.study_id)
            preds.to_csv(out / f"predictions.{s}.csv", index=False)
            summary[s] = {
                "labels_per_study": float(pred.sum(1).mean()),
                "empty_label_sets": int((pred.sum(1) == 0).sum()),
            }
        return {"n_studies": len(embedded), **summary, "device": str(device)}

    relpath = f"predictions/{dataset}/{classifier}"
    return artifacts.publish(relpath, config, inputs, build, overwrite)


def _finetuned(
    dataset, classifier, clf_dir, clf_record, clf_config, overwrite, cxr_root
):
    """predictions/<dataset>/<classifier>/ of a finetune classifier, from the images."""
    from labrag.classifiers import finetune, logistic
    from labrag.stages import finetune as finetune_stage

    adapter = DATASETS[dataset]
    sections, labeler = clf_config["sections"], clf_config["labeler"]
    model_dir, weights_record = weights.resolve(finetune_stage.embed_weights())
    multiclass = "classes" in clf_config
    positive = (
        "zeros"
        if multiclass
        else {v: k for k, v in classify.BINARIZE.items()}[clf_config["binarize"]]
    )
    lab_dirs = {
        s: paths.SHARED_ROOT / "labels" / dataset / s / labeler
        for s in sections
        if s in adapter.SECTIONS
        and (paths.SHARED_ROOT / "labels" / dataset / s / labeler).is_dir()
    }
    inputs = {
        "classifier": clf_record,
        "labels": {s: artifacts.artifact_record(d) for s, d in lab_dirs.items()},
        "sources": [artifacts.file_record(cxr_root / f) for f in adapter.SOURCE_FILES],
        "splits": [
            artifacts.file_record(paths.SPLITS_DIR / f) for f in adapter.SPLIT_FILES
        ],
        "images": {"root": str(cxr_root / adapter.IMAGE_ROOT)},
        "weights": weights_record,
    }
    config = {
        "stage": "predict",
        "dataset": dataset,
        "classifier": classifier,
        "image": clf_config["image"],
        "threshold": {
            s: (
                "max F1 on this dataset's val"
                if s in lab_dirs
                else "the classifier's (max F1 on its pooled val)"
            )
            for s in sections
        },
        "consistency": clf_config["consistency"],
    }
    if multiclass:
        config["threshold"] = clf_config["threshold"]
        config["classes"] = clf_config["classes"]

    def build(out: Path) -> dict:
        studies = adapter.load_studies(cxr_root)
        role = adapter.classifier_roles(studies, paths.SPLITS_DIR).to_numpy()
        ids = studies.study_id.reset_index(drop=True)
        saved = torch.load(clf_dir / "model.pt", weights_only=False)
        names, stored = saved["labels"], np.asarray(saved["threshold"])
        assert saved["sections"] == sections, "model.pt sections differ"
        attention.deterministic()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone = finetune.load_backbone(model_dir)
        classes = len(saved["classes"]) if multiclass else 1
        model = finetune.build(
            backbone,
            len(names),
            len(sections),
            clf_config["classifier"] == "attn-cls",
            classes,
        )
        model.load_state_dict(saved["state_dict"])
        model.to(device)
        root = cxr_root / adapter.IMAGE_ROOT
        images = finetune.Images(
            [root / p for p in studies.image_path], finetune.load_processor(model_dir)
        )
        workers = min(
            finetune_stage.MAX_WORKERS, max(0, len(os.sched_getaffinity(0)) - 1)
        )
        P_full = finetune.predict_proba(model, images, device, workers)
        P = finetune.positive_probs(P_full, len(sections), classes)
        n, summary, metrics = len(names), {}, []
        val = role == "val"
        for i, s in enumerate(sections):
            Ps = P[:, i * n : (i + 1) * n]
            codes = None
            if multiclass:
                codes = classify.argmax_codes(P_full, i, len(sections), names)
            else:
                t = stored[i]
            if s in lab_dirs:
                labels = artifacts.read_table(lab_dirs[s] / "labels.csv").set_index(
                    "study_id"
                )
                has = ids.isin(labels.index).to_numpy()
                Y = labels.reindex(ids)[names].isin(classify.POSITIVE[positive])
                Y = Y.to_numpy(dtype=int)
                v = val & has
                if multiclass:
                    pred_v = classify.codes_positive(codes[v])
                    extra = {}
                else:
                    t = np.array(
                        [logistic.f1_threshold(y, p) for y, p in zip(Y[v].T, Ps[v].T)]
                    )
                    pred_v = classify.predict_labels(Ps[v], t, names, False)
                    extra = {"threshold": t}
                metrics.append(
                    pd.DataFrame(
                        {
                            "section": s,
                            "label": names,
                            **extra,
                            **classify._scores("val", Y[v], Ps[v], pred_v),
                        }
                    )
                )
            if multiclass:
                pred = classify.codes_positive(codes)
                block = P_full.reshape(len(P_full), len(sections), classes, n)[:, i]
                for c, name in enumerate(saved["classes"]):
                    probs = pd.DataFrame(block[:, c], columns=names)
                    probs.insert(0, "role", role)
                    probs.insert(0, "study_id", ids)
                    probs.to_csv(out / f"probs.{s}.{name}.csv", index=False)
                preds = pd.DataFrame(codes, columns=names)
            else:
                pred = classify.predict_labels(Ps, t, names, False)
                probs = pd.DataFrame(Ps, columns=names)
                probs.insert(0, "role", role)
                probs.insert(0, "study_id", ids)
                probs.to_csv(out / f"probs.{s}.csv", index=False)
                preds = pd.DataFrame(pred, columns=names)
            preds.insert(0, "study_id", ids)
            preds.to_csv(out / f"predictions.{s}.csv", index=False)
            summary[s] = {
                "labels_per_study": float(pred.sum(1).mean()),
                "empty_label_sets": int((pred.sum(1) == 0).sum()),
            }
            if multiclass:
                summary[s] |= {
                    f"{name}_per_study": float((codes == code).sum(1).mean())
                    for name, code in (("negative", 0), ("uncertain", -1))
                }
        if metrics:
            pd.concat(metrics, ignore_index=True).to_csv(
                out / "metrics.csv", index=False
            )
        return {"n_studies": len(ids), **summary, "device": str(device)}

    relpath = f"predictions/{dataset}/{classifier}"
    return artifacts.publish(relpath, config, inputs, build, overwrite)
