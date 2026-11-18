"""Stage ``predict``: a trained attention-head classifier applied to another dataset.

Writes ``predictions/<dataset>/<classifier>/`` for ``--classifier
<source>/<name>/<section>/<labeler>`` (a classify artifact under ``classifiers/``): per
section s of the classifier, probs.<s>.csv (study_id, role, one column per label) and
predictions.<s>.csv (0/1, the classifier's thresholds and its source's consistency
rule) for every study of the dataset's frontal token store, read in the format the
classifier was trained on.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from labrag import artifacts, paths
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
