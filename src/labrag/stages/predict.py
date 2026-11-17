"""Stage ``predict``: a trained attention-head classifier applied to another dataset.

Writes ``predictions/<dataset>/<section>/<source>.<name>.<labeler>/`` for
``--classifier <source>/<name>/<labeler>`` (a classify artifact under ``classifiers/``):
probs.csv (study_id, role, one column per label) and predictions.csv (0/1, the
classifier's thresholds and its source's consistency rule) for every study with that
section's tokens. The tokens are read in the format the classifier was trained on.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from labrag import artifacts, paths
from labrag.classifiers import attention, quant
from labrag.data import DATASETS, check_section, labels_only, padchest
from labrag.stages import classify


def run(
    dataset: str,
    classifier: str,
    section: str | None = None,
    overwrite: bool = False,
    cxr_root: Path | None = None,
) -> Path:
    check_section("predict", dataset, section)
    cxr_root = cxr_root or paths.CXR_ROOT
    adapter = DATASETS[dataset]
    clf_dir = paths.SHARED_ROOT / "classifiers" / classifier
    clf_record = artifacts.artifact_record(clf_dir)
    clf_config = artifacts.read_manifest(clf_dir)["config"]
    if clf_config["classifier"] not in ("attn", "attn-cls"):
        raise SystemExit("predict: only the attn classifiers (they read the tokens)")
    source = clf_config["dataset"]
    if source == dataset:
        raise SystemExit("predict: same dataset; classify already predicts it")
    fmt = {v: k for k, v in quant.DESCRIPTION.items()}[clf_config["tokens"]]
    encoder = clf_config["encoder"]
    emb_dir = paths.SHARED_ROOT / "embeddings" / dataset / f"{encoder}.tokens"
    emb_dir = emb_dir / (section or "")
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
        "section": section,
        "classifier": classifier,
        "tokens": clf_config["tokens"],
        "threshold": "the classifier's (max F1 on its own val)",
        "consistency": clf_config["consistency"],
    }
    _, name, labeler = classifier.split("/")

    def build(out: Path) -> dict:
        studies = adapter.load_studies(cxr_root).set_index("study_id")
        studies["role"] = adapter.classifier_roles(studies, paths.SPLITS_DIR)
        embedded = artifacts.read_table(emb_dir / "studies.csv")
        studies = studies.loc[embedded.study_id]
        image = embedded.image_path.to_numpy()
        assert (image == studies.image_path).all(), "image mismatch"
        saved = torch.load(clf_dir / "model.pt", weights_only=True)
        names, threshold = saved["labels"], np.asarray(saved["threshold"])
        params = clf_config["params"]

        attention.deterministic()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        keep = 1 if clf_config["classifier"] == "attn-cls" else None
        rows = np.arange(len(embedded))
        X = classify._load_tokens(emb_dir, rows, fmt, device, keep)
        model = attention.Head(
            len(names), X.dim, params["heads"], params["ffn"], params["dropout"]
        ).to(device)
        model.load_state_dict(saved["state_dict"])
        P = attention.predict_proba(model, X, torch.arange(len(X), device=device))
        pred = (P >= threshold).astype(int)
        if labels_only(source):
            pred = classify.consistent(pred, names, padchest.NO_FINDING, ())
        else:
            pred = classify.consistent(pred, names)

        probs = pd.DataFrame(P, columns=names)
        probs.insert(0, "role", studies.role.to_numpy())
        probs.insert(0, "study_id", embedded.study_id)
        probs.to_csv(out / "probs.csv", index=False)
        preds = pd.DataFrame(pred, columns=names)
        preds.insert(0, "study_id", embedded.study_id)
        preds.to_csv(out / "predictions.csv", index=False)
        return {
            "n_studies": len(preds),
            "labels_per_study": float(pred.sum(1).mean()),
            "empty_label_sets": int((pred.sum(1) == 0).sum()),
            "device": str(device),
        }

    relpath = f"predictions/{dataset}/" + (f"{section}/" if section else "")
    relpath += f"{source}.{name}.{labeler}"
    return artifacts.publish(relpath, config, inputs, build, overwrite)
