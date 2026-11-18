"""Stage ``labels``: reference labels per study.

Writes ``labels/<dataset>/<section>/<labeler>/labels.csv``: one row per study with that
section, columns study_id + one per condition (1 positive, 0 negative, -1 uncertain,
blank = not mentioned).

A labels-only dataset (PadChest) has no section: ``labels/<dataset>/<labeler>/`` holds
labels.csv (one row per labelled study, 1 present / 0 absent, columns most common
first) and vocabulary.csv (label, train count), from the release's own labels.
"""

import subprocess
from pathlib import Path

import pandas as pd
import torch

from labrag import artifacts, paths, weights
from labrag.data import DATASETS, check_section, labels_only, padchest
from labrag.labelers import chexbert

SECTIONS = ("findings", "impression")
LABELERS = ("chexbert14", *padchest.VOCABULARIES)


def run(
    dataset: str,
    section: str | None = None,
    labeler: str = "chexbert14",
    overwrite: bool = False,
    cxr_root: Path | None = None,
) -> Path:
    check_section("labels", dataset, section)
    cxr_root = cxr_root or paths.CXR_ROOT
    if labels_only(dataset):
        return _release_labels(dataset, labeler, overwrite, cxr_root)
    if labeler != "chexbert14":
        raise SystemExit(f"labels: {labeler} is for PadChest only")
    adapter = DATASETS[dataset]
    chexbert_dir, chexbert_record = weights.resolve("chexbert")
    bert_dir, bert_record = weights.resolve("bert-base-uncased")
    code = subprocess.run(
        ["git", "-C", str(chexbert.UPSTREAM_SRC.parent), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    inputs = {
        "sources": [artifacts.file_record(cxr_root / f) for f in adapter.SOURCE_FILES],
        "chexbert_weights": chexbert_record,
        "bert_weights": bert_record,
        "chexbert_code": {"path": "external/CheXbert", "commit": code},
    }
    config = {
        "stage": "labels",
        "dataset": dataset,
        "section": section,
        "labeler": labeler,
    }

    def build(out: Path) -> dict:
        studies = adapter.load_studies(cxr_root)
        studies = studies[studies[section].notna()].reset_index(drop=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, tokenizer = chexbert.load_model(
            chexbert_dir / "chexbert.pth", bert_dir, device
        )
        classes = chexbert.predict(studies[section].tolist(), model, tokenizer, device)
        labels = chexbert.to_chexpert(classes)
        labels.insert(0, "study_id", studies.study_id)
        labels.to_csv(out / "labels.csv", index=False)
        return {"n_studies": len(labels), "device": str(device)}

    return artifacts.publish(
        f"labels/{dataset}/{section}/{labeler}", config, inputs, build, overwrite
    )


def _release_labels(
    dataset: str, labeler: str, overwrite: bool, cxr_root: Path
) -> Path:
    """PadChest: the release's labels, restricted to a vocabulary drawn on train."""
    if labeler not in padchest.VOCABULARIES:
        raise SystemExit(
            f"labels: {dataset} takes --labeler {list(padchest.VOCABULARIES)}"
        )
    adapter = DATASETS[dataset]
    n = padchest.VOCABULARIES[labeler]
    inputs = {
        "sources": [artifacts.file_record(cxr_root / f) for f in adapter.SOURCE_FILES],
        "splits": [
            artifacts.file_record(paths.SPLITS_DIR / f) for f in adapter.SPLIT_FILES
        ],
    }
    config = {
        "stage": "labels",
        "dataset": dataset,
        "labeler": labeler,
        "labels": f"release Labels column; the {n} most common on train",
        "left_out": list(padchest.NOT_FINDINGS),
    }

    def build(out: Path) -> dict:
        studies = adapter.load_studies(cxr_root)
        roles = adapter.classifier_roles(studies, paths.SPLITS_DIR)
        vocab = padchest.vocabulary(studies, roles, n)
        names = [label for label, _ in vocab]
        labelled = studies[studies.labels.notna()].reset_index(drop=True)
        labels = pd.DataFrame(
            [[int(x in set(ls)) for x in names] for ls in labelled.labels],
            columns=names,
        )
        labels.insert(0, "study_id", labelled.study_id)
        labels.to_csv(out / "labels.csv", index=False)
        pd.DataFrame(vocab, columns=["label", "train_count"]).to_csv(
            out / "vocabulary.csv", index=False
        )
        return {"n_studies": len(labels), "n_labels": len(names)}

    return artifacts.publish(
        f"labels/{dataset}/{labeler}", config, inputs, build, overwrite
    )
