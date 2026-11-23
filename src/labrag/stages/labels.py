"""Stage ``labels``: reference labels per study.

Writes ``labels/<dataset>/<section>/<labeler>/labels.csv``: one row per study with that
section, columns study_id + one per condition (1 positive, 0 negative, -1 uncertain,
blank = not mentioned). Section ``findings+impression`` (JOINT): the studies with both
sections' labels, combined per label (``combine``: COMBINE_RULE), not CheXbert on the
joined text: that drops stated positives (e.g. Lung Opacity in both sections), and the
classifiers predict section labels, so this keeps predicted and reference labels one
definition. Needs the two sections' labels; no GPU.

Labeler ``devices``: the device types a section names, by rule (labelers/devices.py; 1
named, 0 named as removed or absent, blank = not mentioned), no GPU; next to CheXbert's
one Support Devices label.

A labels-only dataset (PadChest, section ``report``): labels.csv (one row per labelled
study, 1 present / 0 absent, columns most common first) and vocabulary.csv (label,
train count), from the release's own labels.
"""

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from labrag import artifacts, paths, weights
from labrag.data import (
    DATASETS,
    JOINT,
    JOINT_PARTS,
    check_section,
    labels_only,
    padchest,
)
from labrag.labelers import chexbert, devices
from labrag.stages.classify import NO_FINDING, NOT_PATHOLOGY

DEVICES = "devices"
LABELERS = ("chexbert14", DEVICES, *padchest.VOCABULARIES)
# A label's value from several sections: the first of these any gives (NaN last).
COMBINE = (1.0, -1.0, 0.0)
COMBINE_RULE = (
    "per label, the first of positive, uncertain, negative, blank in any section; "
    "No Finding cleared wherever a label other than Support Devices is positive"
)


def run(
    dataset: str,
    section: str,
    labeler: str = "chexbert14",
    overwrite: bool = False,
    cxr_root: Path | None = None,
) -> Path:
    check_section("labels", dataset, section, joint=True)
    cxr_root = cxr_root or paths.CXR_ROOT
    if labels_only(dataset):
        return _release_labels(dataset, section, labeler, overwrite, cxr_root)
    if labeler not in ("chexbert14", DEVICES):
        raise SystemExit(f"labels: {labeler} is for PadChest only")
    if section == JOINT:
        return _joint_labels(dataset, labeler, overwrite)
    adapter = DATASETS[dataset]
    if labeler == DEVICES:
        return _device_labels(dataset, section, overwrite, cxr_root)
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


def _device_labels(dataset: str, section: str, overwrite: bool, cxr_root: Path) -> Path:
    """DEVICES: the rules on each study's section text."""
    adapter = DATASETS[dataset]
    inputs = {
        "sources": [artifacts.file_record(cxr_root / f) for f in adapter.SOURCE_FILES],
    }
    config = {
        "stage": "labels",
        "dataset": dataset,
        "section": section,
        "labeler": DEVICES,
        "labels": devices.NAMES,
    }

    def build(out: Path) -> dict:
        studies = adapter.load_studies(cxr_root)
        studies = studies[studies[section].notna()].reset_index(drop=True)
        labels = devices.label_reports(studies[section])
        labels.insert(0, "study_id", studies.study_id)
        labels.to_csv(out / "labels.csv", index=False)
        named = labels[devices.NAMES].eq(1)
        return {
            "n_studies": len(labels),
            "share_with_a_device": round(float(named.any(axis=1).mean()), 4),
            "named": {n: int(c) for n, c in named.sum().items()},
        }

    return artifacts.publish(
        f"labels/{dataset}/{section}/{DEVICES}", config, inputs, build, overwrite
    )


def _joint_labels(dataset: str, labeler: str, overwrite: bool) -> Path:
    """JOINT: the studies with labels in both JOINT_PARTS, their labels combined."""
    dirs = {
        s: paths.SHARED_ROOT / "labels" / dataset / s / labeler for s in JOINT_PARTS
    }
    inputs = {"labels": {s: artifacts.artifact_record(d) for s, d in dirs.items()}}
    config = {
        "stage": "labels",
        "dataset": dataset,
        "section": JOINT,
        "labeler": labeler,
        "combined": list(JOINT_PARTS),
        "rule": COMBINE_RULE,
    }

    def build(out: Path) -> dict:
        tables = [artifacts.read_table(d / "labels.csv") for d in dirs.values()]
        both = set.intersection(*(set(t.study_id) for t in tables))
        tables = [t[t.study_id.isin(both)].reset_index(drop=True) for t in tables]
        labels = combine(tables, multiclass=True).reset_index()
        labels.to_csv(out / "labels.csv", index=False)
        return {"n_studies": len(labels)}

    return artifacts.publish(
        f"labels/{dataset}/{JOINT}/{labeler}", config, inputs, build, overwrite
    )


def combine(tables: list[pd.DataFrame], multiclass: bool) -> pd.DataFrame:
    """One value per study and label from several sections' (or readouts') tables
    (study_id + label columns, the same in each): the first COMBINE value any gives,
    else blank; 0/1 predictions combine as "either positive". Pathologies win: No
    Finding is blank (multiclass) or 0 wherever a label other than Support Devices is
    positive."""
    tables = [t.set_index("study_id") for t in tables]
    first = tables[0]
    for t in tables[1:]:
        assert t.index.equals(first.index) and list(t.columns) == list(first.columns)
    values = np.stack([t.to_numpy(dtype=float) for t in tables])
    assert np.isin(values[~np.isnan(values)], COMBINE).all(), "unexpected label values"
    out = np.full(first.shape, np.nan)
    for v in COMBINE[::-1]:  # the earliest value written last wins
        out[(values == v).any(0)] = v
    names = list(first.columns)
    if NO_FINDING in names:
        pathology = [j for j, n in enumerate(names) if n not in NOT_PATHOLOGY]
        found = (out[:, pathology] == 1).any(1)
        out[found, names.index(NO_FINDING)] = np.nan if multiclass else 0
    return pd.DataFrame(out, index=first.index, columns=names)


def _release_labels(
    dataset: str, section: str, labeler: str, overwrite: bool, cxr_root: Path
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
        "section": section,
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
        f"labels/{dataset}/{section}/{labeler}", config, inputs, build, overwrite
    )
