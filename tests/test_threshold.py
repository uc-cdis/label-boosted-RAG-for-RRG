"""Threshold: rules, output names, and its predictions on a small classify artifact."""

import numpy as np
import pandas as pd
import pytest

from labrag import artifacts, paths
from labrag.stages import classify, threshold

NAMES = ["Edema", "Support Devices", "No Finding"]


def test_rules():
    Y = np.array([[1, 0], [1, 0], [0, 0], [0, 1]])
    P = np.array([[0.9, 0.2], [0.4, 0.6], [0.7, 0.1], [0.1, 0.3]])
    assert threshold.cutoffs("0.5", Y, P).tolist() == [0.5, 0.5]
    # prevalence: 2 positives -> 2nd highest; 1 positive -> highest; none -> inf
    assert threshold.cutoffs("prevalence", Y, P).tolist() == [0.7, 0.6]
    assert threshold.cutoffs("prevalence", np.zeros((2, 1)), P[:2, :1])[0] == np.inf
    assert threshold.cutoffs("max-f1", Y, P).tolist() == [0.4, 0.3]


def _source(root, name, section):
    """A classify artifact (probs + manifest) over 6 studies, 4 of them val; every
    section of a joined ``section`` gets the same labels and probabilities."""
    labels = pd.DataFrame(
        {"study_id": [f"s{i}" for i in range(6)], "Edema": [1, 1, 0, 0, 1, 0],
         "Support Devices": [1, 0, 1, 0, 0, 0], "No Finding": [0, 0, 0, 1, 0, 1]}
    )  # fmt: skip
    probs = labels[["study_id"]].assign(
        role=["train", "train", "val", "val", "val", "val"],
        **{"Edema": [0.9, 0.8, 0.7, 0.2, 0.6, 0.1],
           "Support Devices": [0.9, 0.1, 0.3, 0.2, 0.1, 0.4],
           "No Finding": [0.1, 0.1, 0.1, 0.9, 0.2, 0.8]},
    )  # fmt: skip
    sections = section.split("-")
    records = {}
    for s in sections:
        lab_rel = f"labels/d/{s}/chexbert14"
        artifacts.publish(
            lab_rel, {"stage": "labels"}, {},
            lambda out: labels.to_csv(out / "labels.csv", index=False),
            shared_root=root,
        )  # fmt: skip
        records[s] = artifacts.artifact_record(root / lab_rel)
    config = {
        "stage": "classify",
        "sections": sections,
        "binarize": classify.BINARIZE["zeros"],
        "consistency": classify.CONSISTENCY,
    }

    def build(out):
        for s in sections:
            probs.to_csv(out / f"probs.{s}.csv", index=False)

    rel = f"classifiers/d/{name}/{section}/chexbert14"
    artifacts.publish(rel, config, {"labels": records}, build, shared_root=root)
    return rel.removeprefix("classifiers/")


@pytest.mark.parametrize("section", ["findings", "findings-impression"])
def test_prevalence_predictions(tmp_path, monkeypatch, section):
    monkeypatch.setattr(paths, "SHARED_ROOT", tmp_path)
    monkeypatch.setenv("LABRAG_COMMIT", "a" * 40)
    src = _source(tmp_path, "raddino.attn-nvfp4-seed1", section)
    dest = threshold.run(src, "prevalence")
    out = f"classifiers/d/raddino.attn-nvfp4-seed1-prev/{section}/chexbert14"
    assert dest == tmp_path / out
    for s in section.split("-"):
        pred = artifacts.read_table(dest / f"predictions.{s}.csv")
        pred = pred.set_index("study_id")
        # val s2..s5 positives: Edema 1 -> t 0.7, Support Devices 1 -> t 0.4, No
        # Finding 2 -> t 0.8; thresholds apply to every row, train included
        assert pred.loc["s2"].tolist() == [1, 0, 0]
        assert pred.loc["s3"].tolist() == [0, 0, 1]
        assert pred.loc["s5"].tolist() == [0, 1, 1]  # Support Devices keeps No Finding
        assert pred.loc["s0"].tolist() == [1, 1, 0]
    assert threshold.trained_manifest(dest)["config"]["stage"] == "classify"


def test_refuses_non_classify(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "SHARED_ROOT", tmp_path)
    with pytest.raises(SystemExit, match="not a classify artifact"):
        threshold.run("d/missing/findings/chexbert14", "0.5")
