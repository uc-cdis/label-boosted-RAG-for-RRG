"""Report-style clusters: the style text, and the stage on a fake dataset."""

import types

import numpy as np
import pandas as pd
import pytest

from labrag import artifacts, data, paths
from labrag.stages import build_prompts, styles


def test_style_text_drops_findings_and_masks_terms():
    text = "Small left effusion. No pneumothorax. Heart size is normal. See 2 views."
    assert styles.style_text(text) == (
        "DROPSENT no FIND. FIND is normal. see NUM views."
    )


def test_style_is_a_same_choice_that_needs_examples():
    assert build_prompts.STYLE in build_prompts.SAME
    with pytest.raises(SystemExit, match="--same style is for conditions with"):
        build_prompts.run("val", "mimic", "findings", "no-context", same="style")


def test_styles_fit_on_train_and_assign_every_study(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "SHARED_ROOT", tmp_path / "shared")
    monkeypatch.setitem(styles.PARAMS, "tfidf", styles.PARAMS["tfidf"] | {"min_df": 1})
    monkeypatch.setitem(styles.PARAMS, "svd_dims", 3)
    monkeypatch.setitem(styles.PARAMS, "k", 2)
    templates = ["No acute process. Lungs are clear.", "Unremarkable. See prior."]
    n = 40
    studies = pd.DataFrame(
        {
            "study_id": [f"s{i}" for i in range(n)],
            "findings": [templates[i % 2] + f" Mild edema {i}." for i in range(n)],
            "split": ["train" if i < 30 else "val" for i in range(n)],
        }
    )
    adapter = types.SimpleNamespace(
        SOURCE_FILES=(),
        SPLIT_FILES=(),
        SECTIONS=("findings",),
        load_studies=lambda root: studies.copy(),
        classifier_roles=lambda s, splits_dir: s.split,
    )
    monkeypatch.setitem(data.DATASETS, "fake", adapter)
    labels = pd.DataFrame(np.zeros((n, 3)), columns=["A", "B", "C"])
    labels.insert(0, "study_id", studies.study_id)
    artifacts.publish(
        "labels/fake/findings/chexbert14",
        {"stage": "labels"},
        {},
        lambda out: labels.to_csv(out / "labels.csv", index=False),
    )
    out = styles.run("fake", "findings", cxr_root=tmp_path)
    assert out == paths.SHARED_ROOT / styles.relpath("fake", "findings")
    table = artifacts.read_table(out / "styles.csv")
    assert list(table.study_id) == list(studies.study_id)
    # same labels everywhere: the two report templates are the two styles
    template = np.arange(n) % 2
    style = table["style"].to_numpy()
    assert (style == template).all() or (style == 1 - template).all()
