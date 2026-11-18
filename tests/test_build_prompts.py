import numpy as np
import pandas as pd
import pytest

from labrag.stages import build_prompts
from labrag.stages.build_prompts import DEFAULT_PROMPT, DIAGNOSTICS, mix_labels

PRED = np.array([[1, 1, 0, 0], [0, 0, 0, 1]])
REF = np.array([[1, 0, 1, 0], [0, 0, 0, 0]])


def test_drop_fp_keeps_only_true_positives():
    # study 2's only prediction is false: it is left with no labels
    assert mix_labels("labrag-drop-fp", PRED, REF).tolist() == [[1, 0, 0, 0], [0] * 4]


def test_add_fn_adds_the_misses():
    assert mix_labels("labrag-add-fn", PRED, REF).tolist() == [
        [1, 1, 1, 0],
        [0, 0, 0, 1],
    ]


def test_every_diagnostic_mixes_and_has_a_prompt():
    for condition in DIAGNOSTICS:
        mix_labels(condition, PRED, REF)
        assert condition in DEFAULT_PROMPT
    with pytest.raises(ValueError):
        mix_labels("labrag", PRED, REF)


def test_predicted_corpus_takes_out_of_fold_labels_for_train(tmp_path):
    names = ["A", "B"]
    pd.DataFrame({"study_id": ["t1", "t2"], "A": [1, 0], "B": [0, 0]}).to_csv(
        tmp_path / "oof_predictions.csv", index=False
    )
    pd.DataFrame(
        {"study_id": ["t1", "t2", "v1"], "A": [0, 1, 1], "B": [1, 1, 0]}
    ).to_csv(tmp_path / "predictions.csv", index=False)
    corpus = pd.DataFrame({"role": ["val", "train", "train"]}, index=["v1", "t2", "t1"])
    y = build_prompts.predicted_corpus(tmp_path, corpus, names)
    assert y.tolist() == [[1, 0], [0, 0], [1, 0]]


def test_predicted_corpus_options_are_checked():
    run = build_prompts.run
    with pytest.raises(SystemExit, match="labrag only"):
        run("pilot", "findings", "retrieval-only", corpus_labels="predicted")
    with pytest.raises(SystemExit, match="-oofK"):
        run("pilot", "findings", "labrag", corpus_labels="predicted")
    with pytest.raises(SystemExit, match="its own labels"):
        run("pilot", "findings", "labrag", classifier="x", predictions="a/b/c")


def test_sections_classifier_gives_its_section_predictions():
    from labrag.stages.build_prompts import classifier_files

    d, f = classifier_files("chexpertplus", "attn-nvfp4-seed1", "findings", "x.u0")
    assert d.parts[-3:] == ("raddino.attn-nvfp4-seed1", "findings", "x.u0")
    assert f == "predictions.csv"
    d, f = classifier_files(
        "chexpertplus", "attn-nvfp4-seed1-sections", "findings", "x.u0"
    )
    assert d.parts[-2:] == ("raddino.attn-nvfp4-seed1-sections", "x.u0")
    assert f == "predictions.findings.csv"
