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
        tmp_path / "oof_predictions.findings.csv", index=False
    )
    pd.DataFrame(
        {"study_id": ["t1", "t2", "v1"], "A": [0, 1, 1], "B": [1, 1, 0]}
    ).to_csv(tmp_path / "predictions.findings.csv", index=False)
    corpus = pd.DataFrame({"role": ["val", "train", "train"]}, index=["v1", "t2", "t1"])
    y = build_prompts.predicted_corpus(tmp_path, "findings", corpus, names)
    assert y.tolist() == [[1, 0], [0, 0], [1, 0]]


CLF = "chexpertplus/raddino.attn-nvfp4-seed1/findings-impression/chexbert14"


def test_predicted_corpus_options_are_checked():
    run = build_prompts.run
    with pytest.raises(SystemExit, match="labrag only"):
        run("pilot", "findings", "retrieval-only", corpus_labels="predicted")
    with pytest.raises(SystemExit, match="-oofK"):
        run("pilot", "findings", "labrag", classifier=CLF, corpus_labels="predicted")
    with pytest.raises(SystemExit, match="its own labels"):
        run("pilot", "findings", "labrag", classifier=CLF, predictions="a/b/c/d")


@pytest.mark.parametrize(
    "predictions, kwargs, match",
    [
        ("padchest/raddino.attn-nvfp4-seed1/report/padchest60", {}, "no padchest60"),
        (CLF, {"uncertain": "ones"}, "no chexbert14 reference"),
    ],
)
def test_reference_corpus_needs_the_experiment_labeler(predictions, kwargs, match):
    # transferred labels come with no reference labels of their own for the corpus
    with pytest.raises(SystemExit, match=match):
        build_prompts.run(
            "pilot", "findings", "labrag", predictions=predictions, **kwargs
        )


@pytest.mark.parametrize(
    "classifier, kwargs, match",
    [
        (None, {}, "need --classifier"),
        ("chexpertplus/raddino.lr/chexbert14", {}, "is not <dataset>"),
        ("padchest/raddino.attn-nvfp4-seed1/report/padchest60", {}, "--predictions"),
        (CLF, {"uncertain": "ones"}, "gives chexbert14.u1, not chexbert14"),
        ("chexpertplus/raddino.lr/impression/chexbert14", {}, "no findings readout"),
    ],
)
def test_classifier_path_is_checked(classifier, kwargs, match):
    with pytest.raises(SystemExit, match=match):
        build_prompts.run(
            "pilot", "findings", "labrag", classifier=classifier, **kwargs
        )


def test_corpus_studies_without_predictions_are_refused():
    pred = pd.DataFrame({"A": [1]}, index=["t1"])
    corpus = pd.DataFrame({"role": ["train", "train"]}, index=["t1", "lateral"])
    with pytest.raises(SystemExit, match="1 corpus studies have no predicted"):
        build_prompts.predicted_rows(pred, corpus)


def test_image_ranking_is_for_label_conditions():
    with pytest.raises(SystemExit, match="--ranking is for conditions with labels"):
        build_prompts.run("pilot", "findings", "retrieval-only", ranking="image")
