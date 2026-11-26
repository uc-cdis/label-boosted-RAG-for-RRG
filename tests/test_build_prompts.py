import numpy as np
import pandas as pd
import pytest

from labrag.data import priors
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


def test_shuffled_labels_come_from_another_patient():
    subjects = np.array([1, 1, 2, 3, 3, 3, 4, 5, 6, 7])
    donor = build_prompts.donors(subjects, 0)
    assert sorted(donor) == list(range(len(subjects)))  # each target's labels used once
    assert not (subjects[donor] == subjects).any()
    assert (donor == build_prompts.donors(subjects, 0)).all()  # fixed by the seed
    with pytest.raises(SystemExit, match="nothing to shuffle"):
        build_prompts.donors(np.array([1, 1]), 0)


@pytest.mark.parametrize(
    "condition, prompt",
    [
        ("no-context", "joint-naive-negatives-indication-prior-changes-image"),
        ("retrieval-only", "joint-no-context-negatives-indication-prior-changes-image"),
        ("retrieval-only", "joint-simple-negatives-indication-prior-changes-image"),
        ("labrag-oracle", "joint-no-context-negatives-indication-prior-changes-image"),
    ],
)
def test_prompt_must_fit_the_condition(condition, prompt):
    with pytest.raises(SystemExit, match="does not fit condition"):
        build_prompts.run("test", "mimic", "findings+impression", condition, prompt)


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
        run(
            "pilot",
            "chexpertplus",
            "findings",
            "retrieval-only",
            corpus_labels="predicted",
        )
    with pytest.raises(SystemExit, match="-oofK"):
        run(
            "pilot",
            "chexpertplus",
            "findings",
            "labrag",
            classifier=CLF,
            corpus_labels="predicted",
        )
    with pytest.raises(SystemExit, match="its own labels"):
        run(
            "pilot",
            "chexpertplus",
            "findings",
            "labrag",
            classifier=CLF,
            predictions="a/b/c/d",
        )
    with pytest.raises(SystemExit, match="are for predicted labels"):
        run("pilot", "chexpertplus", "findings", "labrag-oracle", predictions=CLF)


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
            "pilot",
            "chexpertplus",
            "findings",
            "labrag",
            predictions=predictions,
            **kwargs,
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
            "pilot",
            "chexpertplus",
            "findings",
            "labrag",
            classifier=classifier,
            **kwargs,
        )


def test_corpus_studies_without_predictions_are_refused():
    pred = pd.DataFrame({"A": [1]}, index=["t1"])
    corpus = pd.DataFrame({"role": ["train", "train"]}, index=["t1", "lateral"])
    with pytest.raises(SystemExit, match="1 corpus studies have no predicted"):
        build_prompts.predicted_rows(pred, corpus)


def test_image_ranking_is_for_label_conditions():
    with pytest.raises(SystemExit, match="--ranking is for conditions with labels"):
        build_prompts.run(
            "pilot", "chexpertplus", "findings", "retrieval-only", ranking="image"
        )


def test_confidence_is_for_labrag():
    with pytest.raises(SystemExit, match="--confidence is for labrag"):
        build_prompts.run(
            "pilot", "chexpertplus", "findings", "labrag-oracle", confidence=True
        )


def test_sides_are_for_the_oracle():
    with pytest.raises(SystemExit, match="--sides is for labrag-oracle"):
        build_prompts.run(
            "pilot", "chexpertplus", "findings", "labrag", classifier=CLF, sides=True
        )


def test_sample_takes_one_study_per_drawn_patient():
    target = pd.DataFrame(
        {"subject_id": ["p1", "p1", "p1", "p2", "p3"]},
        index=["s3", "s1", "s2", "s4", "s5"],
    )
    picked = build_prompts.sample_targets(target, patients=2, seed=0)
    assert picked.subject_id.is_unique and len(picked) == 2
    assert list(picked.index) == sorted(picked.index)
    again = build_prompts.sample_targets(target.sample(frac=1, random_state=1), 2, 0)
    assert list(again.index) == list(picked.index)  # row order doesn't matter


def test_experiment_dataset_is_checked():
    with pytest.raises(SystemExit, match="smoke runs on"):
        build_prompts.run("smoke", "chexpertplus", "findings", "retrieval-only")


FT = "chexpertplus-mimic-rexgradient/raddino.ft-attn-seed1/findings-impression/chexbert14"


def test_category_prompts_need_multiclass_predictions():
    with pytest.raises(SystemExit, match="shows label categories"):
        build_prompts.run(
            "val",
            "mimic",
            "findings",
            "labrag",
            predictions=FT,
            prompt="section-verbose-image",
        )
    with pytest.raises(SystemExit, match="multiclass predictions are for labrag"):
        build_prompts.run(
            "val",
            "mimic",
            "findings",
            "labrag",
            predictions=FT + ".mc",
            confidence=True,
        )


def test_absent_prompts_need_multiclass_predictions():
    with pytest.raises(SystemExit, match="lists absent labels"):
        build_prompts.run(
            "val",
            "mimic",
            "findings",
            "labrag",
            predictions=FT,
            prompt="section-simple-absent-image",
        )


def test_joint_takes_only_joint_prompts_and_back():
    with pytest.raises(SystemExit, match="joint-"):
        build_prompts.run("val", "mimic", "findings+impression", "retrieval-only",
                          prompt="section-naive-negatives-indication-image")  # fmt: skip
    with pytest.raises(SystemExit, match="joint-"):
        build_prompts.run("val", "mimic", "findings", "retrieval-only",
                          prompt="joint-naive-negatives-indication-image")  # fmt: skip


def test_joint_needs_both_readouts():
    with pytest.raises(SystemExit, match="no findings\\+impression readouts"):
        build_prompts.run(
            "val", "mimic", "findings+impression", "labrag",
            prompt="joint-simple-negatives-indication-image",
            predictions="chexpertplus-mimic-rexgradient/raddino.ft-attn-seed1/"
            "findings/chexbert14.mc",
        )  # fmt: skip


def test_same_needs_examples_and_the_group():
    with pytest.raises(SystemExit, match="--same reader is for conditions with"):
        build_prompts.run("val", "mimic", "findings", "no-context", same="reader")
    with pytest.raises(SystemExit, match="rexgradient has no readers"):
        build_prompts.run(
            "val", "rexgradient", "findings", "retrieval-only", same="reader"
        )
    with pytest.raises(SystemExit, match="mimic has no sites"):
        build_prompts.run("val", "mimic", "findings", "retrieval-only", same="site")


def test_severity_and_devices_are_for_the_oracle():
    with pytest.raises(SystemExit, match="--severity is for labrag-oracle"):
        build_prompts.run(
            "val", "mimic", "findings", "labrag", predictions=FT, severity=True
        )
    with pytest.raises(SystemExit, match="--incidentals is for labrag-oracle, labrag"):
        build_prompts.run(
            "val", "mimic", "findings", "retrieval-only", incidentals=True
        )
    with pytest.raises(
        SystemExit, match="needs --predictions of a chexbert14.devices.incidentals"
    ):
        build_prompts.run(
            "val",
            "mimic",
            "findings",
            "labrag",
            predictions=FT + ".devices.mc",
            incidentals=True,
        )
    with pytest.raises(SystemExit, match="--devices is for labrag-oracle, labrag"):
        build_prompts.run("val", "mimic", "findings", "retrieval-only", devices=True)
    with pytest.raises(SystemExit, match="needs --predictions of a chexbert14.devices"):
        build_prompts.run(
            "val", "mimic", "findings", "labrag", predictions=FT + ".mc", devices=True
        )
    with pytest.raises(SystemExit, match="reference corpus labels"):
        build_prompts.run(
            "val",
            "mimic",
            "findings",
            "labrag",
            predictions=FT + ".devices.mc",
            uncertain="ones",
        )
    with pytest.raises(SystemExit, match="not both"):
        build_prompts.run(
            "val", "mimic", "findings", "labrag-oracle", sides=True, severity=True
        )


def test_prior_text_takes_the_latest_earlier_study_with_a_report():
    class Adapter:
        SECTIONS = ("findings", "impression")

        @staticmethod
        def load_study_times(cxr_root):
            return pd.Series({"s1": 1, "s2": 2, "s3": 3, "s4": 4, "s5": 1})

    nan = float("nan")
    studies = pd.DataFrame(
        {
            "subject_id": ["a", "a", "a", "a", "b"],
            "findings": ["F1", nan, nan, "F4", "F5"],
            "impression": ["I1", nan, "I3", "I4", nan],
        },
        index=["s1", "s2", "s3", "s4", "s5"],
    )
    got = build_prompts.prior_text(studies, Adapter, None)
    # s2 has no report: s3's prior skips it; s1 and s5 have none
    assert got == {
        "s3": ("s1", "Findings: F1 Impression: I1"),
        "s4": ("s3", "Impression: I3"),
    }


def test_priors_need_a_strictly_earlier_time():
    index = pd.Index(list("pqrst"), name="study_id")  # as load_studies' index
    studies = pd.DataFrame({"subject_id": ["a"] * 4 + ["b"]}, index=index)
    times = pd.Series({"p": 3, "q": 1, "r": 2, "s": 2, "t": 1})
    got = priors(studies, times).fillna("-").to_dict()
    # r and s share a time: neither is the other's prior; p takes s (ties by id)
    assert got == {"p": "s", "q": "-", "r": "q", "s": "q", "t": "-"}


def test_labels_only_takes_no_ranking_or_same():
    labels_only = "joint-labels-only-negatives-indication-prior-changes-image"
    with pytest.raises(SystemExit, match="--ranking is for conditions with labels and"):
        build_prompts.run(
            "test",
            "mimic",
            "findings+impression",
            "labrag-oracle",
            labels_only,
            ranking="image",
        )
    with pytest.raises(SystemExit, match="--same reader is for conditions with"):
        build_prompts.run(
            "test",
            "mimic",
            "findings+impression",
            "labrag-oracle",
            labels_only,
            same="reader",
        )
