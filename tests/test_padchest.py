"""PadChest adapter: canonical frontal image, labels and held-out patients per study."""

import pandas as pd
import pytest

from labrag import paths
from labrag.data import padchest

needs_data = pytest.mark.skipif(
    not all((paths.CXR_ROOT / f).exists() for f in padchest.SOURCE_FILES),
    reason="PadChest sources not mounted",
)


def _images(rows):
    cols = ["study_id", "subject_id", "view", "image_path", "gr_split"]
    df = pd.DataFrame(rows, columns=cols)
    df["labels"] = [("normal",)] * len(df)
    df["label_method"] = "Physician"
    return df


def test_parse_labels():
    assert padchest.parse_labels("['normal']") == ("normal",)
    assert padchest.parse_labels("['b', ' a', '', 'b']") == ("a", "b")
    assert padchest.parse_labels("['']") is None
    assert padchest.parse_labels(float("nan")) is None


def test_frontal_preference_and_path_ties():
    images = _images(
        [
            ("s1", "p1", "L", "a0.png", None),
            ("s1", "p1", "AP_horizontal", "a1.png", None),
            ("s1", "p1", "AP", "a2.png", None),
            ("s2", "p2", "AP_horizontal", "b1.png", None),
            ("s2", "p2", "COSTAL", "b0.png", None),
            ("s3", "p3", "PA", "c2.png", None),
            ("s3", "p3", "PA", "c1.png", None),
        ]
    )
    studies = padchest.canonical_studies(images).set_index("study_id")
    assert studies.image_path.to_dict() == {
        "s1": "a2.png",
        "s2": "b1.png",
        "s3": "c1.png",
    }


def test_studies_without_a_usable_frontal_drop_out():
    (corrupt,) = sorted(padchest.CORRUPT_IMAGES)[:1]
    images = _images(
        [
            ("s1", "p1", "L", "a0.png", None),
            ("s1", "p1", "UNK", "a1.png", None),
            ("s2", "p2", "PA", corrupt, None),
            ("s2", "p2", "L", "b1.png", None),
            ("s3", "p3", "EXCLUDE", "c0.png", None),
            ("s4", "p4", "PA", "d0.png", None),
        ]
    )
    assert padchest.canonical_studies(images).study_id.tolist() == ["s4"]


def test_split_is_strictest_over_all_images_and_roles(tmp_path):
    # s1's lateral names a PadChest-GR test patient: the whole study is held out.
    images = _images(
        [
            ("s1", "p1", "PA", "a0.png", "train"),
            ("s1", "p9", "L", "a1.png", "test"),
            ("s2", "p2", "PA", "b0.png", "validation"),
            ("s3", "p3", "PA", "c0.png", "train"),
            ("s4", "p4", "PA", "d0.png", None),
            ("s5", "p5", "PA", "e0.png", None),
        ]
    )
    studies = padchest.canonical_studies(images).set_index("study_id")
    assert studies.subject_id["s1"] == "p1"
    pd.DataFrame({"subject_id": ["p4"]}).to_csv(
        tmp_path / padchest.CLASSIFIER_VAL_SPLIT_FILE, index=False
    )
    roles = padchest.classifier_roles(studies, tmp_path)
    assert roles.to_dict() == {
        "s1": "padchest_gr_test",
        "s2": "padchest_gr_val",
        "s3": "train",
        "s4": "val",
        "s5": "train",
    }


def test_val_draw_skips_padchest_gr_validation_and_test_patients(monkeypatch):
    monkeypatch.setattr(padchest, "CLASSIFIER_VAL_FRACTION", 1.0)
    studies = pd.DataFrame(
        {
            "subject_id": ["p1", "p2", "p3", "p3", "p4"],
            "split": [None, "train", None, "test", "validation"],
        }
    )
    drawn = padchest.draw_classifier_val_split(studies)
    assert drawn.subject_id.tolist() == ["p1", "p2"]


def test_vocabulary_counts_train_only_and_leaves_out_non_findings():
    studies = pd.DataFrame(
        {
            "labels": [
                ("b", "unchanged"),
                ("a", "b"),
                ("a", "exclude"),
                ("c",),
                None,
                ("c", "d"),
            ]
        }
    )
    roles = pd.Series(["train", "train", "train", "train", "train", "val"])
    assert padchest.vocabulary(studies, roles, 2) == [("a", 2), ("b", 2)]
    assert padchest.vocabulary(studies, roles, 9) == [("a", 2), ("b", 2), ("c", 1)]


@needs_data
def test_release_studies():
    images = padchest.load_images(paths.CXR_ROOT)
    studies = padchest.load_studies(paths.CXR_ROOT)
    assert studies.study_id.is_unique
    assert studies.view.isin(padchest.VIEW_RANK).all()
    assert not studies.image_path.isin(padchest.CORRUPT_IMAGES).any()
    # Labels are per study: every image of a study carries the same set.
    assert (images.groupby("study_id").labels.nunique(dropna=False) == 1).all()
    # Every PadChest-GR study is in, with PadChest-GR's own image and split.
    gr = pd.read_csv(paths.CXR_ROOT / padchest.GR_TABLE, dtype=str)
    gr = gr.drop_duplicates("StudyID").set_index("StudyID")
    s = studies.set_index("study_id").loc[gr.index]
    assert (s.image_path == gr.ImageID).all()
    assert (s.split == gr.split).all()
    # Studies of held-out patients are neither trained on nor used for thresholds.
    roles = padchest.classifier_roles(studies, paths.SPLITS_DIR)
    held_out = set(gr.PatientID[gr.split != "train"])
    assert roles[studies.subject_id.isin(held_out)].str.startswith("padchest_gr").all()
