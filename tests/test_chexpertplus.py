"""CheXpert Plus adapter: canonical image and report text per study."""

import pandas as pd
import pytest

from labrag import paths
from labrag.data import chexpertplus

needs_data = pytest.mark.skipif(
    not (paths.CXR_ROOT / chexpertplus.SOURCE_CSV).exists(),
    reason="CheXpert Plus source CSV not mounted",
)


def _images(rows):
    cols = ["study_id", "view", "image_path", "findings"]
    df = pd.DataFrame(rows, columns=cols)
    df["subject_id"] = df.study_id.str.split("_").str[0]
    df["split"] = "train"
    df["impression"] = None
    return df


def test_view_preference_beats_path_order():
    # Each study's views are listed in path order, so path order alone would pick
    # the first row; the view rank must pick the expected one instead.
    images = _images(
        [
            ("p1_s1", "Lateral", "PNG/a/view1_lateral.png", "lateral"),
            ("p1_s1", "AP", "PNG/a/view2_frontal.png", "ap"),
            ("p1_s1", "PA", "PNG/a/view3_frontal.png", "pa"),
            ("p2_s1", "LL", "PNG/b/view1_frontal.png", "ll"),
            ("p2_s1", "AP", "PNG/b/view2_frontal.png", "ap"),
            ("p3_s1", "Lateral", "PNG/c/view1_lateral.png", "lateral"),
            ("p3_s1", "RL", "PNG/c/view2_frontal.png", "rl"),
            ("p4_s1", "RL", "PNG/d/view1_frontal.png", "rl"),  # LL/RL tie: path
            ("p4_s1", "LL", "PNG/d/view2_frontal.png", "ll"),
        ]
    )
    studies = chexpertplus.canonical_studies(images).set_index("study_id")
    assert studies.findings.to_dict() == {
        "p1_s1": "pa",
        "p2_s1": "ap",
        "p3_s1": "rl",
        "p4_s1": "rl",
    }


def test_path_breaks_ties_within_a_view():
    images = _images(
        [
            ("p1_s1", "AP", "PNG/a/view2_frontal.png", "second"),
            ("p1_s1", "AP", "PNG/a/view1_frontal.png", "first"),
        ]
    )
    studies = chexpertplus.canonical_studies(images)
    assert studies.image_path.tolist() == ["PNG/a/view1_frontal.png"]
    assert studies.findings.tolist() == ["first"]


def test_corrupt_image_is_skipped_and_text_is_cleaned():
    (corrupt,) = chexpertplus.CORRUPT_IMAGES
    images = _images(
        [
            ("p1_s1", "PA", corrupt, "corrupt row"),
            ("p1_s1", "Lateral", "PNG/a/view2_lateral.png", "  kept text \n"),
            ("p2_s1", "PA", "PNG/b/view1_frontal.png", " / "),
        ]
    )
    studies = chexpertplus.canonical_studies(images).set_index("study_id")
    assert studies.loc["p1_s1", "findings"] == "kept text"
    assert pd.isna(studies.loc["p2_s1", "findings"])


@needs_data
def test_release_studies():
    images = chexpertplus.load_images(paths.CXR_ROOT)
    studies = chexpertplus.load_studies(paths.CXR_ROOT)
    assert studies.study_id.is_unique
    assert len(studies) == images.study_id.nunique() - 1  # the corrupt-image study
    assert (studies.split == "valid").sum() > 0
    # 61 valid studies with Findings (docs/revision-plan.md §2)
    assert studies[studies.split == "valid"].findings.notna().sum() == 61
    for path in studies.image_path.head(100):
        assert (paths.CXR_ROOT / "chexpertplus" / path).is_file()
