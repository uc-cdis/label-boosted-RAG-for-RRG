"""CheXpert Plus classifier roles and the frozen validation split."""

import pandas as pd
import pytest

from labrag import paths
from labrag.data import chexpertplus

needs_data = pytest.mark.skipif(
    not (paths.CXR_ROOT / chexpertplus.SOURCE_CSV).exists(),
    reason="CheXpert Plus source CSV not mounted",
)


def test_classifier_roles(tmp_path):
    pd.DataFrame(
        {
            "subject_id": ["p2", "p3"],
            "study_id": ["p2_s1", "p3_s1"],
            "set": ["pilot", "prompt_tuning"],
        }
    ).to_csv(tmp_path / chexpertplus.PILOT_SPLIT_FILE, index=False)
    pd.DataFrame({"subject_id": ["p4"]}).to_csv(
        tmp_path / chexpertplus.CLASSIFIER_VAL_SPLIT_FILE, index=False
    )
    studies = pd.DataFrame(
        {
            "subject_id": ["p1", "p2", "p2", "p3", "p4", "p4", "p5"],
            "split": ["train"] * 6 + ["valid"],
        },
        index=["p1_s1", "p2_s1", "p2_s2", "p3_s1", "p4_s1", "p4_s2", "p5_s1"],
    )
    roles = chexpertplus.classifier_roles(studies, tmp_path)
    assert roles.to_dict() == {
        "p1_s1": "train",
        "p2_s1": "pilot",
        "p2_s2": "pilot",  # every study of a pilot patient, not only the listed one
        "p3_s1": "prompt_tuning",
        "p4_s1": "val",
        "p4_s2": "val",
        "p5_s1": "official_valid",
    }


@needs_data
def test_classifier_val_split_reproduces_committed_file():
    images = chexpertplus.load_images(paths.CXR_ROOT)
    pilot = pd.read_csv(paths.SPLITS_DIR / chexpertplus.PILOT_SPLIT_FILE)
    drawn = chexpertplus.draw_classifier_val_split(images, pilot)
    path = paths.SPLITS_DIR / chexpertplus.CLASSIFIER_VAL_SPLIT_FILE
    assert drawn.to_csv(index=False) == path.read_text()
    assert len(drawn) == 6_412  # 10% of the 64,125 non-pilot train patients
    assert not drawn.subject_id.isin(pilot.subject_id).any()
    train = set(images.loc[images.split == "train", "subject_id"])
    assert drawn.subject_id.isin(train).all()
