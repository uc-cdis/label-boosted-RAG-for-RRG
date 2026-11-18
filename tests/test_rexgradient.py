"""ReXGradient adapter: labelled frontal image per study and its sections."""

import pandas as pd
import pytest

from labrag import paths
from labrag.data import rexgradient

needs_data = pytest.mark.skipif(
    not all((paths.CXR_ROOT / f).exists() for f in rexgradient.SOURCE_FILES),
    reason="ReXGradient not mounted",
)


def _images(rows):
    df = pd.DataFrame(rows, columns=["study_id", "image_path", "view", "findings"])
    df["subject_id"] = "p" + df.study_id
    df["split"] = "train"
    df["impression"] = "imp"
    return df


def test_labelled_frontal_only_by_preference():
    images = _images(
        [
            ("1", "b.png", "AP", "f1"),
            ("1", "c.png", "POSTERO_ANTERIOR", "f1"),
            ("1", "a.png", "LATERAL", "f1"),
            ("2", "b.png", "PA", "f2"),
            ("2", "a.png", "PA", "f2"),
            ("3", "a.png", "UNKNOWN", "f3"),
            ("3", "b.png", "LATERAL", "f3"),
            ("4", "a.png", "N/A", "f4"),
            ("5", "a.png", None, "f5"),
        ]
    )
    studies = rexgradient.canonical_studies(images).set_index("study_id")
    assert studies.image_path.to_dict() == {"1": "c.png", "2": "a.png"}


def test_blank_text_is_nan():
    studies = rexgradient.canonical_studies(_images([("1", "a.png", "PA", "  ")]))
    assert pd.isna(studies.findings.iloc[0])
    assert studies.impression.iloc[0] == "imp"


@needs_data
def test_release_studies():
    studies = rexgradient.load_studies(paths.CXR_ROOT)
    assert studies.study_id.is_unique
    assert studies.image_path.str.startswith("deid_png/").all()
    roles = rexgradient.classifier_roles(studies, paths.SPLITS_DIR)
    counts = roles.value_counts().to_dict()
    assert counts == {"train": 108127, "val": 7745, "test": 7755}, counts
    assert studies.findings.notna().all() and studies.impression.notna().all()
