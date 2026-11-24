"""CheXpert Plus pilot eligibility (tests/test_splits.py checks the committed file)."""

import pytest

from labrag import paths
from labrag.data import chexpertplus

needs_data = pytest.mark.skipif(
    not (paths.CXR_ROOT / chexpertplus.SOURCE_CSV).exists(),
    reason="CheXpert Plus source CSV not mounted",
)


@pytest.mark.data
@needs_data
def test_pilot_eligibility_counts():
    images = chexpertplus.load_images(paths.CXR_ROOT)
    train = images[(images.split == "train") & chexpertplus.has_text(images.findings)]
    per_patient = train.drop_duplicates("study_id").groupby("subject_id").size()
    assert len(per_patient) == 26_634
    assert (per_patient == 1).sum() == 17_972
