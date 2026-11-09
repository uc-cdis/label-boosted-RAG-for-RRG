"""The committed pilot split must be exactly reproducible from the CheXpert Plus release."""

import pytest

from labrag import paths
from labrag.cli import PILOT_SPLIT_FILE
from labrag.data import chexpertplus

needs_data = pytest.mark.skipif(
    not (paths.CXR_ROOT / chexpertplus.SOURCE_CSV).exists(),
    reason="CheXpert Plus source CSV not mounted",
)


@needs_data
def test_pilot_split_reproduces_committed_file():
    images = chexpertplus.load_images(paths.CXR_ROOT)
    drawn = chexpertplus.draw_pilot_split(images)
    committed = (paths.SPLITS_DIR / PILOT_SPLIT_FILE).read_text()
    assert drawn.to_csv(index=False) == committed


@needs_data
def test_pilot_eligibility_counts():
    images = chexpertplus.load_images(paths.CXR_ROOT)
    train = images[(images.split == "train") & chexpertplus.has_text(images.findings)]
    per_patient = train.drop_duplicates("study_id").groupby("subject_id").size()
    assert len(per_patient) == 26_634
    assert (per_patient == 1).sum() == 17_972
