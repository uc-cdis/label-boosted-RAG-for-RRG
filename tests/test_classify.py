"""Classify: the No Finding consistency rule."""

import numpy as np

from labrag.stages import classify

NAMES = ["Edema", "Support Devices", "No Finding"]


def test_pathology_overrides_no_finding_support_devices_does_not():
    pred = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 0], [0, 0, 0]])
    out = classify.consistent(pred, NAMES)
    assert out.tolist() == [[1, 0, 0], [0, 1, 1], [0, 0, 1], [1, 1, 0], [0, 0, 0]]
    assert pred[0, 2] == 1  # input untouched
