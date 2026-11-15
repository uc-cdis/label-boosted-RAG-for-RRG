"""Classify: the No Finding consistency rule, token loading."""

import numpy as np
import torch

from labrag.stages import classify

NAMES = ["Edema", "Support Devices", "No Finding"]


def test_pathology_overrides_no_finding_support_devices_does_not():
    pred = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 0], [0, 0, 0]])
    out = classify.consistent(pred, NAMES)
    assert out.tolist() == [[1, 0, 0], [0, 1, 1], [0, 0, 1], [1, 1, 0], [0, 0, 0]]
    assert pred[0, 2] == 1  # input untouched


def test_load_tokens_keeps_the_first_tokens(tmp_path):
    x = np.random.default_rng(0).normal(size=(5, 4, 16)).astype(np.float16)
    np.save(tmp_path / "tokens.npy", x)
    rows = np.array([3, 0, 4])
    path, cpu = tmp_path / "tokens.npy", torch.device("cpu")
    full = classify._load_tokens(path, rows, "fp16", cpu)
    cls = classify._load_tokens(path, rows, "fp16", cpu, keep=1)
    assert full.take(torch.arange(3)).numpy().tolist() == x[rows].tolist()
    assert cls.take(torch.arange(3)).numpy().tolist() == x[rows, :1].tolist()
