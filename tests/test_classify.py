"""Classify: the No Finding consistency rule, token loading."""

import numpy as np
import pytest
import torch

from labrag.classifiers import quant
from labrag.stages import classify

NAMES = ["Edema", "Support Devices", "No Finding"]


def test_pathology_overrides_no_finding_support_devices_does_not():
    pred = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 0], [0, 0, 0]])
    out = classify.consistent(pred, NAMES)
    assert out.tolist() == [[1, 0, 0], [0, 1, 1], [0, 0, 1], [1, 1, 0], [0, 0, 0]]
    assert pred[0, 2] == 1  # input untouched


def test_padchest_normal_is_cleared_by_any_other_label():
    names = ["NSG tube", "cardiomegaly", "normal"]
    pred = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]])
    out = classify.consistent(pred, names, "normal", ())
    assert out.tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_section_only_for_datasets_with_sections():
    with pytest.raises(SystemExit, match="drop --section"):
        classify.run("padchest", section="findings")
    with pytest.raises(SystemExit, match="needs --section"):
        classify.run("chexpertplus")
    with pytest.raises(SystemExit, match="no uncertain"):
        classify.run("padchest", uncertain="ones")


def test_load_tokens_keeps_the_first_tokens(tmp_path):
    x = np.random.default_rng(0).normal(size=(5, 4, 16)).astype(np.float16)
    np.save(tmp_path / "tokens.npy", x)
    rows = np.array([3, 0, 4])
    path, cpu = tmp_path, torch.device("cpu")
    full = classify._load_tokens(path, rows, "fp16", cpu)
    cls = classify._load_tokens(path, rows, "fp16", cpu, keep=1)
    assert full.take(torch.arange(3)).numpy().tolist() == x[rows].tolist()
    assert cls.take(torch.arange(3)).numpy().tolist() == x[rows, :1].tolist()


def test_packed_store_reads_as_the_float16_store_packed(tmp_path):
    # embed's nvfp4 store must give classify exactly the tokens it would pack from a
    # float16 store of the same hidden states.
    x = np.random.default_rng(0).normal(size=(5, 4, 32)).astype(np.float32)
    fp16_dir, packed_dir = tmp_path / "fp16", tmp_path / "packed"
    fp16_dir.mkdir()
    packed_dir.mkdir()
    np.save(fp16_dir / "tokens.npy", x.astype(np.float16))
    store = quant.create_store(packed_dir, "nvfp4", x.shape)
    for start in (0, 3):  # written in two batches, as embed does
        half = torch.from_numpy(x[start : start + 3]).half()
        quant.write_rows(store, start, quant.pack("nvfp4", half))
    del store
    assert quant.stored_format(packed_dir) == "nvfp4"
    rows, cpu = np.array([4, 0, 2]), torch.device("cpu")
    for keep in (None, 1):
        a = classify._load_tokens(fp16_dir, rows, "nvfp4", cpu, keep)
        b = classify._load_tokens(packed_dir, rows, "nvfp4", cpu, keep)
        for k in a.parts:
            assert torch.equal(
                a.parts[k].view(torch.uint8), b.parts[k].view(torch.uint8)
            )


def test_folds_split_train_patients_only():
    subject = np.array(["a", "a", "b", "c", "d", "e", "e", "f"])
    train = np.array([True, True, True, True, True, True, False, False])
    fold = classify.fold_of(subject, train, 2)
    assert fold[0] == fold[1]  # one patient, one fold
    assert fold[6] == fold[7] == -1  # not train: no fold
    patients = {s: f for s, f, t in zip(subject, fold, train) if t}
    assert sorted(np.bincount(list(patients.values()))) == [2, 3]  # 5 patients


def test_folds_are_for_attn_only():
    with pytest.raises(SystemExit, match="--folds"):
        classify.run("chexpertplus", section="findings", folds=5)
