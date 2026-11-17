import numpy as np
import pytest

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


def test_every_diagnostic_mixes_and_has_a_prompt():
    for condition in DIAGNOSTICS:
        mix_labels(condition, PRED, REF)
        assert condition in DEFAULT_PROMPT
    with pytest.raises(ValueError):
        mix_labels("labrag", PRED, REF)
