import numpy as np
import pandas as pd
import pytest

from labrag import artifacts, paths
from labrag.stages import labels
from labrag.stages.labels import combine

NAMES = ["Cardiomegaly", "Edema", "Support Devices", "No Finding"]
nan = np.nan


def table(rows):
    return pd.DataFrame(rows, columns=["study_id", *NAMES])


def test_combine_takes_positive_then_uncertain_then_negative():
    f = table([["a", 1, nan, nan, nan], ["b", 0, -1, nan, nan], ["c", nan, 0, nan, 1]])
    i = table([["a", 0, 0, nan, nan], ["b", -1, 0, nan, nan], ["c", nan, nan, 1, 1]])
    out = combine([f, i], multiclass=True)
    assert out.index.tolist() == ["a", "b", "c"]
    expected = [[1, 0, nan, nan], [-1, -1, nan, nan], [nan, 0, 1, 1]]
    np.testing.assert_array_equal(out.to_numpy(), np.array(expected, dtype=float))


@pytest.mark.parametrize("multiclass, cleared", [(True, nan), (False, 0)])
def test_combine_pathologies_win_over_no_finding(multiclass, cleared):
    # findings states no pathology (No Finding), impression a pathology: not normal;
    # a device alone keeps No Finding
    f = table([["a", 0, 0, 0, 1], ["b", 0, 0, 1, 1]])
    i = table([["a", 1, 0, 0, 0], ["b", 0, 0, 0, 1]])
    out = combine([f, i], multiclass)
    np.testing.assert_array_equal(out["No Finding"].to_numpy(), [cleared, 1])


def test_combine_refuses_unknown_values():
    with pytest.raises(AssertionError, match="unexpected"):
        combine([table([["a", 2, nan, nan, nan]])], multiclass=True)


def test_joint_labels_combine_the_studies_with_both_sections(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "SHARED_ROOT", tmp_path)
    sections = {
        "findings": table([["a", 0, nan, nan, nan], ["b", nan, nan, nan, 1]]),
        "impression": table([["a", 1, nan, nan, nan], ["c", 1, nan, nan, nan]]),
    }
    for section, t in sections.items():

        def build(out, t=t):
            t.to_csv(out / "labels.csv", index=False)
            return {}

        artifacts.publish(
            f"labels/mimic/{section}/chexbert14", {"stage": "labels"}, {}, build
        )
    out = labels.run("mimic", "findings+impression")
    assert out == tmp_path / "labels/mimic/findings+impression/chexbert14"
    got = pd.read_csv(out / "labels.csv")
    assert got.study_id.tolist() == ["a"]  # b and c lack a section
    assert got.columns.tolist() == ["study_id", *NAMES]
    assert got.Cardiomegaly.tolist() == [1]
    config = artifacts.read_manifest(out)["config"]
    assert config["combined"] == ["findings", "impression"]
