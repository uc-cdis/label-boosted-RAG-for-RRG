import numpy as np
import pandas as pd
import pytest

from labrag import data


def test_add_joint_joins_studies_with_both_sections():
    studies = pd.DataFrame(
        {
            "findings": [" Lungs clear.\n", "Effusion.", np.nan],
            "impression": ["Normal. ", np.nan, "Normal."],
        }
    )
    joint = data.add_joint(studies)[data.JOINT]
    assert joint.iloc[0] == "Lungs clear. Normal."
    assert joint.iloc[1:].isna().all()


def test_joint_is_a_section_only_where_asked_and_both_parts_exist():
    data.check_section("x", "mimic", data.JOINT, joint=True)
    with pytest.raises(SystemExit, match="sections are"):
        data.check_section("x", "mimic", data.JOINT)
    with pytest.raises(SystemExit, match="sections are"):
        data.check_section("x", "padchest", data.JOINT, joint=True)


def test_joint_is_never_a_classifier_readout():
    with pytest.raises(SystemExit, match="sections are"):
        data.split_sections("classify", "mimic", data.JOINT)
