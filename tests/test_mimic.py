"""MIMIC-CXR adapter: frontal image per study and upstream's sections."""

import subprocess
import sys

import pandas as pd
import pytest

from labrag import paths
from labrag.data import mimic

needs_data = pytest.mark.skipif(
    not (paths.CXR_ROOT / mimic.METADATA_CSV).exists(),
    reason="MIMIC-CXR not mounted",
)

REPORT = """                                 FINAL REPORT
 EXAMINATION:  CHEST (PA AND LAT)

 FINDINGS:

 First findings.

 IMPRESSION:

 First impression.

 FINDINGS:

 Last findings.
"""


def test_last_section_of_each_name():
    out = mimic.section(REPORT, "1")
    assert out == {
        "findings": "Last findings.",
        "impression": "First impression.",
        "indication": None,
    }


def test_missing_section_is_none():
    out = mimic.section(" EXAMINATION:  CHEST\n\n IMPRESSION:  Only this.\n", "1")
    assert out == {"findings": None, "impression": "Only this.", "indication": None}


def test_indication_joins_history_then_indication():
    report = (
        " INDICATION:  Eval for PNA.\n\n HISTORY:  Cough.\n\n"
        " REASON FOR EXAM:  Fever.\n\n FINDINGS:  Clear.\n"
    )
    assert mimic.section(report, "1")["indication"] == "Cough. Fever."


def test_custom_rule_reports_have_no_sections():
    names, indices = mimic._parser().custom_mimic_cxr_rules()
    for stem in (next(iter(names)), next(iter(indices))):
        out = mimic.section(REPORT, stem[1:])
        assert out == {"findings": None, "impression": None, "indication": None}


def test_view_preference_and_laterals():
    images = pd.DataFrame(
        [
            ("1", "s1", "b.jpg", "train", "AP"),
            ("1", "s1", "c.jpg", "train", "PA"),
            ("1", "s1", "a.jpg", "train", "LATERAL"),
            ("2", "s2", "b.jpg", "train", "AP"),
            ("2", "s2", "a.jpg", "train", "AP"),
            ("3", "s3", "a.jpg", "train", "LL"),
            ("4", "s4", "a.jpg", "train", None),
        ],
        columns=["subject_id", "study_id", "image_path", "split", "view"],
    )
    studies = mimic.canonical_studies(images).set_index("study_id")
    assert studies.image_path.to_dict() == {"s1": "c.jpg", "s2": "a.jpg"}


@needs_data
def test_sections_match_upstream_sectioned_csv(tmp_path):
    script = mimic.PARSER.parent / "create_section_files.py"
    subprocess.run(
        [
            sys.executable,
            script.name,
            "--reports_path",
            str(paths.CXR_ROOT / mimic.REPORTS_DIR),
            "--output_path",
            str(tmp_path),
            "--no_split",
        ],
        cwd=script.parent,
        check=True,
        capture_output=True,
    )
    up = pd.read_csv(
        tmp_path / "mimic_cxr_sectioned.csv", dtype=str, keep_default_na=False
    )
    up = up.set_index(up.study.str[1:])
    studies = mimic.load_studies(paths.CXR_ROOT).set_index("study_id")
    names, indices = mimic._parser().custom_mimic_cxr_rules()
    custom = {s[1:] for s in [*names, *indices]}
    assert set(studies.index) - set(up.index) <= custom
    up = up[["findings", "impression"]].add_suffix("_up")
    both = studies.join(up, how="inner")
    for section in mimic.SECTIONS:
        assert (both[section].fillna("") == both[f"{section}_up"]).all(), section


@needs_data
def test_release_studies():
    studies = mimic.load_studies(paths.CXR_ROOT)
    assert studies.study_id.is_unique
    assert studies.view.isin(mimic.FRONTAL_VIEWS).all()
    roles = mimic.classifier_roles(studies, paths.SPLITS_DIR)
    assert roles.value_counts().to_dict() == {
        "train": 213365,
        "test": 3041,
        "val": 1733,
    }
