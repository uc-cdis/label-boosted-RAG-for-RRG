"""MIMIC-CXR adapter: MIMIC-CXR-JPG metadata and official split (one row per image), the
free-text reports (one file per study), and the JPGs. Sections come from the authors'
own parser (MIT-LCP mimic-cxr, external/mimic-cxr/txt/section_parser.py), applied as
its create_section_files.py builds mimic_cxr_sectioned.csv: the last section of each
name. The indication is the last "history" then the last "indication" section (the
parser's names for INDICATION, HISTORY, REASON FOR EXAM, ...). Thresholds are set on the
official validate split."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from labrag import paths

METADATA_CSV = "mimic-cxr/mimic-cxr-2.0.0-metadata.csv.gz"
SPLIT_CSV = "mimic-cxr/mimic-cxr-2.0.0-split.csv.gz"
REPORTS_DIR = "mimic-cxr/notes"  # notes/p10/p10000032/s50414267.txt
SOURCE_FILES = (METADATA_CSV, SPLIT_CSV, REPORTS_DIR)  # the reports dir as a tree
PROVIDERS_CSV = "mimic-cxr/mimic-cxr-2.1.0-metadata/cxr-provider-list.csv.gz"
READER_FILES = (PROVIDERS_CSV,)
IMAGE_ROOT = "mimic-cxr/files"  # image_path is relative to this
SECTIONS = ("findings", "impression")
INDICATION = ("history", "indication")  # both in 0.04% of reports: joined in this order

PARSER = paths.REPO_ROOT / "external" / "mimic-cxr" / "txt" / "section_parser.py"

# Frontal views by preference (ties by image_path). LL is a lateral in MIMIC; images
# with no ViewPosition (15,769) are never used, nor are the rare oblique/decubitus ones.
VIEW_RANK = {"PA": 0, "AP": 1}
FRONTAL_VIEWS = tuple(VIEW_RANK)

SPLIT_FILES = ()  # official splits only
ROLES = {"train": "train", "validate": "val", "test": "test"}


def _parser():
    """The upstream section parser, imported unmodified under a private name."""
    spec = importlib.util.spec_from_file_location("mimic_cxr_section_parser", PARSER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def report_path(subject_id: str, study_id: str) -> str:
    """A study's report file, relative to REPORTS_DIR."""
    return f"p{subject_id[:2]}/p{subject_id}/s{study_id}.txt"


def section(text: str, study_id: str, parser=None) -> dict[str, str | None]:
    """Findings, impression and indication of one report. Findings and impression as
    upstream's mimic_cxr_sectioned.csv has them: the last section of that name,
    stripped. Upstream writes no sectioned row for the 54 reports its custom rules
    cover (it only extracts their conclusion), so those have none of the three here."""
    sp = parser or _parser()
    names, indices = sp.custom_mimic_cxr_rules()
    out = dict.fromkeys((*SECTIONS, "indication"))
    if f"s{study_id}" in names or f"s{study_id}" in indices:
        return out
    sections, section_names, _ = sp.section_text(text)

    def last(name: str) -> str | None:
        if name not in section_names:
            return None
        i = len(section_names) - 1 - section_names[::-1].index(name)
        return sections[i].strip() or None

    for name in SECTIONS:
        out[name] = last(name)
    parts = [p for p in map(last, INDICATION) if p]
    out["indication"] = " ".join(parts) or None
    return out


def load_images(cxr_root: Path) -> pd.DataFrame:
    """One row per image: subject_id, study_id, image_path, split, view."""
    meta = pd.read_csv(
        cxr_root / METADATA_CSV,
        dtype=str,
        usecols=["dicom_id", "subject_id", "study_id", "ViewPosition"],
    )
    split = pd.read_csv(cxr_root / SPLIT_CSV, dtype=str)
    df = meta.merge(split, on=["dicom_id", "subject_id", "study_id"], how="outer")
    assert len(df) == len(meta) == len(split), "metadata and split disagree"
    assert df.split.isin(ROLES).all(), f"unexpected splits: {set(df.split)}"
    assert (df.groupby("subject_id").split.nunique() == 1).all(), "patient in 2 splits"
    path = ("p" + df.subject_id.str[:2] + "/p" + df.subject_id + "/s" + df.study_id) + (
        "/" + df.dicom_id + ".jpg"
    )
    return pd.DataFrame(
        {
            "subject_id": df.subject_id,
            "study_id": df.study_id,
            "image_path": path,
            "split": df.split,
            "view": df.ViewPosition,
        }
    )


def load_studies(cxr_root: Path) -> pd.DataFrame:
    """One row per study with a frontal image (see canonical_studies), with its
    findings, impression and indication (NaN where the report has none)."""
    studies = canonical_studies(load_images(cxr_root))
    sp = _parser()
    reports = cxr_root / REPORTS_DIR
    rows = [
        section((reports / report_path(p, s)).read_text(), s, sp)
        for p, s in zip(studies.subject_id, studies.study_id)
    ]
    for name in (*SECTIONS, "indication"):
        studies[name] = [r[name] for r in rows]
    return studies


def load_readers(cxr_root: Path) -> pd.Series:
    """Each study's reader by study_id: the attending radiologist on its report
    (MIMIC-CXR 2.1.0 provider list; every study has one)."""
    providers = pd.read_csv(cxr_root / PROVIDERS_CSV, dtype=str)
    return providers.set_index("study_id").attending_provider_id


def canonical_studies(images: pd.DataFrame) -> pd.DataFrame:
    """One row per study with a frontal image: the preferred one (VIEW_RANK, then
    path)."""
    images = images[images.view.isin(FRONTAL_VIEWS)]
    rank = images.view.map(VIEW_RANK)
    order = np.lexsort((images.image_path.to_numpy(), rank.to_numpy()))
    studies = images.iloc[order].drop_duplicates("study_id").sort_values("study_id")
    return studies.reset_index(drop=True)


def write_splits(cxr_root: Path, splits_dir: Path) -> None:
    """Nothing to draw: the official split sets every role."""


def classifier_roles(studies: pd.DataFrame, splits_dir: Path) -> pd.Series:
    """Each study's role from the official split: train, val (official validate:
    thresholds), test (never trained on)."""
    return studies.split.map(ROLES)
