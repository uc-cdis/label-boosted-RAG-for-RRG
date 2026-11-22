"""ReXGradient-160K adapter: the per-split metadata JSONs with the authors' view
positions (one record per study: patient, Indication, Findings, Impression, and
per-image paths and views) and the released PNGs. Official patient-disjoint train/valid/test split;
thresholds are set on valid."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = {"train": "train", "valid": "val", "test": "test"}  # release split -> role
SOURCE_FILES = tuple(
    f"rexgradient/metadata/{s}_metadata_view_position.json" for s in SPLITS
)
IMAGE_ROOT = "rexgradient"  # image_path is relative to this
SECTIONS = ("findings", "impression")

# Labelled frontal views by preference (ties by image_path); POSTERO_ANTERIOR is a PA
# spelled out. Unlabelled views (UNKNOWN, N/A, None) are never used, nor are laterals,
# obliques or non-chest views: 22.5% of studies, mostly LATERAL + UNKNOWN pairs, drop
# out (test keeps 7,755 of 10,000).
VIEW_RANK = {"PA": 0, "POSTERO_ANTERIOR": 0, "AP": 1}
FRONTAL_VIEWS = tuple(VIEW_RANK)

SPLIT_FILES = ()  # official splits only
SITE_FILES = SOURCE_FILES


def load_images(cxr_root: Path) -> pd.DataFrame:
    """One row per image: subject_id, study_id, image_path, split, view, findings,
    impression, indication. ``image_path`` is relative to ``cxr_root / IMAGE_ROOT`` (the JSONs give
    it relative to metadata/)."""
    rows = []
    for split, source in zip(SPLITS, SOURCE_FILES):
        with open(cxr_root / source) as f:
            records = json.load(f)
        for study_id, r in records.items():
            assert len(r["ImagePath"]) == len(r["ImageViewPosition"])
            for path, view in zip(r["ImagePath"], r["ImageViewPosition"]):
                assert path.startswith("../"), f"unexpected ImagePath {path}"
                rows.append(
                    (
                        r["PatientID"],
                        study_id,
                        path[3:],
                        split,
                        view,
                        r["Findings"],
                        r["Impression"],
                        r["Indication"],
                    )
                )
    df = pd.DataFrame(
        rows,
        columns=[
            "subject_id",
            "study_id",
            "image_path",
            "split",
            "view",
            "findings",
            "impression",
            "indication",
        ],
    )
    assert (df.groupby("subject_id").split.nunique() == 1).all(), "patient in 2 splits"
    assert (df.groupby("study_id").split.nunique() == 1).all(), "study in 2 splits"
    return df


def load_sites(cxr_root: Path) -> pd.Series:
    """Each study's site by study_id: its InstitutionName (hashed in the release;
    every study has one)."""
    sites = {}
    for source in SITE_FILES:
        with open(cxr_root / source) as f:
            sites |= {i: r["InstitutionName"] for i, r in json.load(f).items()}
    return pd.Series(sites, name="site", dtype=str)


def load_studies(cxr_root: Path) -> pd.DataFrame:
    """One row per study (see canonical_studies)."""
    return canonical_studies(load_images(cxr_root))


def canonical_studies(images: pd.DataFrame) -> pd.DataFrame:
    """One row per study with a labelled frontal image: the preferred one (VIEW_RANK,
    then path). Text is stripped; blank is NaN, and so is an indication of "None" (the
    release writes missing metadata as that string)."""
    images = images[images.view.isin(FRONTAL_VIEWS)]
    rank = images.view.map(VIEW_RANK)
    order = np.lexsort((images.image_path.to_numpy(), rank.to_numpy()))
    studies = images.iloc[order].drop_duplicates("study_id").sort_values("study_id")
    for column in (*SECTIONS, "indication"):
        text = studies[column].fillna("").astype(str).str.strip()
        blank = text.eq("") | (text.eq("None") if column == "indication" else False)
        studies[column] = text.where(~blank)
    return studies.reset_index(drop=True)


def write_splits(cxr_root: Path, splits_dir: Path) -> None:
    """Nothing to draw: the official split sets every role."""


def classifier_roles(studies: pd.DataFrame, splits_dir: Path) -> pd.Series:
    """Each study's role from the official split: train, val (official valid:
    thresholds), test (never trained on)."""
    return studies.split.map(SPLITS)
