"""CheXpert Plus adapter.

Reads only the author release file ``chexpertplus/df_chexpert_plus_240401.csv``
(one row per image, with the report already split into sections and the official
``train``/``valid`` split), and the released PNGs under ``chexpertplus/PNG/``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_CSV = "chexpertplus/df_chexpert_plus_240401.csv"

# Corrupt in the release; renamed to .BAK locally so we don't open it. It is
# the only image of its study (which has no Findings), so that study drops out.
CORRUPT_IMAGES = frozenset({"PNG/train/patient32368/study1/view1_frontal.png"})

# Preference rank of a study's image (lower first; ties broken by image_path), and so
# of its canonical report text. ``view`` is ap_pa on frontals and "Lateral" otherwise.
# ap_pa is undocumented in the release (probably the DICOM ViewPosition tag, where LL/RL
# mean left/right lateral) and unreliable. Its 17 LL/RL images are all labelled Frontal;
# viewed by hand (2026-11-11): 13 are a mix of frontals and lateral decubitus,
# 2 are laterals (patient09010/study1/view2, patient28939/study1/view3) and 2 are near-blank
# images (patient25114/study3/view1, patient25979/study8/view1). Mostly frontal, so
# LL/RL rank as "other frontal", after PA/AP and before laterals. Every LL/RL study also
# has a PA or AP image, so this tier never decides a study here. Mislabelled and blank
# images are left as noise; only undecodable files are excluded (CORRUPT_IMAGES).
VIEW_RANK = {"PA": 0, "AP": 1, "LL": 2, "RL": 2, "Lateral": 3}

# Non-blank section values that are section-parser debris, not report text (compared
# after stripping whitespace, case-insensitive). On 2026-11-09 these were the only
# such values in section_findings: "/" (1,145 studies), "AND"/"and" (229), "." (13),
# "," (1), ";" (1). Checked for Findings only.
JUNK_SECTION_VALUES = frozenset({",", ".", "/", ";", "and"})

N_PILOT = 200
N_PROMPT_TUNING = 200
PILOT_SEED = 20261109


def load_images(cxr_root: Path) -> pd.DataFrame:
    """One row per image: subject_id, study_id, image_path, split, view, findings, impression.

    ``image_path`` is the PNG path relative to ``cxr_root / "chexpertplus"`` (the CSV
    names .jpg files; the release images are PNGs).
    """
    df = pd.read_csv(
        cxr_root / SOURCE_CSV,
        usecols=[
            "path_to_image",
            "deid_patient_id",
            "frontal_lateral",
            "ap_pa",
            "section_findings",
            "section_impression",
            "split",
        ],
    )
    parts = df.path_to_image.str.extract(r"^(train|valid)/(patient\d+)/(study\d+)/")
    assert parts.notna().all().all(), "unexpected path_to_image format"
    assert (parts[0] == df.split).all(), "path prefix disagrees with split column"
    assert (parts[1] == df.deid_patient_id).all(), "path disagrees with patient id"
    frontal = df.frontal_lateral.eq("Frontal")
    assert (
        frontal == df.ap_pa.notna()
    ).all(), "ap_pa should be set exactly on frontals"
    view = df.ap_pa.where(frontal, "Lateral")
    assert view.isin(VIEW_RANK).all(), f"unexpected views: {set(view) - set(VIEW_RANK)}"
    return pd.DataFrame(
        {
            "subject_id": df.deid_patient_id,
            "study_id": parts[1] + "_" + parts[2],
            "image_path": "PNG/"
            + df.path_to_image.str.replace(r"\.jpg$", ".png", regex=True),
            "split": df.split,
            "view": view,
            "findings": df.section_findings,
            "impression": df.section_impression,
        }
    )


def load_studies(cxr_root: Path) -> pd.DataFrame:
    """One row per study (see canonical_studies)."""
    return canonical_studies(load_images(cxr_root))


def canonical_studies(images: pd.DataFrame) -> pd.DataFrame:
    """One row per study: its preferred image and that image's report text.

    Images of one study can carry slightly different anonymized report text
    (Stanford-AIMI/chexpert-plus#13). The study keeps the row of its preferred image:
    lowest VIEW_RANK, then by image_path. Corrupt images are skipped. Sections
    without report text (blank or parser debris, see has_text) become NaN; kept text
    is whitespace-stripped.
    Columns: subject_id, study_id, split, view, image_path, findings, impression.
    """
    images = images[~images.image_path.isin(CORRUPT_IMAGES)]
    rank = images.view.map(VIEW_RANK)
    order = np.lexsort((images.image_path.to_numpy(), rank.to_numpy()))
    studies = images.iloc[order].drop_duplicates("study_id").sort_values("study_id")
    for section in ("findings", "impression"):
        text = studies[section].fillna("").astype(str).str.strip()
        studies[section] = text.where(has_text(studies[section]))
    return studies.reset_index(drop=True)


def has_text(section: pd.Series) -> pd.Series:
    """True where a section holds report text (not blank, not parser debris)."""
    s = section.fillna("").astype(str).str.strip()
    return s.ne("") & ~s.str.lower().isin(JUNK_SECTION_VALUES)


def draw_pilot_split(images: pd.DataFrame, seed: int = PILOT_SEED) -> pd.DataFrame:
    """Draw the 400 pilot patients (docs/revision-plan.md §2, "Leakage controls").

    Eligible: official-train patients with exactly one study that has Findings.
    The first N_PILOT drawn form the "pilot" set, the rest "prompt_tuning".
    Returns columns subject_id, study_id, set, in draw order.
    """
    studies = images[(images.split == "train") & has_text(images.findings)]
    studies = studies.drop_duplicates("study_id")[["subject_id", "study_id"]]
    n = studies.groupby("subject_id").size()
    eligible = np.array(sorted(n[n == 1].index))
    rng = np.random.default_rng(seed)
    picked = rng.choice(eligible, size=N_PILOT + N_PROMPT_TUNING, replace=False)
    sets = ["pilot"] * N_PILOT + ["prompt_tuning"] * N_PROMPT_TUNING
    pilot = pd.DataFrame({"subject_id": picked, "set": sets})
    return pilot.merge(studies, on="subject_id")[["subject_id", "study_id", "set"]]
