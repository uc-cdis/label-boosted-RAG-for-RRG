"""CheXpert Plus adapter.

Reads only the author release file ``chexpertplus/df_chexpert_plus_240401.csv``
(one row per image, with the report already split into sections and the official
``train``/``valid`` split).
"""

from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_CSV = "chexpertplus/df_chexpert_plus_240401.csv"

# Non-blank section values that are section-parser debris, not report text (compared
# after stripping whitespace, case-insensitive). On 2026-11-09 these were the only
# such values in section_findings: "/" (1,145 studies), "AND"/"and" (229), "." (13),
# "," (1), ";" (1). Checked for Findings only.
JUNK_SECTION_VALUES = frozenset({",", ".", "/", ";", "and"})

N_PILOT = 200
N_PROMPT_TUNING = 200
PILOT_SEED = 20261109


def load_images(cxr_root: Path) -> pd.DataFrame:
    """One row per image: subject_id, study_id, image_path, split, findings, impression."""
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
    return pd.DataFrame(
        {
            "subject_id": df.deid_patient_id,
            "study_id": parts[1] + "_" + parts[2],
            "image_path": df.path_to_image,
            "split": df.split,
            "frontal_lateral": df.frontal_lateral,
            "ap_pa": df.ap_pa,
            "findings": df.section_findings,
            "impression": df.section_impression,
        }
    )


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
