"""CheXpert Plus adapter: the author release CSV (one row per image, sections and the
official split) and the released PNGs."""

from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_CSV = "chexpertplus/df_chexpert_plus_240401.csv"
SOURCE_FILES = (SOURCE_CSV,)
IMAGE_ROOT = "chexpertplus"  # image_path is relative to this

# Corrupt (renamed .BAK locally); its study's only image, so the study drops out.
CORRUPT_IMAGES = frozenset({"PNG/train/patient32368/study1/view1_frontal.png"})

# Preference rank of a study's image (ties by image_path), which also picks its report
# text. ``view`` is ap_pa on frontals, else "Lateral". ap_pa is undocumented; its 17
# LL/RL images are labelled Frontal and mostly are (by eye), so they rank as "other
# frontal". Every LL/RL study also has PA or AP, so the tier never decides here.
VIEW_RANK = {"PA": 0, "AP": 1, "LL": 2, "RL": 2, "Lateral": 3}
FRONTAL_VIEWS = ("PA", "AP", "LL", "RL")  # labelled Frontal in the release

# Section-parser debris, not report text (stripped, case-insensitive; the only such
# values in section_findings, 2026-11-09).
JUNK_SECTION_VALUES = frozenset({",", ".", "/", ";", "and"})

N_PILOT = 200
N_PROMPT_TUNING = 200
PILOT_SEED = 20261109

# Frozen split files under paths.SPLITS_DIR (splits/README.md).
PILOT_SPLIT_FILE = "chexpertplus_pilot_split.csv"
CLASSIFIER_VAL_SPLIT_FILE = "chexpertplus_classifier_val_split.csv"
SPLIT_FILES = (PILOT_SPLIT_FILE, CLASSIFIER_VAL_SPLIT_FILE)  # read by classifier_roles
CLASSIFIER_VAL_FRACTION = 0.1
CLASSIFIER_VAL_SEED = 20261111


def load_images(cxr_root: Path) -> pd.DataFrame:
    """One row per image: subject_id, study_id, image_path, split, view, findings, impression.

    ``image_path`` is the PNG path relative to ``cxr_root / IMAGE_ROOT`` (the CSV
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
    """One row per study: its preferred image (VIEW_RANK, then path) and that image's
    report text, since a study's images can carry different anonymized text
    (Stanford-AIMI/chexpert-plus#13). Text is stripped; no text (has_text) is NaN."""
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


def draw_classifier_val_split(
    images: pd.DataFrame, pilot: pd.DataFrame, seed: int = CLASSIFIER_VAL_SEED
) -> pd.DataFrame:
    """The classifier-threshold patients: a fraction of non-pilot official-train
    patients (official valid has only 61 Findings studies). Per patient, so the same for
    every section. Returns column subject_id, sorted."""
    train = images.loc[images.split == "train", "subject_id"]
    eligible = np.array(sorted(set(train) - set(pilot.subject_id)))
    n = round(CLASSIFIER_VAL_FRACTION * len(eligible))
    picked = np.random.default_rng(seed).choice(eligible, size=n, replace=False)
    return pd.DataFrame({"subject_id": sorted(picked)})


def write_splits(cxr_root: Path, splits_dir: Path) -> None:
    """Redraw every frozen split file in place (the validation draw excludes the pilot
    patients, so the pilot comes first); git diff shows any change."""
    images = load_images(cxr_root)
    pilot = draw_pilot_split(images)
    drawn = {
        PILOT_SPLIT_FILE: pilot,
        CLASSIFIER_VAL_SPLIT_FILE: draw_classifier_val_split(images, pilot),
    }
    for name, split in drawn.items():
        split.to_csv(splits_dir / name, index=False)
        print(f"Wrote {splits_dir / name} ({len(split)} rows)")


def classifier_roles(studies: pd.DataFrame, splits_dir: Path) -> pd.Series:
    """Each study's role: train, val (thresholds), pilot / prompt_tuning (never trained
    on), official_valid (extra check only)."""
    pilot = pd.read_csv(splits_dir / PILOT_SPLIT_FILE)
    val = pd.read_csv(splits_dir / CLASSIFIER_VAL_SPLIT_FILE)
    pilot_set = studies.subject_id.map(pilot.set_index("subject_id").set)
    role = pd.Series("train", index=studies.index)
    role[studies.subject_id.isin(val.subject_id)] = "val"
    role[pilot_set.notna()] = pilot_set[pilot_set.notna()]
    role[studies.split == "valid"] = "official_valid"
    assert not (studies.subject_id.isin(val.subject_id) & pilot_set.notna()).any()
    return role
