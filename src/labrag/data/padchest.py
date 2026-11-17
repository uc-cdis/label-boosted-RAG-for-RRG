"""PadChest adapter: the author labels CSV (one row per image; labels per study) and the
released PNGs. Labels only (LABELS_ONLY): the reports are stemmed Spanish, so studies
carry no report section. PadChest-GR's official split decides which patients are held
out; the threshold patients are drawn here."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_CSV = "padchest/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz"
GR_TABLE = "padchest-gr/master_table.csv"
SOURCE_FILES = (SOURCE_CSV, GR_TABLE)
IMAGE_ROOT = "padchest/images"  # image_path is relative to this
LABELS_ONLY = True  # no report sections: stages take no --section

# Labels that are not image findings, left out of every PadChest vocabulary.
NOT_FINDINGS = ("unchanged", "exclude", "suboptimal study")
NO_FINDING = "normal"
VOCABULARIES = {"padchest60": 60}  # labeler name -> number of findings

# Undecodable (renamed .BAK locally); none is in PadChest-GR.
CORRUPT_IMAGES = frozenset(
    {
        "216840111366964012283393834152009033102258826_00-059-087.png",
        "216840111366964012283393834152009033140208626_00-059-118.png",
        "216840111366964012339356563862009068084200743_00-045-105.png",
        "216840111366964012339356563862009072111404053_00-043-192.png",
        "216840111366964012373310883942009117084022290_00-064-025.png",
        "216840111366964012373310883942009152114636712_00-102-045.png",
        "216840111366964012373310883942009170084120009_00-097-074.png",
        "216840111366964012373310883942009180082307973_00-097-011.png",
        "216840111366964012487858717522009280135853083_00-075-001.png",
        "216840111366964012558082906712009300162151055_00-078-079.png",
        "216840111366964012558082906712009301143450268_00-075-157.png",
        "216840111366964012558082906712009327122220177_00-102-064.png",
        "216840111366964012819207061112010281134410801_00-129-131.png",
        "216840111366964012819207061112010306085429121_04-020-102.png",
        "216840111366964012819207061112010307142602253_04-014-084.png",
        "216840111366964012819207061112010315104455352_04-024-184.png",
        "216840111366964012959786098432011033083840143_00-176-115.png",
        "216840111366964012989926673512011074122523403_00-163-058.png",
        "216840111366964012989926673512011083134050913_00-168-009.png",
        "216840111366964012989926673512011101154138555_00-191-086.png",
        "216840111366964012989926673512011132200139442_00-157-099.png",
        "216840111366964012989926673512011151082430686_00-157-045.png",
        "216840111366964013076187734852011178154626671_00-145-086.png",
        "216840111366964013076187734852011287092959219_00-195-171.png",
        "216840111366964013076187734852011291090445391_00-196-188.png",
        "216840111366964013590140476722013028161046120_02-015-149.png",
        "216840111366964013590140476722013043111952381_02-065-198.png",
        "216840111366964013590140476722013049100117076_02-063-097.png",
        "216840111366964013590140476722013058110301622_02-056-111.png",
        "216840111366964013649110343042013092101343018_02-075-146.png",
        "216840111366964013686042548532013208193054515_02-026-007.png",
        "216840111366964013829543166512013353113303615_02-092-190.png",
        "216840111366964013962490064942014134093945580_01-178-104.png",
    }
)

# Frontal views by preference (ties by image_path); a study with none drops out.
# AP_horizontal is AP lying down. L, COSTAL (rib views), EXCLUDE and UNK are never used.
VIEW_RANK = {"PA": 0, "AP": 1, "AP_horizontal": 2}

# PadChest-GR split of a patient, strictest first: a study whose images name several
# patients (one in the release) takes the strictest of their splits.
GR_SPLITS = ("test", "validation", "train")

# Frozen split file under paths.SPLITS_DIR (splits/README.md).
CLASSIFIER_VAL_SPLIT_FILE = "padchest_classifier_val_split.csv"
SPLIT_FILES = (CLASSIFIER_VAL_SPLIT_FILE,)  # read by classifier_roles
CLASSIFIER_VAL_FRACTION = 0.1
CLASSIFIER_VAL_SEED = 20261117


def parse_labels(value) -> tuple[str, ...] | None:
    """The Labels column's list literal as a sorted tuple of stripped, distinct labels;
    None if it holds no label (missing, or only empty strings)."""
    if pd.isna(value):
        return None
    labels = sorted({x.strip() for x in ast.literal_eval(value)} - {""})
    return tuple(labels) or None


def load_images(cxr_root: Path) -> pd.DataFrame:
    """One row per image: subject_id, study_id, image_path, view, labels, label_method,
    gr_split (the patient's PadChest-GR split, NaN outside PadChest-GR)."""
    df = pd.read_csv(
        cxr_root / SOURCE_CSV,
        dtype=str,
        usecols=[
            "ImageID",
            "StudyID",
            "PatientID",
            "Projection",
            "Labels",
            "MethodLabel",
        ],
    )
    assert df.ImageID.is_unique, "duplicate ImageID"
    gr = pd.read_csv(cxr_root / GR_TABLE, dtype=str, usecols=["PatientID", "split"])
    gr = gr.drop_duplicates()
    assert gr.PatientID.is_unique, "a PadChest-GR patient in two splits"
    assert gr.split.isin(GR_SPLITS).all(), f"unexpected splits: {set(gr.split)}"
    return pd.DataFrame(
        {
            "subject_id": df.PatientID,
            "study_id": df.StudyID,
            "image_path": df.ImageID,
            "view": df.Projection,
            "labels": df.Labels.map(parse_labels),
            "label_method": df.MethodLabel,
            "gr_split": df.PatientID.map(gr.set_index("PatientID").split),
        }
    )


def load_studies(cxr_root: Path) -> pd.DataFrame:
    """One row per study (see canonical_studies)."""
    return canonical_studies(load_images(cxr_root))


def canonical_studies(images: pd.DataFrame) -> pd.DataFrame:
    """One row per study with a usable frontal image: the preferred one (VIEW_RANK,
    then path). ``split`` is the strictest PadChest-GR split over all the study's
    images (laterals and corrupt ones included), NaN outside PadChest-GR."""
    strictness = images.gr_split.map({s: i for i, s in enumerate(GR_SPLITS)})
    strictest = strictness.groupby(images.study_id).min()
    split = strictest.map(dict(enumerate(GR_SPLITS)))
    images = images[
        ~images.image_path.isin(CORRUPT_IMAGES) & images.view.isin(VIEW_RANK)
    ]
    rank = images.view.map(VIEW_RANK)
    order = np.lexsort((images.image_path.to_numpy(), rank.to_numpy()))
    studies = images.iloc[order].drop_duplicates("study_id").sort_values("study_id")
    studies = studies.drop(columns="gr_split")
    studies.insert(2, "split", studies.study_id.map(split))
    return studies.reset_index(drop=True)


def vocabulary(
    studies: pd.DataFrame, roles: pd.Series, n: int
) -> list[tuple[str, int]]:
    """The ``n`` most common findings (NOT_FINDINGS left out) among the train-role
    studies, as (label, train count), most common first, ties by name."""
    labels = studies.labels[roles.eq("train") & studies.labels.notna()]
    counts = labels.explode().value_counts()
    counts = counts.drop(list(NOT_FINDINGS), errors="ignore").reset_index()
    counts.columns = ["label", "count"]
    counts = counts.sort_values(["count", "label"], ascending=[False, True])
    return list(counts.head(n).itertuples(index=False, name=None))


def draw_classifier_val_split(
    studies: pd.DataFrame, seed: int = CLASSIFIER_VAL_SEED
) -> pd.DataFrame:
    """The classifier-threshold patients: a fraction of the patients outside
    PadChest-GR validation and test (PadChest has no official validation split, and
    PadChest-GR validation is too small for per-label thresholds). Returns column
    subject_id, sorted."""
    held_out = set(studies.subject_id[studies.split.isin(["validation", "test"])])
    eligible = np.array(sorted(set(studies.subject_id) - held_out))
    n = round(CLASSIFIER_VAL_FRACTION * len(eligible))
    picked = np.random.default_rng(seed).choice(eligible, size=n, replace=False)
    return pd.DataFrame({"subject_id": sorted(picked)})


def write_splits(cxr_root: Path, splits_dir: Path) -> None:
    """Redraw the frozen split file in place; git diff shows any change."""
    split = draw_classifier_val_split(load_studies(cxr_root))
    split.to_csv(splits_dir / CLASSIFIER_VAL_SPLIT_FILE, index=False)
    print(f"Wrote {splits_dir / CLASSIFIER_VAL_SPLIT_FILE} ({len(split)} rows)")


def classifier_roles(studies: pd.DataFrame, splits_dir: Path) -> pd.Series:
    """Each study's role: train, val (drawn patients: thresholds), padchest_gr_val and
    padchest_gr_test (patients of PadChest-GR validation and test: never used).
    PadChest-GR train patients stay in train (docs/revision-plan.md §5)."""
    val = pd.read_csv(splits_dir / CLASSIFIER_VAL_SPLIT_FILE, dtype=str)
    role = pd.Series("train", index=studies.index)
    role[studies.subject_id.isin(val.subject_id)] = "val"
    gr = studies.split.map(
        {"validation": "padchest_gr_val", "test": "padchest_gr_test"}
    )
    assert not (role.eq("val") & gr.notna()).any(), "a drawn patient is in PadChest-GR"
    return gr.fillna(role)
