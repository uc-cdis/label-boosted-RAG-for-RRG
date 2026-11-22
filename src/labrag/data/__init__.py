"""Dataset adapters, one module per dataset.

Each adapter module provides:
- ``load_studies(cxr_root)``: one row per study with a frontal image (study_id,
  subject_id, split, view, image_path, and one column per report section, or
  ``labels`` for a labels-only dataset); lateral images are never used;
- ``SOURCE_FILES``: the author release files (or directories of them) it reads,
  relative to cxr_root;
- ``IMAGE_ROOT``: the directory ``image_path`` is relative to, under cxr_root;
- ``classifier_roles(studies, splits_dir)``: each study's role for the classifiers
  (``train``, ``val``, or a held-out role), from the frozen files ``SPLIT_FILES``;
- ``write_splits(cxr_root, splits_dir)``: redraws those files (only for splits drawn
  here, not official ones);
- ``SECTIONS``: the report sections its labels describe (PadChest: ``report``, the
  whole report);
- ``LABELS_ONLY`` (optional, default False): its labels come from the release, not
  from a labeler run on report text;
- ``FRONTAL_VIEWS``: the ``view`` values of a frontal image;
- ``load_readers(cxr_root)`` and ``READER_FILES`` (optional): each study's reader, the
  radiologist on its report, by study_id, and the release files that give it.
"""

import pandas as pd

from labrag import paths
from labrag.data import chexpertplus, mimic, padchest, rexgradient

DATASETS = {
    "chexpertplus": chexpertplus,
    "mimic": mimic,
    "padchest": padchest,
    "rexgradient": rexgradient,
}


JOIN = "-"  # findings-impression: one classifier over several sections' labels
# Findings+Impression generated, labelled and scored as one text: a section of every
# dataset with both. Not JOIN, which names a classifier's readouts, one per section.
JOINT = "findings+impression"
JOINT_PARTS = ("findings", "impression")
SECTIONS = (
    *dict.fromkeys(s for a in DATASETS.values() for s in a.SECTIONS),
    JOINT,
)


def labels_only(dataset: str) -> bool:
    """True for a dataset whose labels come from the release (PadChest)."""
    return getattr(DATASETS[dataset], "LABELS_ONLY", False)


def readers(dataset: str, cxr_root) -> pd.Series | None:
    """Each study's reader by study_id (str), or None for a dataset without them."""
    adapter = DATASETS[dataset]
    if not hasattr(adapter, "load_readers"):
        return None
    return adapter.load_readers(cxr_root)


def has_joint(dataset: str) -> bool:
    """True for a dataset with both JOINT_PARTS."""
    return set(JOINT_PARTS) <= set(DATASETS[dataset].SECTIONS)


def check_section(stage: str, dataset: str, section: str, joint=False) -> None:
    """``section`` must be one of the dataset's sections, or JOINT if ``joint`` (a
    stage that handles it) and the dataset has both its parts."""
    known = list(DATASETS[dataset].SECTIONS)
    if joint and has_joint(dataset):
        known.append(JOINT)
    if section not in known:
        raise SystemExit(f"{stage}: {dataset} sections are {known}")


def add_joint(studies: pd.DataFrame) -> pd.DataFrame:
    """``studies`` with a JOINT column: the parts' texts, stripped, joined by a space
    (NaN unless the study has both)."""
    parts = [studies[p].str.strip() for p in JOINT_PARTS]
    return studies.assign(**{JOINT: parts[0] + " " + parts[1]})


def split_sections(stage: str, dataset: str, section: str) -> list[str]:
    """The sections of a (possibly joined) section name, in the dataset's order."""
    parts = section.split(JOIN)
    order = list(DATASETS[dataset].SECTIONS)
    for s in parts:
        check_section(stage, dataset, s)
    if parts != sorted(set(parts), key=order.index):
        raise SystemExit(f"{stage}: join sections once each, in the order {order}")
    return parts


def write_splits(dataset: str) -> None:
    """Redraw a dataset's frozen split files in place under paths.SPLITS_DIR."""
    DATASETS[dataset].write_splits(paths.CXR_ROOT, paths.SPLITS_DIR)
