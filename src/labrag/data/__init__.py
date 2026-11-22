"""Dataset adapters, one module per dataset.

Each adapter module provides:
- ``load_studies(cxr_root)``: one row per study with a frontal image (study_id,
  subject_id, split, view, image_path, and one column per report section, or
  ``labels`` for a labels-only dataset);
- ``load_images(cxr_root)``: one row per image (study_id, view, image_path, ...), the
  rows load_studies picks from;
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
- ``FRONTAL_VIEWS``, ``VIEW_RANK``: the ``view`` values of a frontal image, and their
  preference; ``LATERAL_VIEWS``: those of a lateral (only study_images uses laterals);
- ``CORRUPT_IMAGES`` (optional): image_paths never used;
- ``load_readers(cxr_root)`` and ``READER_FILES`` (optional): each study's reader, the
  radiologist on its report, by study_id, and the release files that give it;
- ``load_sites(cxr_root)`` and ``SITE_FILES`` (optional): the same for each study's
  site, the institution that made it.
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


GROUPS = ("reader", "site")  # study attributes an adapter may give (load_<group>s)


def has_group(dataset: str, group: str) -> bool:
    """True if the dataset's adapter gives each study's ``group`` (GROUPS)."""
    return hasattr(DATASETS[dataset], f"load_{group}s")


def group_files(dataset: str, group: str) -> tuple[str, ...]:
    """The release files that give each study's ``group``, relative to cxr_root."""
    return getattr(DATASETS[dataset], f"{group.upper()}_FILES")


def groups(dataset: str, group: str, cxr_root) -> pd.Series | None:
    """Each study's ``group`` (GROUPS) by study_id (str), or None for a dataset
    without it."""
    if not has_group(dataset, group):
        return None
    return getattr(DATASETS[dataset], f"load_{group}s")(cxr_root)


def readers(dataset: str, cxr_root) -> pd.Series | None:
    """Each study's reader by study_id (str), or None for a dataset without them."""
    return groups(dataset, "reader", cxr_root)


MAX_IMAGES = 4  # per study in study_images; 0.1% of the pooled studies have more


def study_images(dataset: str, cxr_root, ids, preferred) -> list[list[str]]:
    """Each study's images (image_paths), for studies ``ids`` with their
    ``preferred`` frontal (load_studies' image_path): that one first, then the
    study's other frontals by VIEW_RANK, then its laterals; ties by path; at most
    MAX_IMAGES. Other views and CORRUPT_IMAGES are never used."""
    adapter = DATASETS[dataset]
    images = adapter.load_images(cxr_root)
    corrupt = getattr(adapter, "CORRUPT_IMAGES", frozenset())
    frontal = images.view.isin(adapter.FRONTAL_VIEWS)
    lateral = images.view.isin(adapter.LATERAL_VIEWS)
    first = dict(zip(ids, preferred))
    rank = images.view.map(adapter.VIEW_RANK).where(
        frontal, max(adapter.VIEW_RANK.values()) + 1
    )
    rank = rank.where(images.image_path.ne(images.study_id.map(first)), -1)
    keep = (frontal | lateral) & images.study_id.isin(first)
    images = images[keep & ~images.image_path.isin(corrupt)].assign(rank=rank)
    images = images.sort_values(["study_id", "rank", "image_path"])
    by = images.groupby("study_id").image_path.agg(list)
    out = [by[i][:MAX_IMAGES] for i in ids]
    assert all(o[0] == p for o, p in zip(out, preferred)), "preferred frontal missing"
    return out


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
