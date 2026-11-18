"""Dataset adapters, one module per dataset.

Each adapter module provides:
- ``load_studies(cxr_root)``: one row per study (study_id, subject_id, split, view,
  image_path, and one column per report section, or ``labels`` for a labels-only
  dataset);
- ``SOURCE_FILES``: the author release files it reads, relative to cxr_root;
- ``IMAGE_ROOT``: the directory ``image_path`` is relative to, under cxr_root;
- ``classifier_roles(studies, splits_dir)``: each study's role for the classifiers
  (``train``, ``val``, or a held-out role), from the frozen files ``SPLIT_FILES``;
- ``write_splits(cxr_root, splits_dir)``: redraws those files (only for splits drawn
  here, not official ones);
- ``SECTIONS``: the report sections its labels describe (PadChest: ``report``, the
  whole report);
- ``LABELS_ONLY`` (optional, default False): its labels come from the release, not
  from a labeler run on report text;
- ``FRONTAL_VIEWS``: the ``view`` values of a frontal image, for the classifiers'
  frontal-only token store.
"""

from labrag import paths
from labrag.data import chexpertplus, padchest

DATASETS = {"chexpertplus": chexpertplus, "padchest": padchest}


SECTIONS = tuple(dict.fromkeys(s for a in DATASETS.values() for s in a.SECTIONS))
JOIN = "-"  # findings-impression: one classifier over several sections' labels


def labels_only(dataset: str) -> bool:
    """True for a dataset whose labels come from the release (PadChest)."""
    return getattr(DATASETS[dataset], "LABELS_ONLY", False)


def check_section(stage: str, dataset: str, section: str) -> None:
    """``section`` must be one of the dataset's sections."""
    if section not in DATASETS[dataset].SECTIONS:
        raise SystemExit(
            f"{stage}: {dataset} sections are {list(DATASETS[dataset].SECTIONS)}"
        )


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
