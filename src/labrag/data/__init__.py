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
- ``LABELS_ONLY`` (optional, default False): no report sections; its labels come from
  the release, and stages take no ``--section`` for it.
- ``FRONTAL_VIEWS`` (datasets with sections): the ``view`` values of a frontal image,
  for the classifiers' frontal-only token store.
"""

from labrag import paths
from labrag.data import chexpertplus, padchest

DATASETS = {"chexpertplus": chexpertplus, "padchest": padchest}


def labels_only(dataset: str) -> bool:
    """True for a dataset without report sections (PadChest)."""
    return getattr(DATASETS[dataset], "LABELS_ONLY", False)


def check_section(stage: str, dataset: str, section: str | None) -> None:
    """A labels-only dataset takes no --section; every other dataset needs one."""
    if labels_only(dataset) and section is not None:
        raise SystemExit(f"{stage}: {dataset} has no report sections; drop --section")
    if not labels_only(dataset) and section is None:
        raise SystemExit(f"{stage}: {dataset} needs --section")


def write_splits(dataset: str) -> None:
    """Redraw a dataset's frozen split files in place under paths.SPLITS_DIR."""
    DATASETS[dataset].write_splits(paths.CXR_ROOT, paths.SPLITS_DIR)
