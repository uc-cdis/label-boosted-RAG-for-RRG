"""Dataset adapters, one module per dataset.

Each adapter module provides:
- ``load_studies(cxr_root)``: one row per study (study_id, subject_id, split, view,
  image_path, and one column per report section);
- ``SOURCE_CSV``: the author release file it reads, relative to cxr_root;
- ``IMAGE_ROOT``: the directory ``image_path`` is relative to, under cxr_root.
"""

from labrag.data import chexpertplus

DATASETS = {"chexpertplus": chexpertplus}
