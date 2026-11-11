"""Filesystem roots. Override per host with environment variables."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Read-only dataset releases. Only author-released source files are read from here;
# locally derived files that live alongside them (e.g. cxr/derived/) are never used.
CXR_ROOT = Path(os.environ.get("LABRAG_CXR_ROOT", "/opt/gpudata/cxr"))

# Model weights, pre-pulled on every node (configs/weights.yaml). Never downloaded by jobs.
MODELS_ROOT = Path(os.environ.get("LABRAG_MODELS_ROOT", "/opt/gpudata/models"))

# Shared NFS storage visible from every node: the artifacts later stages read, and the
# results (docs/revision-plan.md §8). Jobs build on local disk and publish here once.
SHARED_ROOT = Path(
    os.environ.get("LABRAG_SHARED_ROOT", "/opt/gpu_working/steven/labrag-shared")
)

# All pipeline outputs.
OUT_ROOT = Path(os.environ.get("LABRAG_OUT_ROOT", "/home/songs1/labrag-outputs"))

# Committed, frozen split files (this repo's splits/ directory).
SPLITS_DIR = REPO_ROOT / "splits"

# Committed configs: one YAML per kind (weights, models, sampling, ...).
CONFIGS_DIR = REPO_ROOT / "configs"

# Committed reference tables transcribed by hand (e.g. the PadChest location tree).
RESOURCES_DIR = REPO_ROOT / "resources"
