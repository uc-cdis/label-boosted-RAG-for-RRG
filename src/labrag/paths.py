"""Filesystem roots. Override per host with environment variables."""

import os
from pathlib import Path

# Read-only dataset releases. Only author-released source files are read from here;
# locally derived files that live alongside them (e.g. cxr/derived/) are never used.
CXR_ROOT = Path(os.environ.get("LABRAG_CXR_ROOT", "/opt/gpudata/cxr"))

# All pipeline outputs.
OUT_ROOT = Path(os.environ.get("LABRAG_OUT_ROOT", "/home/songs1/labrag-outputs"))

# Committed, frozen split files (this repo's splits/ directory).
SPLITS_DIR = Path(__file__).resolve().parents[2] / "splits"
