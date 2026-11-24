"""Shared test setup."""

import os

# Tiny CPU models: a thread per core (and per DDP rank / loader worker) only contends;
# the fine-tune tests ran ~4x slower with the default on a many-core host. Set before
# torch is imported so spawned ranks and workers inherit it; overrides job.sh's cap.
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[var] = "1"
