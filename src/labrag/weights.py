"""Locate and verify pre-pulled model weights (configs/weights.yaml)."""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from labrag import paths
from labrag.artifacts import sha256_file

HASH_WORKERS = min(16, int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))


def load_config() -> dict:
    return yaml.safe_load((paths.CONFIGS_DIR / "weights.yaml").read_text())


def revision(model_dir: Path, filename: str) -> str | None:
    """The commit `hf download --local-dir` recorded for a file (None if not recorded)."""
    meta = model_dir / ".cache" / "huggingface" / "download" / f"{filename}.metadata"
    return meta.read_text().splitlines()[0].strip() if meta.is_file() else None


def resolve(name: str, models_root: Path | None = None) -> tuple[Path, dict]:
    """Check one entry of configs/weights.yaml and return (model dir, provenance record).

    Fails if any listed file is missing or its sha256 differs from the config. The record
    (repo, dir, and per file its sha256 and revision) goes into artifact manifests.
    """
    entry = load_config()[name]
    model_dir = (models_root or paths.MODELS_ROOT) / entry["repo"]
    for filename in entry["files"]:
        if not (model_dir / filename).is_file():
            raise SystemExit(
                f"weights {name}: missing {model_dir / filename} "
                "(pre-pull it; jobs never download)"
            )
    # hashlib releases the GIL, so threads hash files in parallel (55 GB in ~20 s).
    with ThreadPoolExecutor(max_workers=HASH_WORKERS) as pool:
        hashes = dict(
            zip(
                entry["files"],
                pool.map(sha256_file, [model_dir / f for f in entry["files"]]),
            )
        )
    files = {}
    for filename, expected in entry["files"].items():
        path = model_dir / filename
        actual = hashes[filename]
        if actual != expected:
            raise SystemExit(
                f"weights {name}: sha256 of {path} is {actual}, expected {expected}"
            )
        files[filename] = {"sha256": actual, "revision": revision(model_dir, filename)}
    return model_dir, {"repo": entry["repo"], "dir": str(model_dir), "files": files}
