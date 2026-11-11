"""Locate and verify pre-pulled model weights (configs/weights.yaml)."""

from pathlib import Path

import yaml

from labrag import paths
from labrag.artifacts import sha256_file


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
    files = {}
    for filename, expected in entry["files"].items():
        path = model_dir / filename
        if not path.is_file():
            raise SystemExit(
                f"weights {name}: missing {path} (pre-pull it; jobs never download)"
            )
        actual = sha256_file(path)
        if actual != expected:
            raise SystemExit(
                f"weights {name}: sha256 of {path} is {actual}, expected {expected}"
            )
        files[filename] = {"sha256": actual, "revision": revision(model_dir, filename)}
    return model_dir, {"repo": entry["repo"], "dir": str(model_dir), "files": files}
