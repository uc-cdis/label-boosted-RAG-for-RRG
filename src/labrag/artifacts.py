"""Artifacts: human-readable output directories under paths.SHARED_ROOT.

Each artifact is one directory, e.g. ``labels/chexpertplus/findings/chexbert14/``,
holding a few output files plus ``manifest.json``: how it was made (commit, full config,
inputs with their sha256, node, times, package versions) and the sha256 of each output.

Publishing rules:
- the directory is built on local disk, then copied to shared storage once and renamed
  into place, so readers never see a half-written artifact;
- same path and same config: already done, skipped (even at a newer commit);
- same path, different config: refused. Pick a new name, or pass overwrite=True;
- same path and config, but built from inputs that have changed since (stale_inputs):
  refused as stale. Rebuild it with overwrite=True. A stage also refuses to read a
  stale artifact (artifact_record), so staleness never passes downstream.
"""

import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path

from labrag import paths

MANIFEST = "manifest.json"

# Package versions recorded in every manifest (the full set is in uv.lock at the commit).
VERSIONED_PACKAGES = ("numpy", "pandas", "scikit-learn", "torch", "transformers")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 23):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: Path) -> dict:
    """Provenance record of an input file."""
    return {"path": str(path), "sha256": sha256_file(path)}


def git_commit(repo: Path = paths.REPO_ROOT) -> dict:
    """The code version: the runner's pinned commit, or the dev checkout's HEAD."""
    if sha := os.environ.get("LABRAG_COMMIT"):
        return {"sha": sha, "dirty": False}  # the runner refuses dirty trees
    git = ["git", "-C", str(repo)]
    sha = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True)
    status = subprocess.run(
        [*git, "status", "--porcelain"], capture_output=True, text=True
    )
    return {"sha": sha.stdout.strip(), "dirty": bool(status.stdout.strip())}


def read_manifest(artifact_dir: Path) -> dict | None:
    path = artifact_dir / MANIFEST
    return json.loads(path.read_text()) if path.is_file() else None


def artifact_record(artifact_dir: Path) -> dict:
    """Provenance record of an input artifact, after checking its files' sha256 and
    that it isn't stale. Stages call this on every artifact they read."""
    manifest = read_manifest(artifact_dir)
    if manifest is None:
        raise SystemExit(f"MISSING: {artifact_dir} has no {MANIFEST}; build it first")
    for name, sha in manifest["outputs"].items():
        if sha256_file(artifact_dir / name) != sha:
            raise SystemExit(f"CHANGED: {artifact_dir / name} differs from {MANIFEST}")
    if reasons := stale_inputs(manifest):
        raise SystemExit(
            f"STALE INPUT: {artifact_dir} was built from inputs that changed:\n"
            + _bullets(reasons)
            + "\nRebuild it first (--overwrite)."
        )
    return {
        "path": str(artifact_dir),
        "commit": manifest["commit"]["sha"],
        "outputs": manifest["outputs"],
    }


def stale_inputs(manifest: dict) -> list[str]:
    """Why an artifact's recorded inputs no longer hold (empty: they all do): an
    upstream artifact's outputs (sha256) changed, or a model's files are no longer
    pinned in configs/weights.yaml. Reads manifests and config only, no hashing."""
    from labrag import weights  # imported here: weights imports this module

    pins = None
    reasons = []
    for kind, record in _input_records(manifest["inputs"]):
        if kind == "artifact":
            upstream = read_manifest(Path(record["path"]))
            if upstream is None:
                reasons.append(f"{record['path']} is gone")
            elif upstream["outputs"] != record["outputs"]:
                reasons.append(f"{record['path']} has other outputs now")
        else:
            pins = pins or weights.load_config()
            files = {f: v["sha256"] for f, v in record["files"].items()}
            same_repo = [
                e["files"] for e in pins.values() if e["repo"] == record["repo"]
            ]
            if not same_repo:
                reasons.append(f"weights {record['repo']}: no longer pinned")
            elif files not in same_repo:
                reasons.append(f"weights {record['repo']}: pinned to other files now")
    return reasons


def _input_records(inputs):
    """(kind, record) of every upstream artifact ("artifact") and model ("weights")
    in a manifest's inputs."""
    if isinstance(inputs, dict):
        if "path" in inputs and "outputs" in inputs:
            yield "artifact", inputs
        elif "repo" in inputs and "files" in inputs:
            yield "weights", inputs
        else:
            for v in inputs.values():
                yield from _input_records(v)
    elif isinstance(inputs, list):
        for v in inputs:
            yield from _input_records(v)


def stale_tree(root: Path | None = None) -> tuple[int, dict[Path, list[str]]]:
    """(artifacts seen, {stale dir: reasons}) under ``root``: stale by its own inputs
    or through any upstream artifact."""
    manifests = {}
    for dirpath, dirnames, filenames in os.walk(root or paths.SHARED_ROOT):
        # hidden: publish's .incoming-/.old- dirs; an artifact holds no artifacts
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        if MANIFEST in filenames:
            manifests[Path(dirpath)] = read_manifest(Path(dirpath))
            dirnames[:] = []
    stale = {d: r for d, m in manifests.items() if (r := stale_inputs(m))}
    upstreams = {
        d: [Path(r["path"]) for k, r in _input_records(m["inputs"]) if k == "artifact"]
        for d, m in manifests.items()
    }
    while True:
        new = {
            d: [f"{u} is stale" for u in ups if u in stale]
            for d, ups in upstreams.items()
            if d not in stale and any(u in stale for u in ups)
        }
        if not new:
            return len(manifests), stale
        stale |= new


def _bullets(lines: list[str]) -> str:
    return "\n".join(f"  - {line}" for line in lines)


def _normalize(config: dict) -> dict:
    """The config as it reads back from JSON, so in-memory and stored configs compare."""
    return json.loads(json.dumps(config))


def now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def is_built(
    relpath: str, config: dict, overwrite: bool = False, shared_root: Path | None = None
) -> bool:
    """True (and prints SKIP) if publish would skip the artifact; SystemExit where it
    would refuse. Lets a stage skip setup (e.g. a model server) when all is built."""
    final = (shared_root or paths.SHARED_ROOT) / relpath
    config = _normalize(config)
    existing = read_manifest(final)
    if existing is not None and not overwrite:
        if existing["config"] == config:
            if reasons := stale_inputs(existing):
                raise SystemExit(
                    f"STALE: {final} was built from inputs that changed:\n"
                    + _bullets(reasons)
                    + "\nUse --overwrite to rebuild it."
                )
            print(f"SKIP: {final} already built with this config")
            return True
        raise SystemExit(
            f"REFUSED: {final} exists with a different config.\n"
            f"  existing: {json.dumps(existing['config'], sort_keys=True)}\n"
            f"  new:      {json.dumps(config, sort_keys=True)}\n"
            "Use a new name, or --overwrite to replace it."
        )
    if final.exists() and existing is None:
        raise SystemExit(
            f"REFUSED: {final} exists but has no {MANIFEST}; not touching it"
        )
    return False


def publish(
    relpath: str,
    config: dict,
    inputs: dict,
    build: Callable[[Path], dict | None],
    overwrite: bool = False,
    shared_root: Path | None = None,
    started: str | None = None,
) -> Path:
    """Build the artifact at <shared_root>/<relpath>, unless an identical one exists.

    ``build(out_dir)`` writes the outputs into a local staging dir and may return extra
    manifest fields. ``inputs``: name -> provenance record. ``started`` (``now()``):
    when the work began, if before build. Returns the artifact directory.
    """
    final = (shared_root or paths.SHARED_ROOT) / relpath
    config = _normalize(config)
    if is_built(relpath, config, overwrite, shared_root):
        return final

    started = started or now()
    with tempfile.TemporaryDirectory(prefix="labrag-") as stage:
        stage = Path(stage)
        extra = build(stage) or {}
        outputs = {p.name: sha256_file(p) for p in sorted(stage.iterdir())}
        manifest = {
            "artifact": relpath,
            "config": config,
            "inputs": inputs,
            "outputs": outputs,
            **extra,
            "commit": git_commit(),
            "node": socket.gethostname(),
            "started": started,
            "finished": now(),
            "python": platform.python_version(),
            "packages": {p: importlib.metadata.version(p) for p in VERSIONED_PACKAGES},
        }
        (stage / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")

        # One copy to shared storage under a hidden name, then rename(s) into place.
        final.parent.mkdir(parents=True, exist_ok=True)
        tag = uuid.uuid4().hex[:8]
        incoming = final.parent / f".{final.name}.incoming-{tag}"
        shutil.copytree(stage, incoming)
        # mkdtemp's 0700 survives copytree; use the umask's default instead.
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(incoming, 0o777 & ~umask)
    if final.exists():
        old = final.parent / f".{final.name}.old-{tag}"
        os.rename(final, old)
        os.rename(incoming, final)
        shutil.rmtree(old)
    else:
        os.rename(incoming, final)
    print(f"Wrote {final}")
    return final
