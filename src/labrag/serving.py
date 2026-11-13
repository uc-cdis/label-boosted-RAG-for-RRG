"""Serving a generator with `vllm serve` in Docker, for the length of one job.

``vllm_server`` starts the pinned image (configs/containers.yaml) on the job's GPUs
(CUDA_VISIBLE_DEVICES, which Slurm sets to host GPU indices), waits until it answers,
and always stops it on exit, including when the job is cancelled. The container:
- sees only the job's GPUs, tensor-parallel across all of them;
- runs as the job's user (so its cache files are ours), with HOME on a node-local
  cache dir (vLLM's torch.compile and Triton caches survive between jobs);
- mounts the weights read-only and never downloads (HF_HUB_OFFLINE), with usage stats
  off;
- uses the host network on a free localhost port; its log is streamed into the job log.
"""

import contextlib
import json
import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import yaml

from labrag import paths

STARTUP_TIMEOUT_S = 3600  # first start on a node compiles kernels
STOP_TIMEOUT_S = 30


def load_config(kind: str) -> dict:
    """One of configs/{models,containers,sampling}.yaml."""
    return yaml.safe_load((paths.CONFIGS_DIR / f"{kind}.yaml").read_text())


def image_record(name: str) -> dict:
    """The pinned image, after checking this node has it (jobs never pull)."""
    entry = load_config("containers")[name]
    ref = f"{entry['image']}@{entry['digest']}"
    if shutil.which("docker") is None:
        raise SystemExit(f"docker not found on {socket.gethostname()}")
    out = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", ref],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise SystemExit(
            f"container {name}: {ref} not on {socket.gethostname()}; pre-pull it with\n"
            f"  docker pull {entry['image']}:{entry['tag']}@{entry['digest']}\n"
            f"{out.stderr.strip()}"
        )
    return {**entry, "ref": ref, "image_id": out.stdout.strip()}


def gpu_devices() -> list[str]:
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not devices:
        raise SystemExit("CUDA_VISIBLE_DEVICES is not set: request GPUs from Slurm")
    return devices.split(",")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get_json(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.loads(r.read() or b"{}")
    except OSError:
        return None


def gpu_names(devices: list[str]) -> list[str] | None:
    """Host GPU model of each device (for cost per GPU-second); None without nvidia-smi."""
    if shutil.which("nvidia-smi") is None:
        return None
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    names = dict(line.split(", ", 1) for line in out.stdout.strip().splitlines())
    return [names.get(d) for d in devices]


def metrics_text(base_url: str) -> str | None:
    """The server's Prometheus metrics (/metrics), as text."""
    try:
        with urllib.request.urlopen(f"{base_url}/metrics", timeout=30) as r:
            return r.read().decode()
    except OSError:
        return None


def counter_totals(text: str) -> dict[str, float]:
    """Every ``vllm:*_total`` counter in a /metrics text, summed over its labels."""
    totals: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.startswith("vllm:"):
            continue
        name, value = line.split("{", 1)[0].split(" ", 1)[0], line.rsplit(" ", 1)[1]
        if name.endswith("_total"):
            totals[name] = totals.get(name, 0.0) + float(value)
    return dict(sorted(totals.items()))


def _running(name: str) -> bool:
    out = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Running}}", name],
        capture_output=True,
        text=True,
    )
    return out.stdout.strip() == "true"


def raise_on_sigterm() -> None:
    """Turn Slurm's SIGTERM (scancel, time limit) into SystemExit, so cleanup runs."""

    def handler(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handler)


@contextlib.contextmanager
def vllm_server(
    served_name: str, model_dir: Path, image: dict, serve_args: list[str]
) -> Iterator[dict]:
    """Yield the server's URLs, vLLM version, devices and GPU models while it runs."""
    raise_on_sigterm()
    devices = gpu_devices()
    port = _free_port()
    name = f"labrag-{os.environ.get('SLURM_JOB_ID', 'local')}-{port}"
    cache = paths.RUN_ROOT / "vllm-cache"
    cache.mkdir(parents=True, exist_ok=True)
    cmd = [
        "docker", "run", "--rm", "--detach", "--name", name,
        "--gpus", f'"device={",".join(devices)}"',
        "--ipc=host", "--network=host",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/cache", "-e", "USER=labrag",
        "-e", "HF_HUB_OFFLINE=1", "-e", "VLLM_NO_USAGE_STATS=1", "-e", "DO_NOT_TRACK=1",
        "-v", f"{model_dir}:{model_dir}:ro",
        "-v", f"{cache}:/cache",
        image["ref"],
        str(model_dir),
        "--served-model-name", served_name,
        "--host", "127.0.0.1", "--port", str(port),
        "--tensor-parallel-size", str(len(devices)),
        *serve_args,
    ]  # fmt: skip
    print("starting:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    logs = subprocess.Popen(["docker", "logs", "--follow", name])
    base = f"http://127.0.0.1:{port}"
    try:
        start = time.monotonic()
        while _get_json(f"{base}/health") is None:
            if not _running(name):
                raise SystemExit(f"vLLM container {name} exited during startup")
            if time.monotonic() - start > STARTUP_TIMEOUT_S:
                raise SystemExit(f"vLLM not healthy after {STARTUP_TIMEOUT_S} s")
            time.sleep(5)
        version = (_get_json(f"{base}/version") or {}).get("version")
        print(
            f"vLLM {version} ready after {time.monotonic() - start:.0f} s", flush=True
        )
        yield {
            "base_url": f"{base}/v1",
            "root_url": base,
            "gpus": gpu_names(devices),
            "version": version,
            "container": name,
            "devices": devices,
            "startup_s": round(time.monotonic() - start),
        }
    finally:
        subprocess.run(
            ["docker", "stop", "--time", str(STOP_TIMEOUT_S), name],
            capture_output=True,
        )
        logs.terminate()
        print(f"stopped {name}", flush=True)
