"""Stage ``baseline``: a published report generator, run as published, on our targets.

Writes ``generations/<set>/<dataset>/findings+impression/baseline/<inputs>/<model>/
<set>/generations.jsonl.gz`` (study_id, set, report), at generate's depth (<inputs>
in the prompt's place: ``upstream`` = what its own pipeline feeds it), so evaluate and
compare take it like any condition. The targets and their references are those of the
set's no-context prompt artifact (every condition's targets are the same).

Each model is a service of baselines/compose.yaml: its upstream's code (data prep and
inference) and pins in its own image, built from this checkout and tagged with the
commit; it sees our datasets read-only at its upstream's own paths, and gets what its
upstream pipeline feeds it, not our prompts. ``prepare`` builds the model's inputs on
CPU, then ``generate`` runs one container per GPU of the job on an equal share of the
targets. Weights (and the model's published outputs) are checked (sha256) first.

CXRMate-2 publishes its own MIMIC-CXR and ReXGradient test reports: the manifest's
``parity`` counts the targets whose Findings and Impression equal them exactly,
separately for studies with more than 5 images (current + prior), of which its
processor samples 5 at random: those can't match.
"""

import concurrent.futures
import fcntl
import gzip
import json
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import yaml

from labrag import artifacts, paths, serving, weights
from labrag.data import JOINT

NAME = "baseline"
COMPOSE_FILE = paths.REPO_ROOT / "baselines" / "compose.yaml"
TARGETS = (
    "prompts/{set}/{dataset}/{section}/no-context/"
    "joint-no-context-negatives-indication-prior-changes-image"
)
MODELS = {
    "cxrmate-2": {
        "prepare": "cxrmate-2-prepare",  # builds the image
        "generate": "cxrmate-2",
        "upstream": "external/cxrmate-2",
        "weights": "baseline.cxrmate-2",
        "inputs": "upstream",
        # our study id -> upstream's: ReXGradient's is the StudyInstanceUid
        "upstream_id": {
            "mimic": lambda s: s,
            "rexgradient": lambda s: s.split("_s", 1)[1],
        },
        # its processor samples this many of a study's images (processor_config.json)
        "max_images": 5,
        # the outputs upstream published, in the weights dir
        "published": {
            "mimic": "cxrmate2_generated_reports_mimic_cxr_test.csv",
            "rexgradient": "cxrmate2_generated_reports_rexgradient_test.csv",
        },
    },
}


def run(model: str, dataset: str, target_set: str, overwrite: bool = False):
    spec = MODELS[model]
    if dataset not in spec["upstream_id"]:
        raise SystemExit(f"baseline {model}: no {dataset} pipeline upstream")
    prompt_rel = TARGETS.format(set=target_set, dataset=dataset, section=JOINT)
    rel = (
        f"generations/{target_set}/{dataset}/{JOINT}/{NAME}/{spec['inputs']}/{model}/"
        f"{target_set}"
    )
    config = {
        "stage": NAME,
        "model": model,
        "inputs": spec["inputs"],
        "dataset": dataset,
        "set": target_set,
        "prompts": prompt_rel,  # the targets and references only
        "image": image_source(spec),
    }
    if artifacts.is_built(rel, config, overwrite):
        return paths.SHARED_ROOT / rel

    prompt_dir = paths.SHARED_ROOT / prompt_rel
    inputs = {"prompts": artifacts.artifact_record(prompt_dir)}
    with gzip.open(prompt_dir / "prompts.jsonl.gz", "rt") as f:
        targets = [
            r["study_id"] for line in f if (r := json.loads(line))["set"] == target_set
        ]
    if not targets:
        raise SystemExit(f"baseline: no targets in set {target_set!r}")
    to_upstream = spec["upstream_id"][dataset]
    upstream_ids = [to_upstream(t) for t in targets]
    if len(set(upstream_ids)) != len(targets):
        raise SystemExit("baseline: two targets share an upstream id")
    model_dir, inputs["weights"] = weights.resolve(spec["weights"])
    devices = serving.gpu_devices()

    def build(out: Path) -> dict:
        image = build_image(spec)
        serving.raise_on_sigterm()
        ncpu = int(
            os.environ.get("SLURM_CPUS_PER_TASK") or len(os.sched_getaffinity(0))
        )
        # node-local disk: the prepared inputs are GBs (MIMIC-CXR test ~16 GB)
        paths.RUN_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=paths.RUN_ROOT, prefix="labrag-baseline-"
        ) as work:
            work = Path(work)
            try:
                run_service(
                    spec["prepare"],
                    image,
                    work,
                    ncpu,
                    "prepare",
                    dataset,
                    str(work / "db"),
                )
                shards = [upstream_ids[i :: len(devices)] for i in range(len(devices))]
                with concurrent.futures.ThreadPoolExecutor(len(devices)) as pool:
                    futures = [
                        pool.submit(
                            generate_shard,
                            spec,
                            image,
                            work,
                            dataset,
                            i,
                            d,
                            shard,
                            max(1, ncpu // len(devices)),
                        )
                        for i, (d, shard) in enumerate(zip(devices, shards))
                    ]
                    parts = [f.result() for f in futures]
            finally:
                compose("down", "--timeout", "30", tag=image["tag"], check=False)
        generated = pd.concat([p for p, _ in parts]).set_index("study_id")
        generated = generated.loc[upstream_ids]
        records = [
            {
                "study_id": t,
                "set": target_set,
                "report": f"Findings: {g.findings} Impression: {g.impression}",
            }
            for t, g in zip(targets, generated.itertuples())
        ]
        with open(out / "generations.jsonl.gz", "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as f:
                for r in records:
                    f.write((json.dumps(r, ensure_ascii=False) + "\n").encode())
        extra = {
            "n_generations": len(records),
            "empty_sections": {
                s: int((generated[s] == "").sum()) for s in ("findings", "impression")
            },
            "image": image,
            "timing": [t for _, t in parts],
        }
        if name := spec.get("published", {}).get(dataset):
            extra["parity"] = parity(generated, model_dir / name, spec["max_images"])
            print("parity with the published outputs:", json.dumps(extra["parity"]))
        return extra

    return artifacts.publish(rel, config, inputs, build, overwrite)


def generate_shard(
    spec: dict,
    image: dict,
    work: Path,
    dataset: str,
    i: int,
    device: str,
    upstream_ids: list[str],
    threads: int,
) -> tuple[pd.DataFrame, dict]:
    """One GPU's share: its out.csv (texts as written, "" stays "") and timing."""
    targets_txt = work / f"targets-{i}.txt"
    out_csv = work / f"out-{i}.csv"
    targets_txt.write_text("".join(f"{u}\n" for u in upstream_ids))
    run_service(
        spec["generate"],
        image,
        work,
        threads,
        "generate",
        dataset,
        str(work / "db"),
        str(targets_txt),
        str(out_csv),
        device=device,
        prefix=f"gpu {device}",
    )
    out = pd.read_csv(out_csv, dtype=str, keep_default_na=False)
    if list(out["study_id"]) != upstream_ids:
        raise SystemExit(f"{out_csv}: not one report per target, in order")
    timing = json.loads(out_csv.with_suffix(".json").read_text())
    return out, {"gpu": device, **timing}


def parity(generated: pd.DataFrame, published_csv: Path, max_images: int) -> dict:
    """How many targets' sections equal the published ones (all are published), for
    the studies whose images are all used and for those sampled from."""
    published = pd.read_csv(published_csv, dtype=str, keep_default_na=False)
    published = published.set_index("study_id").loc[generated.index]
    same = {s: generated[s] == published[s] for s in ("findings", "impression")}
    same["both"] = same["findings"] & same["impression"]
    sampled = generated["n_images"].astype(int) > max_images
    return {
        "published": published_csv.name,
        **{
            group: {
                "n": int(rows.sum()),
                **{f"{s}_equal": int(same[s][rows].sum()) for s in same},
            }
            for group, rows in (("all_images", ~sampled), ("sampled", sampled))
        },
    }


def run_service(
    service: str,
    image: dict,
    work: Path,
    threads: int,
    *args: str,
    device: str | None = None,
    prefix: str | None = None,
) -> None:
    """One run of a model's container, ``work`` mounted at the same path."""
    compose(
        "run",
        "--rm",
        "--no-TTY",
        "--pull",
        "never",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--volume",
        f"{work}:{work}",
        "--env",
        f"OMP_NUM_THREADS={threads}",
        service,
        *args,
        tag=image["tag"],
        device=device,
        prefix=prefix,
    )


def compose_service(name: str) -> dict:
    return yaml.safe_load(COMPOSE_FILE.read_text())["services"][name]


def image_source(spec: dict) -> dict:
    """The git ids the model's image is built from (and the compose file it runs by:
    its mounts and environment)."""
    context = COMPOSE_FILE.parent / compose_service(spec["prepare"])["build"]["context"]
    return {
        "service": spec["generate"],
        "upstream_commit": _git("rev-parse", f"HEAD:{spec['upstream']}"),
        "definition_tree": _git(
            "rev-parse", f"HEAD:{context.resolve().relative_to(paths.REPO_ROOT)}"
        ),
        "compose_blob": _git(
            "rev-parse", f"HEAD:{COMPOSE_FILE.relative_to(paths.REPO_ROOT)}"
        ),
    }


def build_image(spec: dict) -> dict:
    """Build the model's image from this checkout; its tag and image id."""
    commit = artifacts.git_commit()
    # A dirty dev tree builds what's on disk, which no commit describes.
    tag = "dev" if commit["dirty"] else commit["sha"]
    ref = f"{compose_service(spec['prepare'])['image'].split(':')[0]}:{tag}"
    # one build per node at a time (evaluate.build_image's lock)
    paths.RUN_ROOT.mkdir(parents=True, exist_ok=True)
    with open(paths.RUN_ROOT / "docker-build.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        compose("build", spec["prepare"], tag=tag)
        built = serving.image_id(ref)
    if built is None:
        raise SystemExit(f"{ref}: not found after the build")
    return {"ref": ref, "tag": tag, "image_id": built}


def compose(
    *args: str,
    tag: str,
    check: bool = True,
    device: str | None = None,
    prefix: str | None = None,
) -> None:
    """docker compose as this job's own project, on one host GPU index (``device``);
    output lines get ``[prefix] ``."""
    env = {
        **os.environ,
        "LABRAG_COMMIT": tag,
        "LABRAG_MODELS_ROOT": str(paths.MODELS_ROOT),
        "LABRAG_CXR_ROOT": str(paths.CXR_ROOT),
    }
    if device is not None:
        env["CUDA_VISIBLE_DEVICES"] = device
    project = f"labrag-baselines-{os.environ.get('SLURM_JOB_ID') or os.getpid()}"
    cmd = ["docker", "compose", "-f", str(COMPOSE_FILE), "-p", project, *args]
    print(f"[{prefix}] +" if prefix else "+", " ".join(cmd), flush=True)
    with subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ) as proc:
        for line in proc.stdout:
            print(f"[{prefix}] {line}" if prefix else line, end="", flush=True)
    if check and proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(paths.REPO_ROOT), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
