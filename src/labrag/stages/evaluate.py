"""Stage ``evaluate``: per-study scores of generated reports, by official metric code.

Reads a generations artifact (stage generate) and the prompt artifact it was generated
from (for the reference text), and writes
``scores/<generations path>/<metric>/scores.jsonl.gz``: one JSON record per study of
the set, in generation order, with ``study_id``, ``set`` and the metric's per-study
scores. The manifest's ``summary`` has the means and, for CXR-Report-Metric, the
leaderboard's ``1/RadCliQ-v1`` = 1 / mean(RadCliQ-v1) (corpus-level, as ReXrank).

Metrics: ``cxr-report-metric`` (BLEU-2, BERTScore, SembScore, RadGraph F1, RadCliQ;
CPU only) and ``ratescore`` (one GPU: submit with --gres=gpu:1; the container gets the
job's GPU from CUDA_VISIBLE_DEVICES).

Each metric is one service of eval/compose.yaml: the upstream's own code and pins in its
own image (docs/revision-plan.md §7). The stage builds the image from this checkout
before running it (BuildKit's cache makes that near-instant when nothing changed); the
image is tagged with the commit, so jobs at different commits never share an image.

Inputs are formatted as ReXrank's pipeline does (rajpurkarlab/ReXrank-metric,
scripts/json_to_csv.py):
- whitespace runs (CheXpert Plus keeps its line wraps) are collapsed to one space; the
  texts ReXrank scores have none (e.g. its CheXpert Plus example patient64545_study1);
- the section name is prepended, ``Findings: <text>``, to the reference (as ReXrank)
  and to the generated report (ReXrank scores submissions as sent, and they carry it;
  our generations don't, so it is added here, at evaluation time only);
- study ids are integers 0..n-1 in generation order (upstream casts them with int()).

Model weights: each metric lists the configs/weights.yaml entries its container reads
under /models (the host weights root, mounted read-only); they are checked (present,
sha256) before anything is built and recorded in the manifest's inputs.

Studies without a report (the output cap was hit while thinking, or an empty answer)
are refused for now: how to score them is not decided yet.
"""

import gzip
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from labrag import artifacts, paths, serving, weights

COMPOSE_FILE = paths.REPO_ROOT / "eval" / "compose.yaml"
SECTION_PREFIX = {"findings": "Findings: "}

# metric -> compose service, the upstream submodule its image is built from, whether it
# runs on the job's GPU, the configs/weights.yaml entries it reads under /models, and its
# per-study score columns
METRICS = {
    "cxr-report-metric": {
        "service": "cxr-report-metric",
        "upstream": "external/CXR-Report-Metric",
        "gpu": False,
        "weights": [
            "chexbert",
            "cxr-report-metric.bert-base-uncased",
            "cxr-report-metric.distilroberta-base",
            "cxr-report-metric.biomedbert",
            "cxr-report-metric.radgraph",
        ],
        "scores": [
            "bleu_score",
            "bertscore",
            "semb_score",
            "radgraph_combined",
            "RadCliQ-v0",
            "RadCliQ-v1",
        ],
    },
    "ratescore": {
        "service": "ratescore",
        "upstream": "external/RaTEScore",
        "gpu": True,
        "weights": ["ratescore.rate-ner-deberta", "ratescore.biolord-2023-c"],
        "scores": ["ratescore"],
    },
}


def format_report(text: str, section: str) -> str:
    """Report text as ReXrank scores it: whitespace collapsed, section name prepended."""
    return SECTION_PREFIX[section] + " ".join(text.split())


def run(generations: list[str], metric: str, overwrite: bool = False) -> list[Path]:
    spec = METRICS[metric]
    jobs = []  # (generations relpath, output relpath, config)
    for rel in generations:
        rel = rel.strip("/").removeprefix("generations/")
        config = {
            "stage": "evaluate",
            "generations": f"generations/{rel}",
            "metric": metric,
            # What determines the scores: the upstream commit and this repo's image
            # definition (Dockerfile, patch, wrapper), by git tree hash. Not
            # eval/compose.yaml: it also holds the other services and run settings
            # (recorded in the manifest's image record instead).
            "image": image_source(spec),
            "format": {
                "whitespace": "runs collapsed to one space",
                "prefix": SECTION_PREFIX,
                "study_ids": "0..n-1 in generation order",
            },
        }
        jobs.append((f"generations/{rel}", f"scores/{rel}/{metric}", config))
    todo = [j for j in jobs if not artifacts.is_built(j[1], j[2], overwrite)]
    for _, out, _ in jobs:
        if all(out != t[1] for t in todo):
            print(f"SKIP: {paths.SHARED_ROOT / out} already built with this config")
    if not todo:
        return [paths.SHARED_ROOT / out for _, out, _ in jobs]

    # Check every input before building the image: the weights the container reads
    # (present, with the sha256 in configs/weights.yaml), then the artifacts.
    weight_records = {name: weights.resolve(name)[1] for name in spec["weights"]}
    loaded = []
    for gen_rel, out, config in todo:
        gen_dir = paths.SHARED_ROOT / gen_rel
        gen_record = artifacts.artifact_record(gen_dir)
        gen_config = artifacts.read_manifest(gen_dir)["config"]
        prompt_dir = paths.SHARED_ROOT / gen_config["prompts"]
        prompt_record = artifacts.artifact_record(prompt_dir)
        section = artifacts.read_manifest(prompt_dir)["config"]["section"]
        rows = _read_jsonl(gen_dir / "generations.jsonl.gz")
        references = {
            r["study_id"]: r["reference"]
            for r in _read_jsonl(prompt_dir / "prompts.jsonl.gz")
            if r["set"] == gen_config["set"]
        }
        pairs = pair_reports(rows, references, section, gen_rel)
        inputs = {
            "generations": gen_record,
            "prompts": prompt_record,
            "weights": weight_records,
        }
        loaded.append((out, config, inputs, rows, pairs))

    image = build_image(spec)
    built = []
    for out, config, inputs, rows, pairs in loaded:
        scores, timing = score(spec, image, pairs)
        records = [
            {"study_id": r["study_id"], "set": r["set"], **s}
            for r, s in zip(rows, scores)
        ]
        extra = {
            "image": image,
            "timing": timing,
            "summary": summarize(records, spec["scores"]),
        }

        def build(dest: Path, records=records, extra=extra, out=out) -> dict:
            with open(dest / "scores.jsonl.gz", "wb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as f:
                    for r in records:
                        f.write((json.dumps(r) + "\n").encode())
            print(out, json.dumps(extra["summary"], indent=2))
            return extra

        built.append(
            artifacts.publish(out, config, {**inputs, "image": image}, build, overwrite)
        )
    return built


def pair_reports(
    rows: list[dict], references: dict[str, str], section: str, name: str
) -> pd.DataFrame:
    """Integer id, formatted reference and formatted report per generation, in order."""
    missing = [r["study_id"] for r in rows if r["study_id"] not in references]
    if missing:
        raise SystemExit(f"{name}: {len(missing)} studies have no reference")
    empty = [r["study_id"] for r in rows if not (r["report"] or "").strip()]
    if empty:
        raise SystemExit(
            f"{name}: {len(empty)} studies have no report (e.g. {empty[0]}); how to "
            "score them is not decided yet"
        )
    if len({r["study_id"] for r in rows}) != len(rows):
        raise SystemExit(f"{name}: duplicate study ids")
    return pd.DataFrame(
        {
            "study_id": range(len(rows)),
            "reference": [
                format_report(references[r["study_id"]], section) for r in rows
            ],
            "generated": [format_report(r["report"], section) for r in rows],
        }
    )


def score(spec: dict, image: dict, pairs: pd.DataFrame) -> tuple[list[dict], dict]:
    """Run the metric's image on one set of pairs; per-study scores in pair order."""
    if spec["gpu"]:  # compose reads CUDA_VISIBLE_DEVICES itself (eval/compose.yaml)
        devices = serving.gpu_devices()
        if len(devices) != 1:
            raise SystemExit(f"{spec['service']} runs on one GPU, got {devices}")
    # Local disk, not NFS: the container writes its intermediates next to the inputs.
    with tempfile.TemporaryDirectory(prefix="labrag-eval-") as work:
        work = Path(work)
        for column, name in (("reference", "gt"), ("generated", "pred")):
            pairs.rename(columns={column: "report"})[["study_id", "report"]].to_csv(
                work / f"{name}.csv", index=False
            )
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
            f"OMP_NUM_THREADS={torch_threads()}",
            spec["service"],
            str(work / "gt.csv"),
            str(work / "pred.csv"),
            str(work / "out.csv"),
            tag=image["tag"],
        )
        out = pd.read_csv(work / "out.csv")
        timing = json.loads((work / "out.json").read_text())
    out = out.set_index("study_id").loc[pairs["study_id"], spec["scores"]]
    if out.isna().any().any():
        raise SystemExit("the metric returned missing scores")
    return [{k: float(v) for k, v in row.items()} for _, row in out.iterrows()], timing


def torch_threads() -> int:
    """Threads for torch in the container: half the job's CPUs.

    torch 1.6 parallelizes even tiny ops, so one preempted thread stalls each op;
    threads == CPUs on a shared node ran RadGraph ~7x slower (2026-11-13). Our Slurm
    doesn't confine jobs, so the job's share is --cpus-per-task, not the node.
    """
    ncpu = int(os.environ.get("SLURM_CPUS_PER_TASK") or len(os.sched_getaffinity(0)))
    return max(1, ncpu // 2)


def image_source(spec: dict) -> dict:
    """Git ids of what the image is built from (committed state of this checkout)."""
    service_dir = f"eval/{spec['service']}"
    return {
        "service": spec["service"],
        "upstream_commit": _git("rev-parse", f"HEAD:{spec['upstream']}"),
        "definition_tree": _git("rev-parse", f"HEAD:{service_dir}"),
    }


def build_image(spec: dict) -> dict:
    """Build the service's image from this checkout; its tag and image id."""
    commit = artifacts.git_commit()
    # A dirty dev tree builds what's on disk, which no commit describes.
    tag = "dev" if commit["dirty"] else commit["sha"]
    compose("build", spec["service"], tag=tag)
    ref = f"labrag/{spec['service']}:{tag}"
    image_id = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", ref],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {
        "ref": ref,
        "tag": tag,
        "image_id": image_id,
        "compose_blob": _git("rev-parse", "HEAD:eval/compose.yaml"),
    }


def compose(*args: str, tag: str) -> None:
    # The container mounts the weights root the stage checked.
    env = {
        **os.environ,
        "LABRAG_COMMIT": tag,
        "LABRAG_MODELS_ROOT": str(paths.MODELS_ROOT),
    }
    cmd = ["docker", "compose", "-f", str(COMPOSE_FILE), *args]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=True)


def summarize(records: list[dict], columns: list[str]) -> dict:
    df = pd.DataFrame(records)
    summary = {"n": len(df), "mean": {c: float(df[c].mean()) for c in columns}}
    if "RadCliQ-v1" in columns:
        summary["1/RadCliQ-v1"] = float(1.0 / np.mean(df["RadCliQ-v1"]))
    return summary


def _read_jsonl(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as f:  # one read of each NFS input
        return [json.loads(line) for line in f]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(paths.REPO_ROOT), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
