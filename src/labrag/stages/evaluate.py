"""Stage ``evaluate``: per-study scores of generated reports, by official metric code.

Reads a generations artifact (stage generate) and the prompt artifact it was generated
from (for the reference text), and writes
``scores/<generations path>/<metric>/scores.jsonl.gz``: one JSON record per study of
the set, in generation order, with ``study_id``, ``set`` and the metric's per-study
scores. The manifest's ``summary`` has the means and, for CXR-Report-Metric, the
leaderboard's ``1/RadCliQ-v1`` = 1 / mean(RadCliQ-v1) (corpus-level, as ReXrank).

Metrics: ``cxr-report-metric`` (BLEU-2, BERTScore, SembScore, RadGraph F1, RadCliQ;
CPU only), ``ratescore`` and ``green`` (GREEN score, its clinically significant error
counts and matched findings, plus the judge's text). The GPU metrics run on one GPU:
submit with --gres=gpu:1; the container gets the job's GPU from CUDA_VISIBLE_DEVICES.

``green`` serves GREEN's judge with vLLM (compose service green-judge, on the job's
GPU; the client is service green-vllm); GREEN's image does all the text work with
upstream's code (see eval/green/run.py). One container run scores every set of the
job, in one request queue; the job's compose project, server included, is taken down
at the end. ``green-hf``, the reference: the same scores by upstream's own HF generate
(service green-hf, ≈1.6 s per study on an H200). ``labrag evaluate --metric green[-hf]
--repro`` runs upstream's
5-case reproduction test through either judge path.

Each metric is one service of eval/compose.yaml: the upstream's own code and pins in its
own image (docs/revision-plan.md §7). The stage builds the image from this checkout
before running it (BuildKit's cache makes that near-instant when nothing changed); the
image is tagged with the commit, so jobs at different commits never share an image.

Inputs (Findings only, for now; other sections are refused until their format is
decided):
- content only, as ReXrank's Findings board scores every metric since 2026-08-06
  (rajpurkarlab/ReXrank gh-pages 1db77de): no section header on either text; a
  leading "Findings:" on a generated report is removed (our references have none);
- whitespace runs (CheXpert Plus keeps its line wraps) are collapsed to one space. Our
  correction, not ReXrank's (its json_to_csv.py passes texts as they are): upstream's
  BLEU splits on " " only, so a line break would join two words into one token;
- study ids are integers 0..n-1 in generation order, as ReXrank-metric's json_to_csv.py
  (upstream casts them with int()).

Model weights: each metric lists the configs/weights.yaml entries its container reads
under /models (the host weights root, mounted read-only); they are checked (present,
sha256) before anything is built and recorded in the manifest's inputs.

Studies without a report (the output cap was hit while thinking, or an empty answer)
are scored as the text "EMPTY" and stay in every mean; each record has ``empty_report``
and the summary counts them. Empty references are refused.
"""

import gzip
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from labrag import artifacts, paths, serving, weights

COMPOSE_FILE = paths.REPO_ROOT / "eval" / "compose.yaml"
SECTIONS = ["findings"]  # the sections whose scoring format is decided
FINDINGS_HEADER = re.compile(r"^findings\s*:\s*", re.IGNORECASE)
# An empty generated report (thinking hit the output cap, or an empty answer) is scored
# as this text by every metric, so each metric gives its own verdict on a report that
# says nothing (the preprint's rrg/eval.py did the same, spelled "EMTPY").
EMPTY_REPORT = "EMPTY"
FORMAT = {
    "whitespace": "runs collapsed to one space",
    "section_header": "none; a leading 'Findings:' is removed",
    "study_ids": "0..n-1 in generation order",
    "empty_report": f"scored as the text {EMPTY_REPORT!r}",
}

# metric -> compose service, the upstream submodule its image is built from, whether it
# runs on the job's GPU, the configs/weights.yaml entries it reads under /models, its
# per-study score columns (numbers, averaged in the summary) and optional text columns
# (kept per study, e.g. an LLM judge's output). Optional: ``build``, the service that
# builds the image (default: the service itself); ``judge``, the compose service that
# serves the metric's LLM judge (``run`` starts it; the stage takes it down).
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
    "green": {
        "service": "green-vllm",
        "build": "green-hf",
        "upstream": "external/GREEN",
        "gpu": True,
        "weights": ["green.radllama2-7b"],
        "scores": [
            "green_score",
            "green_sig_false_finding",
            "green_sig_missing_finding",
            "green_sig_location",
            "green_sig_severity",
            "green_sig_false_comparison",
            "green_sig_missing_comparison",
            "green_matched_findings",
            "green_unparsed",
        ],
        "texts": ["green_analysis"],
        "judge": "green-judge",
    },
}
METRICS["green-hf"] = {  # the reference: upstream's HF generate
    **{k: v for k, v in METRICS["green"].items() if k not in ("build", "judge")},
    "service": "green-hf",
}


def format_report(text: str) -> str:
    """Findings text as scored: whitespace collapsed, a leading "Findings:" removed."""
    return FINDINGS_HEADER.sub("", " ".join(text.split()), count=1)


def run(generations: list[str], metric: str, overwrite: bool = False) -> list[Path]:
    spec = METRICS[metric]
    jobs = []  # (generations relpath, output relpath, config)
    for rel in generations:
        rel = rel.strip("/").removeprefix("generations/")
        config = {
            "stage": "evaluate",
            "generations": f"generations/{rel}",
            "metric": metric,
            # What determines the scores: the service that runs, the upstream commit
            # and this repo's image definition (Dockerfile, patch, wrapper), by git
            # tree hash. Not eval/compose.yaml: it also holds the other services and
            # run settings (recorded in the manifest's image record instead).
            "image": image_source(spec),
            "format": FORMAT,
        }
        if "judge" in spec:
            config["judge"] = judge_definition(spec)
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
    judge_image = judge_image_record(spec) if "judge" in spec else None
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
        if judge_image:
            inputs["judge_image"] = judge_image
        loaded.append((out, config, inputs, rows, pairs))

    image = build_image(spec)
    if "judge" in spec:  # one run and one server for every set, a set at a time
        started = artifacts.now()
        judged = score_sets(spec, image, {out: pairs for out, *_, pairs in loaded})
    built = []
    for k, (out, config, inputs, rows, pairs) in enumerate(loaded):
        if "judge" in spec:
            scores, timing = judged[k]
        else:
            started = artifacts.now()
            scores, timing = score(spec, image, pairs)
        records = [
            {"study_id": r["study_id"], "set": r["set"], "empty_report": e, **s}
            for r, e, s in zip(rows, pairs["empty_report"].tolist(), scores)
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
            artifacts.publish(
                out,
                config,
                {**inputs, "image": image},
                build,
                overwrite,
                started=started,
            )
        )
    return built


def pair_reports(
    rows: list[dict], references: dict[str, str], section: str, name: str
) -> pd.DataFrame:
    """Integer id, formatted reference and formatted report per generation, in order."""
    if section not in SECTIONS:
        raise SystemExit(f"{name}: how to score {section} is not decided yet")
    missing = [r["study_id"] for r in rows if r["study_id"] not in references]
    if missing:
        raise SystemExit(f"{name}: {len(missing)} studies have no reference")
    reference = [format_report(references[r["study_id"]]) for r in rows]
    if not all(reference):
        raise SystemExit(f"{name}: empty reference texts")
    generated = [format_report(r["report"] or "") for r in rows]
    if len({r["study_id"] for r in rows}) != len(rows):
        raise SystemExit(f"{name}: duplicate study ids")
    return pd.DataFrame(
        {
            "study_id": range(len(rows)),
            "reference": reference,
            "generated": [text or EMPTY_REPORT for text in generated],
            "empty_report": [not text for text in generated],
        }
    )


def score(spec: dict, image: dict, pairs: pd.DataFrame) -> tuple[list[dict], dict]:
    """Run the metric's image on one set of pairs; per-study scores in pair order."""
    check_gpu(spec)
    # Local disk, not NFS: the container writes its intermediates next to the inputs.
    with tempfile.TemporaryDirectory(prefix="labrag-eval-") as work:
        work = Path(work)
        write_pairs(pairs, work)
        run_service(
            spec,
            image,
            work,
            *(str(work / f) for f in ("gt.csv", "pred.csv", "out.csv")),
        )
        timing = json.loads((work / "out.json").read_text())
        return read_scores(spec, work / "out.csv", pairs), timing


def score_sets(
    spec: dict, image: dict, sets: dict[str, pd.DataFrame]
) -> list[tuple[list[dict], dict]]:
    """Run a judged metric on several sets of pairs (artifact name -> pairs) in one run
    of its ``score`` command, which judges them one set at a time on one server.

    Per set, in order: the per-study scores in pair order and the timing. Everything at
    the timing's top level is the set's own: its time on the server, finish reasons,
    token counts and server counters (the counters' change over the set), and the job's
    GPU. ``run`` is shared by every set of the run, so count it once per run: the sets
    and ``overhead_seconds``, the run's wall time (server startup and shutdown included)
    minus the sets' own times."""
    devices = check_gpu(spec)
    with tempfile.TemporaryDirectory(prefix="labrag-eval-") as work:
        work = Path(work)
        dirs = [work / str(i) for i in range(len(sets))]
        for d, pairs in zip(dirs, sets.values()):
            d.mkdir()
            write_pairs(pairs, d)
        start = time.monotonic()
        with_judge(spec, image, lambda: run_service(
            spec, image, work, "score", *map(str, dirs)
        ))  # fmt: skip
        run_seconds = time.monotonic() - start
        results = []
        for d, pairs in zip(dirs, sets.values()):
            timing = json.loads((d / "out.json").read_text())
            timing["gpus"] = serving.gpu_names(devices)
            before, after = (
                serving.counter_totals((d / f"out.{when}.prom").read_text())
                for when in ("start", "end")
            )
            timing["vllm_counters"] = {
                k: v - before.get(k, 0.0) for k, v in after.items()
            }
            results.append((read_scores(spec, d / "out.csv", pairs), timing))
        shared = {
            "sets": list(sets),
            "overhead_seconds": round(
                run_seconds - sum(t["seconds"] for _, t in results)
            ),
        }
        return [(scores, {**timing, "run": shared}) for scores, timing in results]


def with_judge(spec: dict, image: dict, run) -> None:
    """Call ``run`` (a compose run that starts the judge service), then always take the
    job's compose project down (the server included), after copying the judge's log
    into the job log; also when the job is cancelled (Slurm's SIGTERM)."""
    serving.raise_on_sigterm()
    try:
        run()
    finally:
        compose("logs", "--no-color", spec["judge"], tag=image["tag"], check=False)
        compose("down", "--timeout", "30", tag=image["tag"], check=False)


def repro(metric: str) -> None:
    """Upstream's reproduction test of an LLM-judge metric through its judge path (the
    service's ``--repro``); exits 1 on a mismatch."""
    spec = METRICS[metric]
    if spec["upstream"] != "external/GREEN":
        raise SystemExit(f"--repro: no reproduction test for {metric}")
    check_gpu(spec)
    weights.resolve(spec["weights"][0])
    if "judge" in spec:
        judge_image_record(spec)
    image = build_image(spec)
    with tempfile.TemporaryDirectory(prefix="labrag-eval-") as work:
        work = Path(work)
        if "judge" in spec:
            with_judge(spec, image, lambda: run_service(spec, image, work, "--repro"))
        else:
            run_service(spec, image, work, "--repro")


def check_gpu(spec: dict) -> list[str]:
    """The job's one GPU, for a GPU metric (compose reads CUDA_VISIBLE_DEVICES itself)."""
    if not spec["gpu"]:
        return []
    devices = serving.gpu_devices()
    if len(devices) != 1:
        raise SystemExit(f"{spec['service']} runs on one GPU, got {devices}")
    return devices


def write_pairs(pairs: pd.DataFrame, work: Path) -> None:
    """The service's inputs: gt.csv and pred.csv (study_id, report) in ``work``."""
    for column, name in (("reference", "gt"), ("generated", "pred")):
        pairs.rename(columns={column: "report"})[["study_id", "report"]].to_csv(
            work / f"{name}.csv", index=False
        )


def run_service(spec: dict, image: dict, work: Path, *args: str) -> None:
    """One run of the metric's container, with ``work`` mounted at the same path."""
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
        *args,
        tag=image["tag"],
    )


def read_scores(spec: dict, out_csv: Path, pairs: pd.DataFrame) -> list[dict]:
    """Per-study scores (and texts) of a service's out.csv, in pair order."""
    out = pd.read_csv(out_csv).set_index("study_id")
    texts = spec.get("texts", [])
    if texts:  # as written: "" and "NA" stay strings
        out[texts] = pd.read_csv(
            out_csv, keep_default_na=False, dtype={t: str for t in texts}
        ).set_index("study_id")[texts]
    out = out.loc[pairs["study_id"]]
    if out[spec["scores"]].isna().any().any():
        raise SystemExit("the metric returned missing scores")
    return [
        {
            **{k: float(row[k]) for k in spec["scores"]},
            **{k: row[k] for k in texts},
        }
        for _, row in out.iterrows()
    ]


def judge_definition(spec: dict) -> dict:
    """The judge service's image and command (what determines its outputs), for the
    artifact config; the rest of eval/compose.yaml is not part of it."""
    service = yaml.safe_load(COMPOSE_FILE.read_text())["services"][spec["judge"]]
    return {"service": spec["judge"], **{k: service[k] for k in ("image", "command")}}


def judge_image_record(spec: dict) -> dict:
    """The judge's pinned image, after checking this node has it (jobs never pull)."""
    ref = judge_definition(spec)["image"]
    out = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", ref],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise SystemExit(
            f"judge image {ref} not on this node; pre-pull it with\n"
            f"  docker pull {ref}\n{out.stderr.strip()}"
        )
    return {"ref": ref, "image_id": out.stdout.strip()}


def torch_threads() -> int:
    """Threads for torch in the container: half the job's CPUs.

    torch 1.6 parallelizes even tiny ops, so one preempted thread stalls each op;
    threads == CPUs on a shared node ran RadGraph ~7x slower (2026-11-13). Our Slurm
    doesn't confine jobs, so the job's share is --cpus-per-task, not the node.
    """
    ncpu = int(os.environ.get("SLURM_CPUS_PER_TASK") or len(os.sched_getaffinity(0)))
    return max(1, ncpu // 2)


def image_source(spec: dict) -> dict:
    """The compose service the metric runs (green-vllm and green-hf share one image,
    so this is what tells them apart) and the git ids of what its image is built from
    (committed state of this checkout).
    """
    _, context, _ = image_build(spec)
    return {
        "service": spec["service"],
        "upstream_commit": _git("rev-parse", f"HEAD:{spec['upstream']}"),
        "definition_tree": _git("rev-parse", f"HEAD:{context}"),
    }


def image_build(spec: dict) -> tuple[str, str, str]:
    """The compose service that builds the metric's image, its build context (relative
    to the repo) and its image name without the tag, as eval/compose.yaml defines them.
    """
    service = spec.get("build", spec["service"])
    definition = yaml.safe_load(COMPOSE_FILE.read_text())["services"][service]
    context = (COMPOSE_FILE.parent / definition["build"]["context"]).resolve()
    name = definition["image"].split(":")[0]
    return service, str(context.relative_to(paths.REPO_ROOT)), name


def build_image(spec: dict) -> dict:
    """Build the service's image from this checkout; its tag and image id."""
    commit = artifacts.git_commit()
    # A dirty dev tree builds what's on disk, which no commit describes.
    tag = "dev" if commit["dirty"] else commit["sha"]
    service, _, name = image_build(spec)
    compose("build", service, tag=tag)
    ref = f"{name}:{tag}"
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


def compose(*args: str, tag: str, check: bool = True) -> None:
    """docker compose on eval/compose.yaml, as this job's own project (jobs sharing a
    node never share a container or network)."""
    # The container mounts the weights root the stage checked.
    env = {
        **os.environ,
        "LABRAG_COMMIT": tag,
        "LABRAG_MODELS_ROOT": str(paths.MODELS_ROOT),
    }
    project = f"labrag-eval-{os.environ.get('SLURM_JOB_ID') or os.getpid()}"
    cmd = ["docker", "compose", "-f", str(COMPOSE_FILE), "-p", project, *args]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=check)


def summarize(records: list[dict], columns: list[str]) -> dict:
    df = pd.DataFrame(records)
    summary = {
        "n": len(df),
        "empty_reports": int(df["empty_report"].sum()),
        "mean": {c: float(df[c].mean()) for c in columns},
    }
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
