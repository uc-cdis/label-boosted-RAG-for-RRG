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

One job runs the whole suite (``SUITE``, the default) at once, so the metrics share its
GPU and CPUs and finish at about the same time: cxr-report-metric starts first, on CPU;
then GREEN's judge starts and sizes its KV cache on the empty GPU; then GREEN and
RaTEScore share the GPU. cxr-report-metric, the slowest per study, scores each set in
chunks of ``chunk`` studies (fixed, so a study's scores never depend on the job's CPU
count), one container per chunk, as many at a time as the job's CPUs allow
(``plan_threads``): --cpus-per-task is what balances it against the GPU metrics. Each
metric publishes its artifacts when its own scores are in; a failed metric doesn't
stop the others (the job then fails at the end).

``green`` serves GREEN's judge with vLLM (compose service green-vllm, on the job's
GPU; the client is service green-client), in vLLM's batch-invariant mode, so a study's
score doesn't depend on what it is batched with; GREEN's image does all the text work
with upstream's code (see eval/green/run.py). One container run scores every set of the
job, in one request queue; the job's compose project, server included, is taken down
at the end. ``green-native``, the reference: the same scores by upstream's own HF
generate (service green-native, ≈1.6 s per study on an H200). ``labrag evaluate
--metric green[-native] --repro`` runs upstream's 5-case reproduction test through
either judge path.

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

import concurrent.futures
import contextlib
import gzip
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import traceback
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
# runs on the job's GPU, torch threads per container, the configs/weights.yaml entries
# it reads under /models, its per-study score columns (numbers, averaged in the summary)
# and optional text columns (kept per study, e.g. an LLM judge's output). Optional:
# ``build``, the service that builds the image (default: the service itself); ``judge``,
# the compose service that serves the metric's LLM judge (the stage starts it before
# the GPU metrics and takes it down at the end); ``chunk``, studies per container run
# (part of the scores' config; containers run in parallel, see plan_threads).
METRICS = {
    "cxr-report-metric": {
        "service": "cxr-report-metric",
        "upstream": "external/CXR-Report-Metric",
        "gpu": False,
        # 8 threads: 0.48 s per study vs 0.42 at 16 (pilot, 2026-11-13), so more
        # containers of fewer threads score more studies per CPU
        "threads": 8,
        # ~15 s of model loads per container run, ≈3% of a chunk
        "chunk": 1000,
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
        "threads": 8,
        "weights": ["ratescore.rate-ner-deberta", "ratescore.biolord-2023-c"],
        "scores": ["ratescore"],
    },
    "green": {
        "service": "green-client",
        "build": "green-native",
        "upstream": "external/GREEN",
        "gpu": True,
        "threads": 4,  # GREEN's text work; the judge runs in its own container
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
        "judge": "green-vllm",
    },
}
METRICS["green-native"] = {  # the reference: upstream's HF generate
    **{k: v for k, v in METRICS["green"].items() if k not in ("build", "judge")},
    "service": "green-native",
}
# The default of `labrag evaluate`: every metric of the suite, in one job.
SUITE = ["cxr-report-metric", "ratescore", "green"]
# artifacts.publish reads the umask by setting it, which isn't thread-safe.
_PUBLISH = threading.Lock()


def format_report(text: str) -> str:
    """Findings text as scored: whitespace collapsed, a leading "Findings:" removed."""
    return FINDINGS_HEADER.sub("", " ".join(text.split()), count=1)


def run(
    generations: list[str], metrics: list[str] | None = None, overwrite: bool = False
) -> list[Path]:
    """Score generations artifacts with ``metrics`` (default: the suite), at once."""
    metrics = metrics or SUITE
    if "green-native" in metrics and len(metrics) > 1:
        raise SystemExit("green-native runs alone (its judge would share the GPU)")
    plans, done = [], []
    for metric in metrics:
        plan, built = prepare(metric, generations, overwrite)
        done += built
        if plan:
            plans.append(plan)
    if not plans:
        return done
    for plan in plans:  # before anything runs: a GPU metric needs the job's one GPU
        check_gpu(plan["spec"])
        plan["image"] = build_image(plan["spec"])
    judged = next((p for p in plans if "judge" in p["spec"]), None)
    plan_threads(plans)
    for plan in plans:  # recorded with the timing: what shared the job's GPU and CPUs
        plan["alongside"] = [p["metric"] for p in plans if p is not plan]

    serving.raise_on_sigterm()
    pool = concurrent.futures.ThreadPoolExecutor(len(plans))
    futures = {}
    try:
        # CPU metrics first: they don't wait for the judge
        for plan in plans:
            if not plan["spec"]["gpu"]:
                futures[pool.submit(score_metric, plan, overwrite)] = plan["metric"]
        session = (
            judge_session(judged["spec"], judged["image"], judged)
            if judged
            else contextlib.nullcontext()
        )
        with session:  # the judge sizes its KV cache before anything else is on the GPU
            for plan in plans:
                if plan["spec"]["gpu"]:
                    futures[pool.submit(score_metric, plan, overwrite)] = plan["metric"]
            concurrent.futures.wait(futures)
    finally:
        pool.shutdown(cancel_futures=True)
    failed = []
    for future, metric in futures.items():
        if future.exception() is None:
            done += future.result()
        else:
            failed.append(metric)
            print(f"[{metric}] FAILED:", flush=True)
            traceback.print_exception(future.exception())
    if failed:
        raise SystemExit(f"evaluate: failed metrics: {', '.join(failed)}")
    return done


def prepare(
    metric: str, generations: list[str], overwrite: bool
) -> tuple[dict | None, list[Path]]:
    """What ``metric`` has to score: the sets not yet built with this config, their
    inputs checked and loaded (None if every set is built), and the built ones."""
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
        if "chunk" in spec:
            config["chunk_studies"] = spec["chunk"]
        jobs.append((f"generations/{rel}", f"scores/{rel}/{metric}", config))
    todo = [j for j in jobs if not artifacts.is_built(j[1], j[2], overwrite)]
    built = []
    for _, out, _ in jobs:
        if all(out != t[1] for t in todo):
            print(f"SKIP: {paths.SHARED_ROOT / out} already built with this config")
            built.append(paths.SHARED_ROOT / out)
    if not todo:
        return None, built

    # Check every input before building the image: the weights the container reads
    # (present, with the sha256 in configs/weights.yaml), then the artifacts.
    weight_records = {name: weights.resolve(name)[1] for name in spec["weights"]}
    judge_image = judge_image_record(spec) if "judge" in spec else None
    sets = []
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
        sets.append({"out": out, "config": config, "inputs": inputs, "rows": rows,
                     "pairs": pairs})  # fmt: skip
    return {"metric": metric, "spec": spec, "sets": sets}, built


def plan_threads(plans: list[dict]) -> None:
    """Torch threads per container, and how many chunk containers run at a time.

    Half the job's CPUs in all, as torch_threads: the GPU metrics' containers first,
    the rest to the chunked metric, at its threads per container. Our Slurm doesn't
    confine jobs, so the job's share is --cpus-per-task, not the node."""
    budget = torch_threads()
    for plan in plans:
        plan["threads"] = min(plan["spec"]["threads"], budget)
        plan["workers"] = 1
    left = budget - sum(p["threads"] for p in plans if "chunk" not in p["spec"])
    for plan in plans:
        if "chunk" in plan["spec"]:
            plan["workers"] = max(1, left // plan["threads"])
        print(
            f"[{plan['metric']}] {len(plan['sets'])} sets, {plan['workers']} "
            f"container(s) at a time, {plan['threads']} threads each",
            flush=True,
        )


def score_metric(plan: dict, overwrite: bool) -> list[Path]:
    """Score every set of the plan, publishing each set's artifact as its scores are
    in (each manifest's ``started``: when the metric began, shared by its sets)."""
    spec, image, sets = plan["spec"], plan["image"], plan["sets"]
    started = artifacts.now()
    start = time.monotonic()
    if "judge" in spec:  # one client run for every set, on the one server
        results = score_sets(spec, image, {s["out"]: s["pairs"] for s in sets}, plan)
    elif "chunk" in spec:
        results = score_chunks(spec, image, sets, plan)
    else:
        results = (
            score(spec, image, s["pairs"], plan["threads"], plan["metric"])
            for s in sets
        )
    built = [
        publish_scores(plan, s, scores, timing, overwrite, started)
        for s, (scores, timing) in zip(sets, results)
    ]
    print(f"[{plan['metric']}] done in {time.monotonic() - start:.0f} s", flush=True)
    return built


def score_chunks(spec: dict, image: dict, sets: list[dict], plan: dict):
    """Scores of each set (in order, as each set's chunks are in), from containers of
    ``spec["chunk"]`` studies each, ``plan["workers"]`` at a time. A set's timing: its
    containers' summed seconds and its chunk count."""
    size = spec["chunk"]
    pool = concurrent.futures.ThreadPoolExecutor(plan["workers"])
    try:
        chunks = [
            [
                pool.submit(score, spec, image, chunk, plan["threads"], plan["metric"])
                for chunk in (
                    s["pairs"].iloc[lo : lo + size]
                    for lo in range(0, len(s["pairs"]), size)
                )
            ]
            for s in sets
        ]
        for s, futures in zip(sets, chunks):
            parts = [f.result() for f in futures]
            timing = {
                "reports": len(s["pairs"]),
                "seconds": round(sum(t["seconds"] for _, t in parts), 1),
                "chunks": len(parts),
                "torch_threads": parts[0][1]["torch_threads"],
                "parallel_containers": plan["workers"],
            }
            yield [x for scores, _ in parts for x in scores], timing
    finally:
        pool.shutdown(cancel_futures=True)


def publish_scores(
    plan: dict, s: dict, scores: list[dict], timing: dict, overwrite: bool, started: str
) -> Path:
    """Publish one set's scores artifact (``s``: an entry of the plan's sets)."""
    spec, image = plan["spec"], plan["image"]
    records = [
        {"study_id": r["study_id"], "set": r["set"], "empty_report": e, **x}
        for r, e, x in zip(s["rows"], s["pairs"]["empty_report"].tolist(), scores)
    ]
    extra = {
        "image": image,
        "timing": {**timing, "alongside": plan.get("alongside", [])},
        "summary": summarize(records, spec["scores"]),
    }

    def build(dest: Path) -> dict:
        with open(dest / "scores.jsonl.gz", "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as f:
                for r in records:
                    f.write((json.dumps(r) + "\n").encode())
        print(s["out"], json.dumps(extra["summary"], indent=2), flush=True)
        return extra

    with _PUBLISH:
        return artifacts.publish(
            s["out"],
            s["config"],
            {**s["inputs"], "image": image},
            build,
            overwrite,
            started=started,
        )


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


def score(
    spec: dict,
    image: dict,
    pairs: pd.DataFrame,
    threads: int | None = None,
    prefix: str | None = None,
) -> tuple[list[dict], dict]:
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
            threads=threads,
            prefix=prefix,
        )
        timing = json.loads((work / "out.json").read_text())
        return read_scores(spec, work / "out.csv", pairs), timing


def score_sets(
    spec: dict, image: dict, sets: dict[str, pd.DataFrame], plan: dict | None = None
) -> list[tuple[list[dict], dict]]:
    """Run a judged metric on several sets of pairs (artifact name -> pairs) in one run
    of its ``score`` command, which judges them one set at a time on the judge server
    (started by judge_session).

    Per set, in order: the per-study scores in pair order and the timing. Everything at
    the timing's top level is the set's own: its time on the server, finish reasons,
    token counts and server counters (the counters' change over the set), and the job's
    GPU. ``run`` is shared by every set of the run, so count it once per run: the sets,
    ``overhead_seconds`` (the client run's wall time minus the sets' own times), the
    judge's startup time and the metrics that shared the job's GPU and CPUs."""
    plan = plan or {}
    devices = check_gpu(spec)
    with tempfile.TemporaryDirectory(prefix="labrag-eval-") as work:
        work = Path(work)
        dirs = [work / str(i) for i in range(len(sets))]
        for d, pairs in zip(dirs, sets.values()):
            d.mkdir()
            write_pairs(pairs, d)
        start = time.monotonic()
        run_service(
            spec, image, work, "score", *map(str, dirs),
            threads=plan.get("threads"), prefix=plan.get("metric"),
        )  # fmt: skip
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
            "judge_startup_seconds": plan.get("judge_startup_seconds"),
        }
        return [(scores, {**timing, "run": shared}) for scores, timing in results]


@contextlib.contextmanager
def judge_session(spec: dict, image: dict, plan: dict | None = None):
    """Start the metric's judge service and wait until it answers; on exit, always
    copy its log into the job log and take the job's compose project down (the server
    included), also when the job is cancelled (Slurm's SIGTERM). Records the startup
    time in ``plan``."""
    serving.raise_on_sigterm()
    start = time.monotonic()
    try:
        compose("up", "--detach", "--wait", spec["judge"], tag=image["tag"])
        if plan is not None:
            plan["judge_startup_seconds"] = round(time.monotonic() - start)
        print(f"{spec['judge']} up in {time.monotonic() - start:.0f} s", flush=True)
        yield
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
    session = contextlib.nullcontext()
    if "judge" in spec:
        session = judge_session(spec, image)
    with tempfile.TemporaryDirectory(prefix="labrag-eval-") as work, session:
        run_service(spec, image, Path(work), "--repro")


def check_gpu(spec: dict) -> list[str]:
    """The job's one GPU, for a GPU metric (compose reads CUDA_VISIBLE_DEVICES)."""
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


def run_service(
    spec: dict,
    image: dict,
    work: Path,
    *args: str,
    threads: int | None = None,
    prefix: str | None = None,
) -> None:
    """One run of the metric's container, with ``work`` mounted at the same path, at
    ``threads`` torch threads (default: torch_threads); its output lines get
    ``[prefix] `` (metrics running at once share the job log)."""
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
        f"OMP_NUM_THREADS={threads or torch_threads()}",
        spec["service"],
        *args,
        tag=image["tag"],
        prefix=prefix,
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
    """The judge service's image, environment and command (what determines its
    outputs, e.g. VLLM_BATCH_INVARIANT), for the artifact config; the rest of
    eval/compose.yaml is not part of it."""
    service = yaml.safe_load(COMPOSE_FILE.read_text())["services"][spec["judge"]]
    keys = ("image", "environment", "command")
    return {"service": spec["judge"], **{k: service[k] for k in keys}}


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
    """The compose service the metric runs (green-client and green-native share one
    image, so this is what tells them apart) and the git ids of what its image is built
    from (committed state of this checkout).
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


def compose(
    *args: str, tag: str, check: bool = True, prefix: str | None = None
) -> None:
    """docker compose on eval/compose.yaml, as this job's own project (jobs sharing a
    node never share a container or network); with ``prefix``, each output line starts
    with ``[prefix] ``."""
    # The container mounts the weights root the stage checked.
    env = {
        **os.environ,
        "LABRAG_COMMIT": tag,
        "LABRAG_MODELS_ROOT": str(paths.MODELS_ROOT),
    }
    project = f"labrag-eval-{os.environ.get('SLURM_JOB_ID') or os.getpid()}"
    cmd = ["docker", "compose", "-f", str(COMPOSE_FILE), "-p", project, *args]
    if prefix is None:
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, env=env, check=check)
        return
    print(f"[{prefix}] +", " ".join(cmd), flush=True)
    # text mode reads progress bars' carriage returns as line ends
    with subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ) as proc:
        for line in proc.stdout:
            print(f"[{prefix}] {line}", end="", flush=True)
    if check and proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


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
