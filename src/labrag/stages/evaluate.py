"""Stage ``evaluate``: per-study scores of generated reports, by official metric code.

Writes ``scores/<generations path>/<metric>/scores.jsonl.gz``: one record per study, in
generation order (references come from the prompt artifact). The manifest summary has
the means and, for CXR-Report-Metric, 1/RadCliQ-v1 = 1 / mean(RadCliQ-v1), as ReXrank.

Each metric is a service of eval/compose.yaml: the upstream's code and pins in its own
image, built from this checkout and tagged with the commit. ``cxr-report-metric`` runs
on CPU; ``ratescore`` and ``green`` (judge served by vLLM) share the job's GPU;
``green-native`` (upstream's HF generate) is the reference and runs alone. ``--repro``
runs GREEN's upstream reproduction test. Weights are checked (sha256) before any build.

One job runs the whole suite (``--metric`` runs one): --gres=gpu:1 at slurm/job.sh's
default 72 CPUs. The judge starts first, sizing its KV cache on the empty GPU; meanwhile
CXR-Report-Metric scores fixed chunks of studies in parallel CPU containers. Torch
threads are half the job's CPUs: 8 RaTEScore + 4 GREEN client + 4 per CXR container, so
--cpus-per-task = 2 × (12 + 4W) runs W containers; 72 gives W = 6, which keeps pace with
the GPU. Chunk size and threads change CXR scores in the last bit, so both are in its
config, and a job too small for them is refused. A failed metric doesn't stop others.

Input format (Findings only so far; deviations from ReXrank: docs/revision-plan.md §7):
no section header (a leading "Findings:" is removed), whitespace collapsed, study ids
0..n-1; an empty report is scored as "EMPTY" and stays in every mean. Empty references
are refused.
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
# Every metric scores an empty report as this text (the preprint did too, as "EMTPY").
EMPTY_REPORT = "EMPTY"
FORMAT = {
    "whitespace": "runs collapsed to one space",
    "section_header": "none; a leading 'Findings:' is removed",
    "study_ids": "0..n-1 in generation order",
    "empty_report": f"scored as the text {EMPTY_REPORT!r}",
}

# Optional keys: ``build`` (the service that builds the image, default the service),
# ``judge`` (the service serving the LLM judge), ``chunk`` (studies per container run;
# chunks run in parallel), ``texts`` (text columns kept per study).
METRICS = {
    "cxr-report-metric": {
        "service": "cxr-report-metric",
        "upstream": "external/CXR-Report-Metric",
        "gpu": False,
        # Thinner containers score more studies per CPU (24 threads: 6×4 is 1.3× 3×8).
        "threads": 4,
        # A job scores one trial's set, which must split: PadChest-GR (915) → 4 chunks,
        # MIMIC (2,347) → 10. Model loads take ~15% of a 250-study run.
        "chunk": 250,
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
        with contextlib.ExitStack() as session:
            # the judge sizes its KV cache before anything else is on the GPU
            if judged:
                try:
                    session.enter_context(
                        judge_session(judged["spec"], judged["image"], judged)
                    )
                except Exception as e:  # its metric fails; the others still run
                    failed_judge = concurrent.futures.Future()
                    failed_judge.set_exception(e)
                    futures[failed_judge] = judged["metric"]
            for plan in plans:
                if plan["spec"]["gpu"] and plan["metric"] not in futures.values():
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
    todo, built = [], []  # todo: (generations relpath, output relpath, config)
    for rel in generations:
        rel = rel.strip("/").removeprefix("generations/")
        config = {
            "stage": "evaluate",
            "generations": f"generations/{rel}",
            "metric": metric,
            # Not all of eval/compose.yaml: it also holds the other services.
            "image": image_source(spec),
            "format": FORMAT,
        }
        if "judge" in spec:
            config["judge"] = judge_definition(spec)
        if "chunk" in spec:  # changes scores in the last bit
            config["chunk_studies"] = spec["chunk"]
            config["torch_threads"] = spec["threads"]
        out = f"scores/{rel}/{metric}"
        if artifacts.is_built(out, config, overwrite):
            built.append(paths.SHARED_ROOT / out)
        else:
            todo.append((f"generations/{rel}", out, config))
    if not todo:
        return None, built

    # Check every input before building the image.
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
    """How many chunk containers run at a time: torch_threads minus the other metrics'
    threads, at the chunked metric's threads each. Threads are never lowered (CXR
    scores depend on them), so a job too small for one container of each is refused."""
    budget = torch_threads()
    for plan in plans:
        plan["threads"] = plan["spec"]["threads"]
        plan["workers"] = 1
    need = sum(p["threads"] for p in plans)
    if need > budget:
        raise SystemExit(
            f"evaluate: {', '.join(p['metric'] for p in plans)} need {need} torch "
            f"threads, i.e. --cpus-per-task >= {2 * need} (got {budget} threads)"
        )
    left = budget - sum(p["threads"] for p in plans if "chunk" not in p["spec"])
    for plan in plans:
        if "chunk" in plan["spec"]:
            size = plan["spec"]["chunk"]
            chunks = sum(-(-len(s["pairs"]) // size) for s in plan["sets"])
            plan["workers"] = max(1, min(left // plan["threads"], chunks))
        print(
            f"[{plan['metric']}] {len(plan['sets'])} sets, {plan['workers']} "
            f"container(s) at a time, {plan['threads']} threads each",
            flush=True,
        )


def score_metric(plan: dict, overwrite: bool) -> list[Path]:
    """Score every set, publishing each as its scores are in."""
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
    """Each set's scores, from ``plan["workers"]`` containers of ``chunk`` studies."""
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
    """Judge several sets (artifact name -> pairs) in one client run, one set at a time.

    Per set: scores in pair order and timing. Timing's top level is the set's own
    (server time, finish reasons, tokens, counter deltas); ``run`` is shared by the
    run's sets (count it once): overhead beyond the sets' own and the judge's startup.
    """
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
    """Start the judge and wait until it answers; on exit (SIGTERM included) copy its
    log into the job log and take the compose project down."""
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
    """One run of the metric's container, ``work`` mounted at the same path."""
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


def compose_service(name: str) -> dict:
    """A service's definition in eval/compose.yaml."""
    return yaml.safe_load(COMPOSE_FILE.read_text())["services"][name]


def image_id(ref: str) -> str | None:
    """The local image id of ``ref`` (None if this node doesn't have it)."""
    out = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", ref],
        capture_output=True,
        text=True,
    )
    return out.stdout.strip() if out.returncode == 0 else None


def judge_definition(spec: dict) -> dict:
    """The judge's image, environment and command: what determines its outputs."""
    service = compose_service(spec["judge"])
    keys = ("image", "environment", "command")
    return {"service": spec["judge"], **{k: service[k] for k in keys}}


def judge_image_record(spec: dict) -> dict:
    """The judge's pinned image, after checking this node has it (jobs never pull)."""
    ref = judge_definition(spec)["image"]
    pulled = image_id(ref)
    if pulled is None:
        raise SystemExit(
            f"judge image {ref} not on this node; pre-pull it with\n  docker pull {ref}"
        )
    return {"ref": ref, "image_id": pulled}


def torch_threads() -> int:
    """Half the job's CPUs (--cpus-per-task; Slurm doesn't confine jobs). torch 1.6
    stalls on preempted threads: threads == CPUs ran RadGraph ~7x slower."""
    ncpu = int(os.environ.get("SLURM_CPUS_PER_TASK") or len(os.sched_getaffinity(0)))
    return max(1, ncpu // 2)


def image_source(spec: dict) -> dict:
    """The service that runs (green-client and green-native share an image) and the
    git ids its image is built from."""
    _, context, _ = image_build(spec)
    return {
        "service": spec["service"],
        "upstream_commit": _git("rev-parse", f"HEAD:{spec['upstream']}"),
        "definition_tree": _git("rev-parse", f"HEAD:{context}"),
    }


def image_build(spec: dict) -> tuple[str, str, str]:
    """The service that builds the image, its build context and untagged image name."""
    service = spec.get("build", spec["service"])
    definition = compose_service(service)
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
    built = image_id(ref)
    if built is None:
        raise SystemExit(f"{ref}: not found after the build")
    return {
        "ref": ref,
        "tag": tag,
        "image_id": built,
        "compose_blob": _git("rev-parse", "HEAD:eval/compose.yaml"),
    }


def compose(
    *args: str, tag: str, check: bool = True, prefix: str | None = None
) -> None:
    """docker compose as this job's own project; output lines get ``[prefix] ``."""
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
