"""Evaluate: input formatting and the round trip through a fake metric container."""

import contextlib
import json
import subprocess
import threading
from pathlib import Path

import pandas as pd
import pytest

from labrag import weights
from labrag.stages import evaluate

SPEC = evaluate.METRICS["cxr-report-metric"]


def test_format_report_collapses_whitespace_without_header():
    text = "\n\nThe heart is\nnormal. \nNo effusion.  \n"
    assert evaluate.format_report(text) == "The heart is normal. No effusion."


@pytest.mark.parametrize(
    "text",
    ["Findings: Heart normal.", "\nFINDINGS:\nHeart normal.", "findings:Heart normal."],
)
def test_format_report_removes_leading_findings_header(text):
    assert evaluate.format_report(text) == "Heart normal."


def test_format_report_keeps_findings_mid_text():
    text = "No new findings: heart normal. Findings: stable."
    assert evaluate.format_report(text) == text


def rows(*reports):
    return [
        {"study_id": f"patient{i}_study1", "set": "pilot", "report": r}
        for i, r in enumerate(reports)
    ]


def test_pair_reports_integer_ids_in_generation_order():
    gen = rows("B report.", "A report.")
    refs = {"patient0_study1": "ref\nzero", "patient1_study1": "ref one"}
    pairs = evaluate.pair_reports(gen, refs, "findings", "x")
    assert pairs["study_id"].tolist() == [0, 1]
    assert pairs["reference"].tolist() == ["ref zero", "ref one"]
    assert pairs["generated"].tolist() == ["B report.", "A report."]


def test_pair_reports_refuses_undecided_sections():
    with pytest.raises(SystemExit, match="not decided"):
        evaluate.pair_reports(rows("a."), {"patient0_study1": "b"}, "impression", "x")


@pytest.mark.parametrize("report", [None, "", " \n ", "Findings: "])
def test_pair_reports_scores_empty_reports_as_placeholder(report):
    gen = rows("fine.", report)
    refs = {"patient0_study1": "a", "patient1_study1": "b"}
    pairs = evaluate.pair_reports(gen, refs, "findings", "x")
    assert pairs["generated"].tolist() == ["fine.", "EMPTY"]
    assert pairs["empty_report"].tolist() == [False, True]


@pytest.mark.parametrize(
    "text",
    [
        "Findings: Lungs clear. Impression: Normal.",
        "FINDINGS:\nLungs  clear.\n\nIMPRESSION:\nNormal.",
        "**Findings:** Lungs clear. **Impression:** Normal.",
        "**Findings**: Lungs clear. **Impression**: Normal.",
    ],
)
def test_split_joint_takes_both_headers(text):
    assert evaluate.split_joint(text) == ("Lungs clear.", "Normal.")


@pytest.mark.parametrize(
    "text",
    [
        "Lungs clear. Impression: Normal.",  # no Findings header
        "Findings: Lungs clear. Normal.",  # no Impression header
        "Impression: Normal. Findings: Lungs clear.",  # wrong order
        "Findings: Lungs clear. Impression: Normal. Impression: Normal.",  # twice
        "Findings: Impression: Normal.",  # empty Findings
        "Findings: Lungs clear. Impression:",  # empty Impression
        "Report. Findings: Lungs clear. Impression: Normal.",  # text before Findings
    ],
)
def test_split_joint_is_strict(text):
    assert evaluate.split_joint(text) is None


def test_pair_reports_joint_scores_content_and_flags_unparsed():
    gen = rows(
        "Findings: Lungs\nclear. Impression: Normal.",
        "Lungs clear. Impression: Normal.",
        "",
    )
    refs = {f"patient{i}_study1": "Lungs  clear. No acute process." for i in range(3)}
    pairs = evaluate.pair_reports(gen, refs, "findings+impression", "x")
    assert pairs["reference"].tolist() == ["Lungs clear. No acute process."] * 3
    assert pairs["generated"].tolist() == [
        "Lungs clear. Normal.",
        "Lungs clear. Normal.",  # unparsed: every header removed, still scored
        "EMPTY",
    ]
    assert pairs["unparsed"].tolist() == [False, True, False]
    assert pairs["empty_report"].tolist() == [False, False, True]


def test_findings_format_is_unchanged_and_joint_has_its_own():
    assert evaluate.FORMATS["findings"] is evaluate.FORMAT
    assert evaluate.FORMAT["section_header"] == "none; a leading 'Findings:' is removed"
    assert evaluate.FORMATS["findings+impression"] != evaluate.FORMAT


def test_pair_reports_refuses_empty_reference():
    with pytest.raises(SystemExit, match="empty reference"):
        evaluate.pair_reports(rows("a."), {"patient0_study1": " \n"}, "findings", "x")


def test_pair_reports_refuses_missing_reference():
    with pytest.raises(SystemExit, match="no reference"):
        evaluate.pair_reports(rows("a."), {}, "findings", "x")


def test_score_round_trip(monkeypatch):
    seen = {}

    def fake_compose(*args, tag, prefix=None):
        seen["args"], seen["tag"] = args, tag
        gt, pred, out = (args[-3], args[-2], args[-1])
        g, p = pd.read_csv(gt), pd.read_csv(pred)
        assert g["study_id"].tolist() == p["study_id"].tolist()
        # upstream writes rows sorted by study_id, with extra columns; reverse to
        # check the stage maps scores back by id, not by position
        df = p.iloc[::-1].copy()
        for i, c in enumerate(SPEC["scores"]):
            df[c] = df["study_id"] + i / 10
        df.to_csv(out, index=False)
        Path(out).with_suffix(".json").write_text('{"seconds": 1.0}')

    monkeypatch.setattr(evaluate, "compose", fake_compose)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "32")
    pairs = evaluate.pair_reports(
        rows("x.", "y.", "z."),
        {f"patient{i}_study1": "r" for i in range(3)},
        "findings",
        "x",
    )
    scores, timing = evaluate.score(SPEC, {"tag": "abc"}, pairs)
    assert [s["bleu_score"] for s in scores] == [0.0, 1.0, 2.0]
    assert scores[2]["RadCliQ-v1"] == pytest.approx(2.5)
    assert timing == {"seconds": 1.0}
    assert seen["tag"] == "abc"
    assert "OMP_NUM_THREADS=16" in seen["args"]
    assert seen["args"][:2] == ("run", "--rm")


def test_gpu_metric_needs_exactly_one_gpu(monkeypatch):
    spec = evaluate.METRICS["ratescore"]

    def fake_compose(*args, tag, prefix=None):
        out = args[-1]
        pd.read_csv(args[-2]).assign(ratescore=0.5).to_csv(out, index=False)
        Path(out).with_suffix(".json").write_text("{}")

    monkeypatch.setattr(evaluate, "compose", fake_compose)
    pairs = evaluate.pair_reports(rows("x."), {"patient0_study1": "r"}, "findings", "x")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    scores, _ = evaluate.score(spec, {"tag": "abc"}, pairs)
    assert scores == [{"ratescore": 0.5}]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    with pytest.raises(SystemExit, match="one GPU"):
        evaluate.score(spec, {"tag": "abc"}, pairs)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    with pytest.raises(SystemExit, match="CUDA_VISIBLE_DEVICES"):
        evaluate.score(spec, {"tag": "abc"}, pairs)


def test_text_columns_kept_as_written(monkeypatch):
    spec = {**evaluate.METRICS["ratescore"], "gpu": False, "texts": ["analysis"]}
    texts = ['two\nlines, "quoted"', "", "NA"]

    def fake_compose(*args, tag, prefix=None):
        out = args[-1]
        df = pd.read_csv(args[-2]).assign(ratescore=0.5, analysis=texts)
        df.to_csv(out, index=False)
        Path(out).with_suffix(".json").write_text("{}")

    monkeypatch.setattr(evaluate, "compose", fake_compose)
    pairs = evaluate.pair_reports(
        rows("x.", "y.", "z."),
        {f"patient{i}_study1": "r" for i in range(3)},
        "findings",
        "x",
    )
    scores, _ = evaluate.score(spec, {"tag": "abc"}, pairs)
    assert scores == [{"ratescore": 0.5, "analysis": t} for t in texts]


def test_summarize_radcliq_is_corpus_level():
    records = [
        {"empty_report": False, "RadCliQ-v1": 1.0},
        {"empty_report": True, "RadCliQ-v1": 3.0},
    ]
    summary = evaluate.summarize(records, {"scores": ["RadCliQ-v1"]})
    assert summary["empty_reports"] == 1
    assert summary["1/RadCliQ-v1"] == pytest.approx(0.5)  # 1/mean, not mean of 1/x
    assert summary["mean"]["RadCliQ-v1"] == pytest.approx(2.0)


def crimson_out(failed, scored_where_failed=False):
    """A fake crimson out.csv writer: study i scores i/10 unless it failed."""
    spec = evaluate.METRICS["crimson"]

    def fake_compose(*args, tag, prefix=None):
        out = args[-1]
        df = pd.read_csv(args[-2])[["study_id"]]
        for c in spec["scores"]:
            df[c] = [
                None if f and not scored_where_failed else i / 10
                for i, f in enumerate(failed)
            ]
        df["crimson_failed"] = failed
        df["crimson_response"] = ["text"] * len(df)
        df.to_csv(out, index=False)
        Path(out).with_suffix(".json").write_text("{}")

    return {**spec, "gpu": False}, fake_compose


def three_pairs(n=3):
    return evaluate.pair_reports(
        rows(*[f"r{i}." for i in range(n)]),
        {f"patient{i}_study1": "a" for i in range(n)},
        "findings",
        "x",
    )


def test_failed_studies_may_lack_scores_and_are_counted(monkeypatch):
    spec, fake = crimson_out([0, 1, 0])
    monkeypatch.setattr(evaluate, "compose", fake)
    scores, _ = evaluate.score(spec, {"tag": "abc"}, three_pairs())
    assert scores[1]["crimson_score"] is None and scores[1]["crimson_failed"] == 1
    assert scores[2]["crimson_score"] == pytest.approx(0.2)
    records = [{"empty_report": False, **s} for s in scores]
    json.dumps(records, allow_nan=False)  # null, not NaN, in the artifact
    summary = evaluate.summarize(records, spec)
    assert summary["failed"] == 1
    assert summary["mean"]["crimson_score"] == pytest.approx(0.1)  # scored studies
    assert summary["mean"]["crimson_failed"] == pytest.approx(1 / 3)


def test_missing_scores_without_failure_are_refused(monkeypatch):
    spec, fake = crimson_out([0, 1])
    monkeypatch.setattr(evaluate, "compose", fake)
    spec = {k: v for k, v in spec.items() if k != "failed"}  # no failure flag
    with pytest.raises(SystemExit, match="missing scores"):
        evaluate.score(spec, {"tag": "abc"}, three_pairs(2))


def test_torch_threads_half_the_job(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    assert evaluate.torch_threads() == 1
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "32")
    assert evaluate.torch_threads() == 16


@pytest.mark.parametrize("metric", sorted(evaluate.METRICS))
def test_metric_weights_are_pinned(metric):
    config = weights.load_config()
    for name in evaluate.METRICS[metric]["weights"]:
        assert config[name]["files"], name


@pytest.mark.parametrize("metric", sorted(evaluate.METRICS))
def test_metric_image_is_built_by_a_compose_service(metric):
    service, context, name = evaluate.image_build(evaluate.METRICS[metric])
    assert (evaluate.paths.REPO_ROOT / context / "Dockerfile").is_file()
    assert name == f"labrag/{Path(context).name}"


def fake_judged_compose(calls, fail=False):
    """A fake compose for a judged metric: `run ... score <dirs>` writes each set's
    out.csv (scores = study_id, rows reversed), out.json (1 s per study) and the
    server's metrics before and after it (7 generated tokens per study)."""
    spec = evaluate.METRICS["green"]

    def fake(*args, tag, check=True, prefix=None):
        calls.append((args[0], check))
        if args[0] != "run":
            return
        if fail:
            raise subprocess.CalledProcessError(1, "docker compose run")
        i = args.index("green-client")
        assert args[i + 1] == "score"
        tokens = 100.0  # the server's counter before the first set
        for d in map(Path, args[i + 2 :]):
            df = pd.read_csv(d / "pred.csv")[["study_id"]]
            for c in spec["scores"]:
                df[c] = df["study_id"] * 1.0
            df["green_analysis"] = "text " + df["study_id"].astype(str)
            df.iloc[::-1].to_csv(d / "out.csv", index=False)
            (d / "out.json").write_text(
                json.dumps(
                    {
                        "reports": len(df),
                        "seconds": float(len(df)),
                        "finish_reason": {"stop": len(df)},
                    }
                )
            )
            for when, value in (("start", tokens), ("end", tokens + 7 * len(df))):
                (d / f"out.{when}.prom").write_text(
                    f'vllm:generation_tokens_total{{a="0"}} {value}\n'
                )
            tokens += 7 * len(df)

    return spec, fake


def judged_sets():
    return {
        f"scores/set{n}/green": evaluate.pair_reports(
            rows(*[f"r{i}." for i in range(n)]),
            {f"patient{i}_study1": "a" for i in range(n)},
            "findings",
            "x",
        )
        for n in (3, 2)
    }


def test_score_sets_one_run(monkeypatch):
    calls = []
    spec, fake = fake_judged_compose(calls)
    monkeypatch.setattr(evaluate, "compose", fake)
    monkeypatch.setattr(evaluate.serving, "gpu_names", lambda devices: ["H200"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    plan = {"judge_startup_seconds": 40}
    (s0, t0), (s1, t1) = evaluate.score_sets(spec, {"tag": "abc"}, judged_sets(), plan)
    assert calls == [("run", True)]  # one container run for both sets
    assert [s["green_score"] for s in s0] == [0.0, 1.0, 2.0]  # mapped back by id
    assert [s["green_analysis"] for s in s1] == ["text 0", "text 1"]
    assert t0["finish_reason"] == {"stop": 3} and t1["reports"] == 2
    assert t0["gpus"] == ["H200"]
    # each set's own counters: the change over the set, not the server's totals
    assert t0["vllm_counters"] == {"vllm:generation_tokens_total": 21.0}
    assert t1["vllm_counters"] == {"vllm:generation_tokens_total": 14.0}
    # shared by the run: the same block in both, naming both sets
    assert t0["run"] == t1["run"]
    assert t0["run"]["sets"] == ["scores/set3/green", "scores/set2/green"]
    assert isinstance(t0["run"]["overhead_seconds"], int)
    assert t0["run"]["judge_startup_seconds"] == 40
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    with pytest.raises(SystemExit, match="one GPU"):
        evaluate.score_sets(spec, {"tag": "abc"}, judged_sets())


@pytest.mark.parametrize("fail", [False, True])
def test_judge_session_always_removes_its_judge_only(monkeypatch, fail):
    calls = []
    monkeypatch.setattr(
        evaluate, "compose", lambda *a, tag, check=True: calls.append(a[:2])
    )
    monkeypatch.setattr(evaluate.serving, "raise_on_sigterm", lambda: None)
    plan = {}
    raises = pytest.raises(RuntimeError) if fail else contextlib.nullcontext()
    session = evaluate.judge_session(evaluate.METRICS["green"], {"tag": "abc"}, plan)
    with raises, session:
        assert calls == [("up", "--detach")]  # up (and healthy) before the body
        if fail:
            raise RuntimeError
    # not `down`: that would also remove the job's other running containers
    assert calls[1:] == [("logs", "--no-color"), ("rm", "--stop")]
    assert isinstance(plan["judge_startup_seconds"], int)


def test_chunks_score_each_set_in_order(monkeypatch):
    seen = []

    def fake_score(spec, image, pairs, threads, prefix):
        seen.append((pairs["study_id"].tolist(), threads))
        scores = [{"bleu_score": float(i)} for i in pairs["study_id"]]
        return scores, {"seconds": 1.5, "torch_threads": threads}

    monkeypatch.setattr(evaluate, "score", fake_score)
    spec = {**SPEC, "chunk": 2}
    sets = [{"pairs": pd.DataFrame({"study_id": range(n)})} for n in (5, 2)]
    plan = {"workers": 3, "threads": 8, "metric": "cxr-report-metric"}
    (s0, t0), (s1, t1) = list(evaluate.score_chunks(spec, {}, sets, plan))
    assert sorted(seen) == [([0, 1], 8), ([0, 1], 8), ([2, 3], 8), ([4], 8)]
    assert [x["bleu_score"] for x in s0] == [0, 1, 2, 3, 4]  # chunks rejoined in order
    assert [x["bleu_score"] for x in s1] == [0, 1]
    assert t0 == {
        "reports": 5,
        "seconds": 4.5,
        "chunks": 3,
        "torch_threads": 8,
        "parallel_containers": 3,
    }
    assert t1["chunks"] == 1


def plans_of(*metrics, studies=5000):
    sets = [{"pairs": pd.DataFrame({"study_id": range(studies)})}]
    return [{"metric": m, "spec": evaluate.METRICS[m], "sets": sets} for m in metrics]


def test_plan_threads_gives_the_rest_to_the_chunked_metric(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "72")  # 36 threads in all
    plans = plans_of(*evaluate.SUITE)
    evaluate.plan_threads(plans)
    by = {p["metric"]: (p["threads"], p["workers"]) for p in plans}
    # ratescore 8 + one judge client 4 at a time (green, then crimson);
    # cxr-report-metric: (36 - 12) // 4 containers of 4
    assert by == {
        "ratescore": (8, 1),
        "green": (4, 1),
        "crimson": (4, 1),
        "cxr-report-metric": (4, 6),
    }
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "30")  # 15 < 8 + 4 + 4: refused
    with pytest.raises(SystemExit, match="need 16 torch threads.*>= 32"):
        evaluate.plan_threads(plans_of(*evaluate.SUITE))
    monkeypatch.setenv(
        "SLURM_CPUS_PER_TASK", "32"
    )  # one container, never fewer threads
    plans = plans_of(*evaluate.SUITE)
    evaluate.plan_threads(plans)
    assert [(p["threads"], p["workers"]) for p in plans] == [(4, 1), (8, 1), (4, 1)] + [
        (4, 1)
    ]
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "96")  # 3 sets of 200: 3 chunks
    plans = plans_of(*evaluate.SUITE, studies=200)
    plans[0]["sets"] = plans[0]["sets"] * 3
    evaluate.plan_threads(plans)
    assert plans[0]["workers"] == 3


def fake_run_env(monkeypatch, session, score_metric):
    """run() with fake prepare, images, judge sessions and scoring; returns the compose
    calls made by run() itself."""
    calls = []
    monkeypatch.setattr(
        evaluate,
        "prepare",
        lambda m, g, o: ({"metric": m, "spec": evaluate.METRICS[m], "sets": []}, []),
    )
    monkeypatch.setattr(evaluate, "check_gpu", lambda spec: [])
    monkeypatch.setattr(evaluate, "build_image", lambda spec: {"tag": "abc"})
    monkeypatch.setattr(evaluate, "judge_session", session)
    monkeypatch.setattr(evaluate, "score_metric", score_metric)
    monkeypatch.setattr(
        evaluate, "compose", lambda *a, tag, check=True: calls.append(a[0])
    )
    monkeypatch.setattr(evaluate.serving, "raise_on_sigterm", lambda: None)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "64")
    return calls


def test_run_holds_the_gpu_one_judge_at_a_time_and_reports_failures(monkeypatch):
    events, lock = [], threading.Lock()
    judge = {"up": None}

    @contextlib.contextmanager
    def fake_session(spec, image, plan):
        with lock:
            judge["up"] = spec["judge"]
            events.append(("up", spec["judge"]))
        yield
        with lock:
            judge["up"] = None
            events.append(("down", spec["judge"]))

    def fake_score_metric(plan, overwrite):
        with lock:
            events.append((plan["metric"], judge["up"]))
        if plan["metric"] == "ratescore":
            raise SystemExit("boom")
        return [plan["metric"]]

    calls = fake_run_env(monkeypatch, fake_session, fake_score_metric)
    with pytest.raises(SystemExit, match="failed metrics: ratescore"):
        evaluate.run(["g"])
    assert ("cxr-report-metric", None) in events  # doesn't wait for a judge
    green_up, green_down = events.index(("up", "green-vllm")), events.index(
        ("down", "green-vllm")
    )
    # GREEN's judge first, RaTEScore beside it, then CRIMSON's judge alone
    assert green_up < events.index(("green", "green-vllm")) < green_down
    assert green_up < events.index(("ratescore", "green-vllm")) < green_down
    assert green_down < events.index(("up", "crimson-vllm"))
    assert ("crimson", "crimson-vllm") in events
    assert events[-1] == ("down", "crimson-vllm")
    assert calls == ["down"]  # the project, once, after every metric


def test_judge_that_fails_to_start_fails_only_its_metric(monkeypatch):
    scored = []

    @contextlib.contextmanager
    def session(spec, image, plan):
        if spec["judge"] == "green-vllm":
            raise subprocess.CalledProcessError(1, ["docker", "compose", "up"])
        yield

    fake_run_env(
        monkeypatch, session, lambda plan, o: scored.append(plan["metric"]) or []
    )
    with pytest.raises(SystemExit, match="failed metrics: green$"):
        evaluate.run(["g"])
    assert sorted(scored) == ["crimson", "cxr-report-metric", "ratescore"]


@pytest.mark.parametrize("metric", ["crimson-native", "cxr-report-metric"])
def test_cancelled_run_without_a_judge_still_removes_its_containers(
    monkeypatch, metric
):
    """SIGTERM mid-score: run() returns without waiting for the scoring thread and
    takes the project down (a native or CPU run's container would outlive the job)."""
    release = threading.Event()

    def blocked(plan, overwrite):  # a container that runs for hours
        release.wait(10)
        return []

    calls = fake_run_env(monkeypatch, None, blocked)
    monkeypatch.setattr(
        evaluate.concurrent.futures,
        "wait",
        lambda *a, **k: (_ for _ in ()).throw(SystemExit(143)),  # SIGTERM arrives
    )
    try:
        with pytest.raises(SystemExit):
            evaluate.run(["g"], [metric])
        assert calls == ["down"]
    finally:
        release.set()


@pytest.mark.parametrize("metric", ["green-native", "crimson-native"])
def test_native_judges_run_alone(metric):
    with pytest.raises(SystemExit, match="alone"):
        evaluate.run(["g"], [metric, "ratescore"])


def test_gpu_phases_one_judge_each_rest_with_the_first():
    names = lambda phases: [[p["metric"] for p in ph] for ph in phases]  # noqa: E731
    assert names(evaluate.gpu_phases(plans_of(*evaluate.SUITE))) == [
        ["green", "ratescore"],
        ["crimson"],
    ]
    assert names(evaluate.gpu_phases(plans_of("ratescore"))) == [["ratescore"]]
    assert evaluate.gpu_phases(plans_of("cxr-report-metric")) == []


def test_judged_metric_records_the_judge_and_native_shares_its_image():
    spec = evaluate.METRICS["green"]
    native = evaluate.METRICS["green-native"]
    assert evaluate.image_build(spec) == ("green-native", "eval/green", "labrag/green")
    source, native_source = evaluate.image_source(spec), evaluate.image_source(native)
    assert source.pop("service") == "green-client"
    assert native_source.pop("service") == "green-native"
    assert source == native_source  # same image
    assert "judge" not in evaluate.METRICS["green-native"]
    judge = evaluate.judge_definition(spec)
    assert judge["service"] == "green-vllm"
    assert "@sha256:" in judge["image"]  # pinned by digest
    assert "--generation-config=vllm" in judge["command"]
    assert "--skip-tokenizer-init" in judge["command"]
    assert "VLLM_BATCH_INVARIANT=1" in judge["environment"]
    assert "TORCHINDUCTOR_DETERMINISTIC=1" in judge["environment"]


def test_crimson_client_and_native_share_an_image_and_the_judge_is_bf16():
    spec, native = evaluate.METRICS["crimson"], evaluate.METRICS["crimson-native"]
    assert evaluate.image_build(spec) == (
        "crimson-native",
        "eval/crimson",
        "labrag/crimson",
    )
    assert evaluate.image_build(native) == evaluate.image_build(spec)
    assert "judge" not in native and native["service"] == "crimson-native"
    judge = evaluate.judge_definition(spec)
    green = evaluate.judge_definition(evaluate.METRICS["green"])
    assert judge["service"] == "crimson-vllm"
    assert judge["image"] == green["image"]  # one pinned vLLM for both judges
    assert judge["environment"] == green["environment"]  # batch-invariant, determin.
    assert "--dtype=bfloat16" in judge["command"]
    assert "--skip-tokenizer-init" in judge["command"]


def test_chunked_metric_config_records_its_container_layout(monkeypatch):
    configs = []
    monkeypatch.setattr(
        evaluate.artifacts,
        "is_built",
        lambda out, config, overwrite: configs.append(config) or True,
    )
    monkeypatch.setattr(evaluate, "scoring_format", lambda rel: evaluate.FORMAT)
    plan, built = evaluate.prepare("cxr-report-metric", ["g"], False)
    assert plan is None and len(built) == 1
    assert configs[0]["chunk_studies"] == 250
    assert configs[0]["torch_threads"] == 4
    evaluate.prepare("ratescore", ["g"], False)
    assert "torch_threads" not in configs[1]


def test_compose_runs_as_the_jobs_own_project(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        evaluate.subprocess, "run", lambda cmd, env, check: seen.update(cmd=cmd)
    )
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    evaluate.compose("down", tag="abc")
    assert seen["cmd"][4:7] == ["-p", "labrag-eval-123", "down"]


def test_build_image_holds_the_node_lock(tmp_path, monkeypatch):
    import fcntl

    from labrag import paths, serving

    monkeypatch.setattr(paths, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(
        evaluate.artifacts, "git_commit", lambda: {"sha": "a" * 40, "dirty": False}
    )
    held = []

    def build(*args, tag):
        with open(tmp_path / "docker-build.lock") as other:
            try:
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held.append(False)
            except BlockingIOError:
                held.append(True)

    monkeypatch.setattr(evaluate, "compose", build)
    monkeypatch.setattr(serving, "image_id", lambda ref: "sha256:x")
    image = evaluate.build_image(evaluate.METRICS["ratescore"])
    assert held == [True]
    assert image["image_id"] == "sha256:x"
