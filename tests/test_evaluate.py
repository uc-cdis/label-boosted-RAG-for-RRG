"""Evaluate: input formatting and the round trip through a fake metric container."""

import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from labrag import weights
from labrag.stages import evaluate

SPEC = evaluate.METRICS["cxr-report-metric"]


def test_format_report_collapses_whitespace_and_prefixes():
    text = "\n\nThe heart is\nnormal. \nNo effusion.  \n"
    assert evaluate.format_report(text, "findings") == (
        "Findings: The heart is normal. No effusion."
    )


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
    assert pairs["reference"].tolist() == ["Findings: ref zero", "Findings: ref one"]
    assert pairs["generated"].tolist() == ["Findings: B report.", "Findings: A report."]


@pytest.mark.parametrize("report", [None, "", " \n "])
def test_pair_reports_refuses_studies_without_report(report):
    gen = rows("fine.", report)
    refs = {"patient0_study1": "a", "patient1_study1": "b"}
    with pytest.raises(SystemExit, match="no report"):
        evaluate.pair_reports(gen, refs, "findings", "x")


def test_pair_reports_refuses_missing_reference():
    with pytest.raises(SystemExit, match="no reference"):
        evaluate.pair_reports(rows("a."), {}, "findings", "x")


def test_score_round_trip(monkeypatch):
    seen = {}

    def fake_compose(*args, tag):
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

    def fake_compose(*args, tag):
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

    def fake_compose(*args, tag):
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
    records = [{"RadCliQ-v1": 1.0}, {"RadCliQ-v1": 3.0}]
    summary = evaluate.summarize(records, ["RadCliQ-v1"])
    assert summary["1/RadCliQ-v1"] == pytest.approx(0.5)  # 1/mean, not mean of 1/x
    assert summary["mean"]["RadCliQ-v1"] == pytest.approx(2.0)


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
    out.csv (scores = study_id, rows reversed), out.json and out.prom."""
    spec = evaluate.METRICS["green"]

    def fake(*args, tag, check=True):
        calls.append((args[0], check))
        if args[0] != "run":
            return
        if fail:
            raise subprocess.CalledProcessError(1, "docker compose run")
        i = args.index("green-vllm")
        assert args[i + 1] == "score"
        for d in map(Path, args[i + 2 :]):
            df = pd.read_csv(d / "pred.csv")[["study_id"]]
            for c in spec["scores"]:
                df[c] = df["study_id"] * 1.0
            df["green_analysis"] = "text " + df["study_id"].astype(str)
            df.iloc[::-1].to_csv(d / "out.csv", index=False)
            (d / "out.json").write_text(
                json.dumps({"reports": len(df), "finish_reason": {"stop": len(df)}})
            )
            (d / "out.prom").write_text('vllm:generation_tokens_total{a="0"} 7.0\n')

    return spec, fake


def judged_sets():
    return [
        evaluate.pair_reports(
            rows(*[f"r{i}." for i in range(n)]),
            {f"patient{i}_study1": "a" for i in range(n)},
            "findings",
            "x",
        )
        for n in (3, 2)
    ]


def test_score_sets_one_run_then_down(monkeypatch):
    calls = []
    spec, fake = fake_judged_compose(calls)
    monkeypatch.setattr(evaluate, "compose", fake)
    monkeypatch.setattr(evaluate.serving, "gpu_names", lambda devices: ["H200"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    (s0, t0), (s1, t1) = evaluate.score_sets(spec, {"tag": "abc"}, judged_sets())
    # one container run for both sets; the judge's log, then the project taken down
    assert calls == [("run", True), ("logs", False), ("down", False)]
    assert [s["green_score"] for s in s0] == [0.0, 1.0, 2.0]  # mapped back by id
    assert [s["green_analysis"] for s in s1] == ["text 0", "text 1"]
    assert t0["finish_reason"] == {"stop": 3} and t1["reports"] == 2
    assert t0["gpus"] == ["H200"] and t0["run_seconds"] >= 0
    assert t0["vllm_counters"] == {"vllm:generation_tokens_total": 7.0}


def test_score_sets_takes_the_judge_down_on_failure(monkeypatch):
    calls = []
    spec, fake = fake_judged_compose(calls, fail=True)
    monkeypatch.setattr(evaluate, "compose", fake)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    with pytest.raises(subprocess.CalledProcessError):
        evaluate.score_sets(spec, {"tag": "abc"}, judged_sets())
    assert calls == [("run", True), ("logs", False), ("down", False)]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    with pytest.raises(SystemExit, match="one GPU"):
        evaluate.score_sets(spec, {"tag": "abc"}, judged_sets())


def test_judged_metric_records_the_judge_and_hf_shares_its_image():
    spec = evaluate.METRICS["green"]
    hf = evaluate.METRICS["green-hf"]
    assert evaluate.image_build(spec) == ("green-hf", "eval/green", "labrag/green")
    assert evaluate.image_source(spec) == evaluate.image_source(hf)
    assert "judge" not in evaluate.METRICS["green-hf"]
    judge = evaluate.judge_definition(spec)
    assert judge["service"] == "green-judge"
    assert "@sha256:" in judge["image"]  # pinned by digest
    assert "--generation-config=vllm" in judge["command"]
    assert "--skip-tokenizer-init" in judge["command"]


def test_compose_runs_as_the_jobs_own_project(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        evaluate.subprocess, "run", lambda cmd, env, check: seen.update(cmd=cmd)
    )
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    evaluate.compose("down", tag="abc")
    assert seen["cmd"][4:7] == ["-p", "labrag-eval-123", "down"]
