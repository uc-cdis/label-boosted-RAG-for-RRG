"""Evaluate: input formatting and the round trip through a fake metric container."""

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
