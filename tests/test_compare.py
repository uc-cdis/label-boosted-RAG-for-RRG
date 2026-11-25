"""Compare: paired bootstrap statistics on synthetic per-study scores."""

import numpy as np
import pytest

from labrag.stages import compare


def records(scores, study_ids=None, column="x"):
    ids = study_ids or [f"patient{i}_study1" for i in range(len(scores))]
    return [
        {"study_id": s, "set": "pilot", column: v, "empty_report": False}
        for s, v in zip(ids, scores)
    ]


def table(values, column="x", metric="m"):
    rows = compare.compare(values, [column], metric)
    return {(r["statistic"], r["kind"], r["a"], r["b"]): r for r in rows}


def test_align_pairs_studies_by_id_not_order():
    a = records([1.0, 2.0, 3.0])
    b = list(reversed(records([10.0, 20.0, 30.0])))
    values = compare.align({"a": a, "b": b}, ["x"])
    assert values["a"][:, 0].tolist() == [1.0, 2.0, 3.0]
    assert values["b"][:, 0].tolist() == [10.0, 20.0, 30.0]


def test_align_refuses_different_studies():
    a = records([1.0, 2.0])
    b = records([1.0, 2.0], ["patient0_study1", "patient9_study1"])
    with pytest.raises(SystemExit, match="different studies"):
        compare.align({"a": a, "b": b}, ["x"])


def test_align_refuses_missing_scores():
    with pytest.raises(SystemExit, match="missing scores"):
        compare.align({"a": records([1.0, float("nan")]), "b": records([1, 2])}, ["x"])


def test_align_refuses_none_scores_unless_skipped():
    a, b = records([1.0, None, 3.0]), records([1.0, 2.0, 3.0])
    with pytest.raises(SystemExit, match="missing scores"):
        compare.align({"a": a, "b": b}, ["x"])
    values = compare.align({"a": a, "b": b}, ["x"], skip={("pilot", "patient1_study1")})
    assert values["a"][:, 0].tolist() == [1.0, 3.0]
    assert values["b"][:, 0].tolist() == [1.0, 3.0]  # paired: left out of both


def crimson_records(scores, failed):
    rows = records(scores, column="crimson_score")
    for r, f in zip(rows, failed):
        others = ("crimson_score", "study_id", "set", "empty_report")
        r.update({c: None if f else 0.0 for c in CRIMSON if c not in others})
        r["crimson_failed"] = f
    return rows


CRIMSON = compare.evaluate.METRICS["crimson"]["scores"]


def test_failed_studies_leave_every_condition_but_count_in_the_rate():
    recs = {
        "a": crimson_records([0.5, None, 0.1, 0.3], [0, 1, 0, 0]),
        "b": crimson_records([0.4, 0.2, None, 0.3], [0, 0, 1, 0]),
    }
    pools = {"a": ["a"], "b": ["b"]}
    rows = compare.metric_tables(recs, pools, "crimson")
    by = {(r["statistic"], r["kind"], r["a"], r["b"]): r for r in rows}
    rate = by[("crimson_failed", "condition", "a", "")]
    assert (rate["n"], rate["estimate"]) == (4, 0.25)
    score = by[("crimson_score", "difference", "a", "b")]
    assert score["n"] == 2  # studies 0 and 3: no condition failed on them
    assert score["estimate"] == pytest.approx(0.05)
    assert {r["statistic"] for r in rows} == set(CRIMSON)


def test_paired_difference_cancels_shared_noise():
    # b = a + 0.1 exactly: study-level noise is shared, so every resample's
    # difference is -0.1 although each condition's own interval is wide
    x = np.random.default_rng(0).normal(size=(200, 1))
    t = table({"a": x, "b": x + 0.1})
    d = t[("x", "difference", "a", "b")]
    assert d["estimate"] == pytest.approx(-0.1)
    assert d["ci_low"] == pytest.approx(-0.1) and d["ci_high"] == pytest.approx(-0.1)
    c = t[("x", "condition", "a", "")]
    assert c["ci_high"] - c["ci_low"] > 0.2
    assert c["ci_low"] < c["estimate"] < c["ci_high"]
    assert c["n"] == 200


def test_identical_conditions_give_zero_difference():
    x = np.random.default_rng(1).uniform(size=(50, 1))
    d = table({"a": x, "b": x.copy()})[("x", "difference", "a", "b")]
    assert (d["estimate"], d["ci_low"], d["ci_high"]) == (0.0, 0.0, 0.0)


def test_every_pair_in_the_given_order():
    x = np.zeros((10, 1))
    t = table({"a": x, "b": x + 1, "c": x + 3})
    diffs = {(r[2], r[3]): t[r]["estimate"] for r in t if r[1] == "difference"}
    assert diffs == {("a", "b"): -1.0, ("a", "c"): -3.0, ("b", "c"): -2.0}


def test_derived_statistic_from_the_mean():
    x = np.full((20, 1), 4.0)
    t = table({"a": x, "b": x / 2}, column="RadCliQ-v1")
    assert t[("1/RadCliQ-v1", "condition", "a", "")]["estimate"] == 0.25
    assert t[("1/RadCliQ-v1", "difference", "a", "b")]["estimate"] == -0.25


def test_deterministic():
    x = np.random.default_rng(2).normal(size=(100, 2))
    values = {"a": x, "b": x[::-1].copy()}
    first = compare.compare(values, ["x", "y"], "m")
    assert compare.compare(values, ["x", "y"], "m") == first


def test_blocks_do_not_change_the_resamples(monkeypatch):
    x = np.random.default_rng(3).normal(size=(30, 1))
    values = {"a": x, "b": 2 * x}
    whole = compare.resampled_means(values)
    monkeypatch.setattr(compare, "BLOCK", 7)
    blocked = compare.resampled_means(values)
    assert all(np.array_equal(whole[g], blocked[g]) for g in values)


def test_conditions_parse_pools_and_lone_artifacts():
    pools = compare.conditions(["generations/x/a", "P=x/b,generations/x/c/"])
    assert pools == {"x/a": ["x/a"], "P": ["x/b", "x/c"]}
    with pytest.raises(SystemExit, match="twice or empty"):
        compare.conditions(["P=x/b", "P=x/c"])
    with pytest.raises(SystemExit, match="twice or empty"):
        compare.conditions(["P="])


def test_pool_is_the_mean_of_its_runs_and_lists_them():
    values = {
        "r1": np.array([[1.0], [2.0], [3.0]]),
        "r2": np.array([[3.0], [4.0], [5.0]]),
        "r3": np.array([[0.0], [0.0], [0.0]]),
    }
    rows = compare.pooled(values, {"P": ["r1", "r2"], "r3": ["r3"]}, ["x"], "m")
    got = {(r["kind"], r["a"], r["b"]): r["estimate"] for r in rows}
    assert got[("condition", "P", "")] == 3.0
    assert got[("difference", "P", "r3")] == 3.0
    assert got[("run", "r1", "P")] == 2.0 and got[("run", "r2", "P")] == 4.0
    assert ("run", "r3", "r3") not in got  # a lone artifact is not a pool


def test_lone_artifacts_give_the_plain_comparison():
    rng = np.random.default_rng(0)
    values = {g: rng.normal(size=(20, 1)) for g in ("a", "b")}
    lone = compare.pooled(values, {"a": ["a"], "b": ["b"]}, ["x"], "m")
    assert lone == compare.compare(values, ["x"], "m")
