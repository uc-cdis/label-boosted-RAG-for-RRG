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


def crimson_records(scores):
    """CRIMSON records; None = the study failed (every score missing, flag set)."""
    rows = records(scores, column="crimson_score")
    for r, x in zip(rows, scores):
        others = ("crimson_score", "crimson_failed")
        r.update({c: None if x is None else 0.0 for c in CRIMSON if c not in others})
        r["crimson_failed"] = int(x is None)
    return rows


CRIMSON = compare.evaluate.METRICS["crimson"]["scores"]


def test_align_allows_missing_scores_only_where_flagged():
    rows = crimson_records([0.4, None])
    values = compare.align({"a": rows}, ["crimson_score"], failed="crimson_failed")
    assert values["a"][0, 0] == 0.4 and np.isnan(values["a"][1, 0])
    rows[1]["crimson_failed"] = 0  # missing but not flagged
    with pytest.raises(SystemExit, match="missing scores"):
        compare.align({"a": rows}, ["crimson_score"], failed="crimson_failed")
    with pytest.raises(SystemExit, match="missing scores"):  # no flag: never missing
        compare.align({"a": crimson_records([0.4, None])}, ["crimson_score"])


def test_failed_studies_each_statistic_over_what_it_can():
    # the worked example: 6 studies, A fails study 3, B study 2, C studies 4 and 5
    recs = {
        "A": crimson_records([0.40, 0.10, None, 0.30, 0.20, 0.50]),
        "B": crimson_records([0.30, None, 0.00, 0.20, 0.40, 0.50]),
        "C": crimson_records([0.50, 0.20, 0.10, None, None, 0.60]),
    }
    rows = compare.metric_tables(recs, {g: [g] for g in recs}, "crimson")
    by = {(r["statistic"], r["kind"], r["a"], r["b"]): r for r in rows}

    def got(stat, kind, a, b=""):
        r = by[(stat, kind, a, b)]
        return r["n"], pytest.approx(r["estimate"])

    # conditions: each over its own scored studies (upstream's mean)
    assert got("crimson_score", "condition", "A") == (5, 0.30)
    assert got("crimson_score", "condition", "B") == (5, 0.28)
    assert got("crimson_score", "condition", "C") == (4, 0.35)
    # differences: over the studies both scored, not the condition means' gap
    assert got("crimson_score", "difference", "A", "B") == (4, 0.0)
    assert got("crimson_score", "difference", "A", "C") == (3, -0.1)
    assert got("crimson_score", "difference", "B", "C") == (3, -0.4 / 3)
    # the failure rate over every study
    assert got("crimson_failed", "condition", "C") == (6, 2 / 6)
    assert got("crimson_failed", "difference", "A", "C") == (6, -1 / 6)
    d = by[("crimson_score", "difference", "A", "C")]
    assert d["ci_low"] <= d["estimate"] <= d["ci_high"]
    assert {r["statistic"] for r in rows} == set(CRIMSON)


def test_pool_scores_a_study_over_the_runs_that_scored_it():
    nan = np.nan
    values = {
        "s1": np.array([[0.4], [nan], [0.3]]),
        "s2": np.array([[0.2], [0.1], [nan]]),
        "b": np.array([[0.0], [0.0], [0.0]]),
    }
    rows = compare.pooled(values, {"P": ["s1", "s2"], "b": ["b"]}, ["x"], "m")
    by = {(r["kind"], r["a"], r["b"]): (r["n"], r["estimate"]) for r in rows}
    assert by[("condition", "P", "")] == (3, pytest.approx(0.7 / 3))  # .3, .1, .3
    assert by[("run", "s1", "P")] == (2, pytest.approx(0.35))  # each run its own mean


def test_derived_statistic_refuses_missing_scores():
    x = np.full((4, 1), 2.0)
    y = x.copy()
    y[0, 0] = np.nan
    with pytest.raises(SystemExit, match="needs RadCliQ-v1 for every study"):
        compare.compare({"a": x, "b": y}, ["RadCliQ-v1"], "m")


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
    x[[2, 5], 0] = np.nan  # the missing-score path too
    pairs = [("a", "b")]
    whole, whole_d = compare.resampled_means(values, pairs)
    monkeypatch.setattr(compare, "BLOCK", 7)
    blocked, blocked_d = compare.resampled_means(values, pairs)
    assert all(np.array_equal(whole[g], blocked[g], equal_nan=True) for g in values)
    assert np.array_equal(whole_d[("a", "b")][0], blocked_d[("a", "b")][0])


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
