"""Stage ``compare``: means with 95% CIs per condition, and paired differences.

Reads the scores artifacts of two or more conditions scored on the same studies; writes
``comparisons/<out>/comparison.csv``: per score column, each condition's mean and each
pair's difference (a − b, in the order given), with a percentile interval from a paired
bootstrap over studies (docs/revision-plan.md §3). Also ``generations/empty_report``
(and ``generations/unparsed`` for Findings+Impression) and 1/RadCliQ-v1 = 1 / mean(RadCliQ-v1).

A condition is one generations artifact, or a pool ``NAME=run1,run2,...`` (e.g.
classifier seeds × sampling seeds): each study's scores averaged over the pool's runs,
so its mean is the mean of the runs' means. Intervals still resample studies only, not
the runs. Each pooled run's own mean is a ``run`` row (``b`` = its pool).

Paired: a drawn study counts in every condition, so shared noise cancels in the
differences. Fixed seed and single-threaded numpy means, so intervals don't depend on
the machine. Empty reports are scored as "EMPTY". Scores are missing only where a
metric failed on a study (CRIMSON): that study is left out of the metric's scores in
every condition (``n`` drops), and the failure flag's mean covers every study.
"""

import csv
import gzip
import itertools
import json
from pathlib import Path

import numpy as np

from labrag import artifacts, paths
from labrag.stages import evaluate

RESAMPLES = 10_000
SEED = 20261115
CI = 0.95
# Resamples drawn at a time (memory: BLOCK × n indices). The draws are the same at any
# block size (tests/test_compare.py), so it is not part of the config.
BLOCK = 500
METHOD = "paired bootstrap over studies; percentile interval"
# Statistics derived from a column's mean (name, column, function of the mean).
DERIVED = [("1/RadCliQ-v1", "RadCliQ-v1", lambda m: 1.0 / m)]
FIELDS = ["metric", "statistic", "kind", "a", "b", "n", "estimate", "ci_low",
          "ci_high"]  # fmt: skip


def run(
    generations: list[str],
    out: str,
    metrics: list[str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Compare the conditions ``generations`` on ``metrics`` (default: the suite)."""
    metrics = metrics or evaluate.SUITE
    pools = conditions(generations)
    runs = [g for members in pools.values() for g in members]
    if len(pools) < 2:
        raise SystemExit("compare: give at least two conditions")
    if len(set(runs)) < len(runs):
        raise SystemExit("compare: a generations artifact is given twice")
    relpath = f"comparisons/{out.strip('/').removeprefix('comparisons/')}"
    listed = [
        f"generations/{m[0]}" if c == m[0] else
        {"name": c, "generations": [f"generations/{g}" for g in m]}
        for c, m in pools.items()
    ]  # fmt: skip
    config = {
        "stage": "compare",
        "generations": listed,
        "metrics": metrics,
        "method": METHOD,
        "resamples": RESAMPLES,
        "seed": SEED,
        "ci": CI,
    }
    if artifacts.is_built(relpath, config, overwrite):
        return paths.SHARED_ROOT / relpath

    inputs, tables = {}, []
    for metric in metrics:
        records = {}
        for g in runs:
            score_dir = paths.SHARED_ROOT / "scores" / g / metric
            inputs[f"{g}/{metric}"] = artifacts.artifact_record(score_dir)
            records[g] = _read_jsonl(score_dir / "scores.jsonl.gz")
        if not tables:  # per-study flags, the same in every metric: from the first
            first = records[runs[0]][0]
            flags = [c for c in evaluate.FLAGS if c in first]
            tables += pooled(align(records, flags), pools, flags, "generations")
        tables += metric_tables(records, pools, metric)

    def build(dest: Path) -> dict:
        with open(dest / "comparison.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
            w.writeheader()
            w.writerows(tables)
        print_table(tables, list(pools))
        return {"conditions": {str(i): c for i, c in enumerate(pools, 1)}}

    return artifacts.publish(relpath, config, inputs, build, overwrite)


def conditions(generations: list[str]) -> dict[str, list[str]]:
    """Condition name -> its generations artifacts (a lone artifact names itself)."""
    pools = {}
    for item in generations:
        name, sep, members = item.partition("=")
        runs = members.split(",") if sep else [name]
        runs = [g.strip("/").removeprefix("generations/") for g in runs]
        name = name if sep else runs[0]
        if name in pools or not all(runs):
            raise SystemExit(f"compare: condition {name!r} given twice or empty")
        pools[name] = runs
    return pools


def metric_tables(records: dict[str, list[dict]], pools, metric: str) -> list:
    """One metric's rows. With a failure flag: its rate over every study, and the
    scores over the studies no condition failed on."""
    spec = evaluate.METRICS[metric]
    columns = spec["scores"]
    if "failed" not in spec:
        return pooled(align(records, columns), pools, columns, metric)
    flag = spec["failed"]
    columns = [c for c in columns if c != flag]
    skip = {
        (r["set"], r["study_id"]) for rows in records.values() for r in rows if r[flag]
    }
    return pooled(align(records, [flag]), pools, [flag], metric) + pooled(
        align(records, columns, skip), pools, columns, metric
    )


def pooled(values, pools, columns: list[str], metric: str) -> list:
    """``compare`` over the pools (per-study means over their runs), then each pooled
    run's own mean as a ``run`` row."""
    means = {c: np.mean([values[g] for g in m], axis=0) for c, m in pools.items()}
    rows = compare(means, columns, metric)
    for c, members in pools.items():
        if len(members) > 1:
            own = compare({g: values[g] for g in members}, columns, metric)
            rows += [
                r | {"kind": "run", "b": c} for r in own if r["kind"] == "condition"
            ]
    return rows


def align(
    records: dict[str, list[dict]], columns: list[str], skip: set | None = None
) -> dict[str, np.ndarray]:
    """Per condition, an (n, columns) array with the studies in one shared order,
    leaving out the (set, study_id) keys in ``skip``.

    Refuses conditions not scored on the same studies of the same set.
    """
    keyed = {}
    for g, rows in records.items():
        keys = [(r["set"], r["study_id"]) for r in rows]
        if len(set(keys)) < len(keys):
            raise SystemExit(f"compare: {g} scores a study twice")
        keyed[g] = dict(zip(keys, rows))
    first, *rest = keyed
    order = sorted(keyed[first])
    for g in rest:
        if set(keyed[g]) != set(order):
            raise SystemExit(f"compare: {g} and {first} score different studies")
    order = [k for k in order if k not in (skip or ())]
    values = {}
    for g, by_key in keyed.items():
        v = np.array(
            [[float(by_key[k][c] if by_key[k][c] is not None else np.nan)
              for c in columns] for k in order],
            dtype=np.float64,
        )  # fmt: skip
        if np.isnan(v).any():
            raise SystemExit(f"compare: {g} has missing scores")
        values[g] = v
    return values


def resampled_means(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per condition, the (RESAMPLES, columns) column means of the paired resamples."""
    n = next(iter(values.values())).shape[0]
    rng = np.random.default_rng(SEED)
    means = {g: np.empty((RESAMPLES, v.shape[1])) for g, v in values.items()}
    for start in range(0, RESAMPLES, BLOCK):
        stop = min(start + BLOCK, RESAMPLES)
        idx = rng.integers(0, n, size=(stop - start, n))  # shared by every condition
        for g, v in values.items():
            for c in range(v.shape[1]):
                means[g][start:stop, c] = v[:, c][idx].mean(axis=1)
    return means


def compare(values: dict[str, np.ndarray], columns: list[str], metric: str) -> list:
    """Rows of the comparison table for one metric (see FIELDS)."""
    n = next(iter(values.values())).shape[0]
    boot = resampled_means(values)
    point = {g: v.mean(axis=0) for g, v in values.items()}
    stats = [(c, i, lambda m: m) for i, c in enumerate(columns)]
    stats += [
        (name, columns.index(col), f) for name, col, f in DERIVED if col in columns
    ]
    lo, hi = (1 - CI) / 2, 1 - (1 - CI) / 2
    rows = []

    def row(stat, kind, a, b, estimate, samples):
        low, high = np.quantile(samples, [lo, hi])
        rows.append({"metric": metric, "statistic": stat, "kind": kind, "a": a,
                     "b": b, "n": n, "estimate": float(estimate),
                     "ci_low": float(low), "ci_high": float(high)})  # fmt: skip

    for stat, i, f in stats:
        for g in values:
            row(stat, "condition", g, "", f(point[g][i]), f(boot[g][:, i]))
        for a, b in itertools.combinations(values, 2):
            row(
                stat,
                "difference",
                a,
                b,
                f(point[a][i]) - f(point[b][i]),
                f(boot[a][:, i]) - f(boot[b][:, i]),
            )
    return rows


def print_table(rows: list[dict], names: list[str]) -> None:
    label = {g: f"[{i}]" for i, g in enumerate(names, 1)}
    for g in names:
        print(f"{label[g]} {g}")
    for r in rows:
        if r["kind"] == "run":
            continue
        who = (
            label[r["a"]]
            if r["kind"] == "condition"
            else f"{label[r['a']]}−{label[r['b']]}"
        )
        print(
            f"{r['metric']:<18} {r['statistic']:<28} {who:<8} "
            f"{r['estimate']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
        )


def _read_jsonl(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]
