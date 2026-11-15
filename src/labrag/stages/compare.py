"""Stage ``compare``: means with 95% CIs per condition, and paired differences.

Reads the scores artifacts of two or more conditions scored on the same studies; writes
``comparisons/<out>/comparison.csv``: per score column, each condition's mean and each
pair's difference (a − b, in the order given), with a percentile interval from a paired
bootstrap over studies (docs/revision-plan.md §3). Also ``generations/empty_report``
and 1/RadCliQ-v1 = 1 / mean(RadCliQ-v1).

Paired: a drawn study counts in every condition, so shared noise cancels in the
differences. Fixed seed and single-threaded numpy means, so intervals don't depend on
the machine. Scores are never missing; empty reports are scored as "EMPTY".
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
    generations = [g.strip("/").removeprefix("generations/") for g in generations]
    if len(generations) < 2:
        raise SystemExit("compare: give at least two generations artifacts")
    if len(set(generations)) < len(generations):
        raise SystemExit("compare: a generations artifact is given twice")
    relpath = f"comparisons/{out.strip('/').removeprefix('comparisons/')}"
    config = {
        "stage": "compare",
        "generations": [f"generations/{g}" for g in generations],
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
        for g in generations:
            score_dir = paths.SHARED_ROOT / "scores" / g / metric
            inputs[f"{g}/{metric}"] = artifacts.artifact_record(score_dir)
            records[g] = _read_jsonl(score_dir / "scores.jsonl.gz")
        if not tables:  # every scores artifact has it; taken once, from the first
            empty = ["empty_report"]
            tables += compare(align(records, empty), empty, "generations")
        columns = evaluate.METRICS[metric]["scores"]
        tables += compare(align(records, columns), columns, metric)

    def build(dest: Path) -> dict:
        with open(dest / "comparison.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
            w.writeheader()
            w.writerows(tables)
        print_table(tables, generations)
        return {"conditions": {str(i): g for i, g in enumerate(generations, 1)}}

    return artifacts.publish(relpath, config, inputs, build, overwrite)


def align(records: dict[str, list[dict]], columns: list[str]) -> dict[str, np.ndarray]:
    """Per condition, an (n, columns) array with the studies in one shared order.

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
    values = {}
    for g, by_key in keyed.items():
        v = np.array(
            [[float(by_key[k][c]) for c in columns] for k in order], dtype=np.float64
        )
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


def print_table(rows: list[dict], generations: list[str]) -> None:
    label = {g: f"[{i}]" for i, g in enumerate(generations, 1)}
    for g in generations:
        print(f"{label[g]} {g}")
    for r in rows:
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
