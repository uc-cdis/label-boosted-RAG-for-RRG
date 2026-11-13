"""Score one set of reports with CXR-Report-Metric, inside the official image.

  docker compose -f eval/compose.yaml run --rm cxr-report-metric <gt.csv> <pred.csv> <out.csv>

(the image's entry point; the paths are inside the container, so mount their directory).

The CSVs have upstream's columns, study_id (integers: upstream casts them with int())
and report, one row per study, the same ids in both. Upstream writes its intermediate
files next to the inputs and into its working directory; out.csv gets one row per study
with bleu_score, bertscore, semb_score, radgraph_combined, RadCliQ-v0 and RadCliQ-v1, and
<out>.json the timing.

Upstream runs CheXbert and RadGraph as subprocesses (os.system) and doesn't check that
they succeeded, so stale outputs of an earlier call could be read silently: they are
deleted first, and every study must come back with every score.

Threads: torch (here and in upstream's subprocesses) follows OMP_NUM_THREADS, which must
be set (unset, torch uses every core of the host). Set it below the CPUs the job has: torch 1.6 parallelizes even tiny ops, so one preempted
thread stalls each op (measured 2026-11-13: 32 threads on 32 pinned CPUs of a loaded
host ran RadGraph ~7x slower than 16).

Python 3.7 (the official image). CPU only (see the Dockerfile).
"""

import json
import os
import shutil
import sys
import time

ROOT = "/opt/CXR-Report-Metric"
# Upstream's caches and temp files, relative to ROOT.
STALE = ["cache", "temp_dygie_input.json", "temp_dygie_output.json"]
SCORES = ["bleu_score", "bertscore", "semb_score", "radgraph_combined", "RadCliQ-v1"]


def read_inputs(gt_csv, pred_csv):
    """Both CSVs, checked: integer study ids (upstream casts them with int()), the same
    unique ids in both, and no empty report (pandas reads "" as NaN, which upstream's
    RadGraph step silently drops). Returns the sorted ids."""
    import pandas as pd

    ids = None
    for name, path in (("gt", gt_csv), ("pred", pred_csv)):
        df = pd.read_csv(path)
        if not pd.api.types.is_integer_dtype(df["study_id"]):
            raise SystemExit(name + ": study_id must be integers")
        if df["report"].isna().any() or (df["report"].str.strip() == "").any():
            raise SystemExit(name + ": every report must be non-empty")
        these = sorted(df["study_id"])
        if len(set(these)) != len(these) or (ids is not None and these != ids):
            raise SystemExit("gt and pred must hold the same unique study ids")
        ids = these
    return ids


def enter_upstream():
    """cd to upstream's root (its paths are relative to it), drop stale outputs."""
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    for name in STALE:
        if os.path.isdir(name):
            shutil.rmtree(name)
        elif os.path.exists(name):
            os.remove(name)


def main(gt_csv, pred_csv, out_csv):
    if not os.environ.get("OMP_NUM_THREADS", "").isdigit():
        raise SystemExit("set OMP_NUM_THREADS (see the docstring)")
    gt_csv, pred_csv, out_csv = (
        os.path.abspath(p) for p in (gt_csv, pred_csv, out_csv)
    )
    ids = read_inputs(gt_csv, pred_csv)
    enter_upstream()

    import CXRMetric.run_eval as run_eval
    import pandas as pd
    import torch

    # radcliq-v1.pkl was pickled from a script, so it unpickles as __main__.CompositeMetric
    # (upstream's test_metric.py imports it into __main__ for the same reason).
    sys.modules["__main__"].CompositeMetric = run_eval.CompositeMetric

    start = time.time()
    use_idf = False  # as upstream's config.py
    run_eval.calc_metric(gt_csv, pred_csv, out_csv, use_idf)
    seconds = time.time() - start

    out = pd.read_csv(out_csv)
    if sorted(out["study_id"]) != ids:
        raise SystemExit("scores are missing for some studies")
    bad = [c for c in SCORES if out[c].isna().any()]
    if bad:
        raise SystemExit("missing values in: " + ", ".join(bad))
    timing = {
        "reports": len(ids),
        "seconds": round(seconds, 1),
        "torch_threads": torch.get_num_threads(),
    }
    with open(os.path.splitext(out_csv)[0] + ".json", "w") as f:
        json.dump(timing, f, indent=2)
    print(
        "scored {reports} studies in {seconds} s ({torch_threads} threads)".format(
            **timing
        )
    )


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
