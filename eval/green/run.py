"""Score one set of reports with GREEN, inside its image.

  docker compose -f eval/compose.yaml run --rm green <gt.csv> <pred.csv> <out.csv>
  docker compose -f eval/compose.yaml run --rm green --repro

(the image's entry point; the paths are inside the container, so mount their directory).

Same inputs as the other services: CSVs with columns study_id (integers) and report, one
row per study, the same ids in both. out.csv gets per study, in input order:
- study_id, green_score (GREEN's per-study score);
- GREEN's counts, renamed (keys without spaces): its six clinically significant error
  counts (a)-(f) as green_sig_{false_finding, missing_finding, location, severity,
  false_comparison, missing_comparison}, and green_matched_findings;
- green_unparsed: 1 if the judge's text lacks a section GREEN parses a score from (GREEN
  then counts that section as 0, so the study scores 0 or near it, silently), else 0;
- green_analysis: the judge's text, which every count is parsed from.
<out>.json gets the timing.

Called as ReXrank calls it (rajpurkarlab/ReXrank-metric, scripts/run_green.py):
GREEN(model, output_dir, cpu=False, compute_summary_stats=False) on the reports as
strings, references first; everything else at upstream's defaults (greedy, fp16, batch
8, max_length 2048). The one change: the judge is loaded from its local copy under
/models (mounted read-only; jobs run with no network), passed as the constructor's own
argument, so upstream's code is unpatched.

--repro: upstream's reproduction test (tests/test_repro.py: 5 report pairs and the judge
texts its authors got) on this image and GPU, with the test's own two checks (texts
equal after its normalization; scores equal to the score of the expected text). The
test file can't run as is: at import it also computes GREEN's summary statistics, which
load a second Hub model (sentence-transformers/paraphrase-mpnet-base-v2) no score
needs. So its inputs are read from the file, not executed.

GPU required: the stage passes the job's one GPU. OMP_NUM_THREADS must be set (unset,
torch would use every core of the host).
"""

import ast
import json
import os
import sys
import time

MODEL = "/models/StanfordAIMI/GREEN-RadLlama2-7b"
REPRO_TEST = "/opt/GREEN/tests/test_repro.py"

SIG_ERRORS = [  # GREEN's sub_categories, in its order -> our column names
    "green_sig_false_finding",
    "green_sig_missing_finding",
    "green_sig_location",
    "green_sig_severity",
    "green_sig_false_comparison",
    "green_sig_missing_comparison",
]


def read_inputs(gt_csv, pred_csv):
    """Both CSVs, checked: integer study ids, the same unique ids in the same order, and
    no empty report (pandas reads "" as NaN). Returns (ids, references, predictions)."""
    import pandas as pd

    frames = {}
    for name, path in (("gt", gt_csv), ("pred", pred_csv)):
        df = pd.read_csv(path)
        if not pd.api.types.is_integer_dtype(df["study_id"]):
            raise SystemExit(f"{name}: study_id must be integers")
        if df["report"].isna().any() or (df["report"].str.strip() == "").any():
            raise SystemExit(f"{name}: every report must be non-empty")
        if df["study_id"].duplicated().any():
            raise SystemExit(f"{name}: duplicate study ids")
        frames[name] = df
    if list(frames["gt"]["study_id"]) != list(frames["pred"]["study_id"]):
        raise SystemExit("gt and pred must hold the same study ids in the same order")
    return (
        list(frames["gt"]["study_id"]),
        [str(r) for r in frames["gt"]["report"]],
        [str(r) for r in frames["pred"]["report"]],
    )


def load_scorer():
    import torch
    from green_score import GREEN

    if not os.environ.get("OMP_NUM_THREADS", "").isdigit():
        raise SystemExit("set OMP_NUM_THREADS")
    if torch.cuda.device_count() != 1:
        # >1 GPU makes GREEN start torch.distributed, which needs torchrun
        raise SystemExit(f"run with one GPU, got {torch.cuda.device_count()}")
    # output_dir: stored by GREEN, never written to when scoring
    return GREEN(MODEL, output_dir="/tmp", cpu=False, compute_summary_stats=False)


def unparsed(scorer, text):
    """True if GREEN finds no section to parse for a score it computes (with
    for_reward=True its parser returns None there instead of counting 0)."""
    return any(
        scorer.parse_error_counts(text, c, for_reward=True)[0] is None
        for c in ("Clinically Significant Errors", "Matched Findings")
    )


def main(gt_csv, pred_csv, out_csv):
    ids, references, predictions = read_inputs(gt_csv, pred_csv)

    import pandas as pd
    import torch

    scorer = load_scorer()
    start = time.time()
    _, _, scores, _, results = scorer(references, predictions)
    seconds = time.time() - start

    if len(scores) != len(ids) or len(results) != len(ids):
        raise SystemExit("scores are missing for some studies")
    if list(results["predictions"]) != predictions:
        raise SystemExit("GREEN returned the studies in a different order")
    out = pd.DataFrame({"study_id": ids, "green_score": scores})
    for column, sub_category in zip(SIG_ERRORS, scorer.sub_categories):
        out[column] = results[sub_category].to_numpy()
    out["green_matched_findings"] = results["Matched Findings"].to_numpy()
    out["green_unparsed"] = [
        int(unparsed(scorer, t)) for t in results["green_analysis"]
    ]
    out["green_analysis"] = results["green_analysis"].to_numpy()
    out.to_csv(out_csv, index=False)
    timing = {
        "reports": len(ids),
        "seconds": round(seconds, 1),
        "device": torch.cuda.get_device_name(0),
        "torch_threads": torch.get_num_threads(),
    }
    with open(os.path.splitext(out_csv)[0] + ".json", "w") as f:
        json.dump(timing, f, indent=2)
    print(f"scored {len(ids)} studies in {timing['seconds']} s on {timing['device']}")


def repro():
    """Upstream's reproduction test (see the module docstring); exit 1 on a mismatch."""
    import difflib
    import math

    import torch

    names = ("refs", "hyps", "expected_green_score_list")
    tree = ast.parse(open(REPRO_TEST).read())
    data = {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) in names
    }
    epsilon = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "epsilon"
    )
    refs, hyps, expected = (data[n] for n in names)

    scorer = load_scorer()
    _, _, _, _, results = scorer(refs, hyps)
    print(f"device: {torch.cuda.get_device_name(0)}")

    def norm(text):  # the test's normalization
        return text.strip().replace("</s>", "").replace(":\n", "")

    failed = 0
    for i, (text, score) in enumerate(
        zip(results["green_analysis"], results["green_score"])
    ):
        want = scorer.compute_green(response=expected[i])
        same_text = norm(text) == norm(expected[i])
        same_score = math.isclose(score, want, rel_tol=epsilon)
        print(
            f"case {i}: text {'OK' if same_text else 'DIFFERS'}, "
            f"score {score:.6f} vs {want:.6f} {'OK' if same_score else 'DIFFERS'}"
        )
        if not same_text:
            diff = difflib.unified_diff(
                norm(expected[i]).splitlines(), norm(text).splitlines(), lineterm=""
            )
            print("\n".join(diff))
        failed += not (same_text and same_score)
    print(f"repro: {len(expected) - failed}/{len(expected)} cases match")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    if sys.argv[1:] == ["--repro"]:
        repro()
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
