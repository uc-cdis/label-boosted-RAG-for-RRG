"""Score one set of reports with RaTEScore, inside its image.

  docker compose -f eval/compose.yaml run --rm ratescore <gt.csv> <pred.csv> <out.csv>

(the image's entry point; the paths are inside the container, so mount their directory).

Same inputs as the cxr-report-metric service: CSVs with columns study_id (integers) and
report, one row per study, the same ids in both. out.csv gets study_id and ratescore,
<out>.json the timing.

Called as ReXrank calls it (rajpurkarlab/ReXrank-metric, scripts/run_ratescore.py):
RaTEScore() with its defaults (batch size 1, the "long" affinity matrix, GPU) on the
reports as strings, predictions first. The one change: the two Hub models its defaults
name are loaded from their local copies under /models (mounted read-only; jobs run with
no network), passed as the constructor's own arguments, so upstream's code is unpatched.

GPU required (use_gpu=True, ReXrank's setting): the stage passes the job's one GPU.
OMP_NUM_THREADS must be set (unset, torch would use every core of the host).
"""

import json
import os
import sys
import time

MODELS = {
    "bert_model": "/models/Angelakeke/RaTE-NER-Deberta",
    "eval_model": "/models/FremyCompany/BioLORD-2023-C",
}


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


def main(gt_csv, pred_csv, out_csv):
    if not os.environ.get("OMP_NUM_THREADS", "").isdigit():
        raise SystemExit("set OMP_NUM_THREADS")
    ids, references, predictions = read_inputs(gt_csv, pred_csv)

    import pandas as pd
    import torch
    from RaTEScore import RaTEScore

    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible: run with the job's GPU")
    scorer = RaTEScore(**MODELS)  # every other argument at upstream's default

    start = time.time()
    scores = scorer.compute_score(predictions, references)
    seconds = time.time() - start

    if len(scores) != len(ids):
        raise SystemExit("scores are missing for some studies")
    pd.DataFrame({"study_id": ids, "ratescore": scores}).to_csv(out_csv, index=False)
    timing = {
        "reports": len(ids),
        "seconds": round(seconds, 1),
        "device": torch.cuda.get_device_name(0),
        "torch_threads": torch.get_num_threads(),
    }
    with open(os.path.splitext(out_csv)[0] + ".json", "w") as f:
        json.dump(timing, f, indent=2)
    print(f"scored {len(ids)} studies in {timing['seconds']} s on {timing['device']}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
