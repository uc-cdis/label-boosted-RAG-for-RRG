"""Score reports with GREEN, inside its image.

  docker compose -f eval/compose.yaml run --rm green-hf <gt.csv> <pred.csv> <out.csv>
  docker compose -f eval/compose.yaml run --rm green-hf --repro
  docker compose -f eval/compose.yaml run --rm green-vllm <gt.csv> <pred.csv> <out.csv>
  docker compose -f eval/compose.yaml run --rm green-vllm score <dir>...
  docker compose -f eval/compose.yaml run --rm green-vllm --repro

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

Two ways to run the judge (eval/compose.yaml):

1. service green-hf: upstream's own HF generate, as ReXrank calls it
   (rajpurkarlab/ReXrank-metric, scripts/run_green.py): GREEN(model, output_dir,
   cpu=False, compute_summary_stats=False) on the reports as strings, references first;
   everything else at upstream's defaults (greedy, fp16, batch 8, max_length 2048). The
   one change: the judge is loaded from its local copy under /models (mounted read-only;
   jobs run with no network), passed as the constructor's own argument, so upstream's
   code is unpatched. One GPU.

2. service green-vllm (GREEN_JUDGE_URL set): the judge served by vLLM (service
   green-judge, greedy, fp16); this container has no GPU and does everything but the
   forward passes, with upstream's code. Per study: the prompt (GREEN's make_prompt and
   chat template), tokenized as GREEN's get_response does, one study per batch (no
   padding: the same ids as the unpadded part of upstream's batch); a request with
   those token ids, the output cap (generate's max_length=2048 minus the prompt) and
   the stop token (the tokenizer's eos, which generate stops on); the returned token ids
   decoded with the prompt, as get_response does (the prompt is part of the decoded
   text there; clean_responses cuts it off); then GREEN's own results code. `score`
   takes directories that each hold one set's gt.csv and pred.csv (out.csv is written
   next to them), and sends every set's requests in one queue. <out>.json also gets the
   finish reasons and token counts; <out>.prom the server's Prometheus metrics at the end.
   One difference from upstream: HF counts max_length on the left-padded batch, so there
   a study's cap also depends on its batch neighbours' prompt lengths; here it is its
   own. It only matters if the judge reaches the cap (see finish_reason "length").

--repro: upstream's reproduction test (tests/test_repro.py: 5 report pairs and the judge
texts its authors got) through the service's judge path, with the test's own two checks
(texts equal after its normalization; scores equal to the score of the expected text).
The test file can't run as is: at import it also computes GREEN's summary statistics,
which load a second Hub model (sentence-transformers/paraphrase-mpnet-base-v2) no score
needs. So its inputs are read from the file, not executed.

OMP_NUM_THREADS must be set (unset, torch would use every core of the host).
"""

import ast
import asyncio
import json
import os
import sys
import time

MODEL = "/models/StanfordAIMI/GREEN-RadLlama2-7b"
REPRO_TEST = "/opt/GREEN/tests/test_repro.py"
MAX_LENGTH = 2048  # GREEN.get_response: model.generate(..., max_length=2048)
JUDGE_URL = os.environ.get("GREEN_JUDGE_URL")  # set by compose for green-vllm
CONCURRENCY = 256  # requests in flight; vLLM batches them
REQUEST_TIMEOUT_S = 3600

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


def check_threads():
    if not os.environ.get("OMP_NUM_THREADS", "").isdigit():
        raise SystemExit("set OMP_NUM_THREADS")


def load_scorer():
    """GREEN with its judge on the one GPU (upstream's generate path)."""
    import torch
    from green_score import GREEN

    check_threads()
    if torch.cuda.device_count() != 1:
        # >1 GPU makes GREEN start torch.distributed, which needs torchrun
        raise SystemExit(f"run with one GPU, got {torch.cuda.device_count()}")
    # output_dir: stored by GREEN, never written to when scoring
    return GREEN(MODEL, output_dir="/tmp", cpu=False, compute_summary_stats=False)


def load_text_scorer():
    """GREEN without its judge (the vLLM path): upstream's constructor sets up the
    tokenizer and chat template; loading the 7B weights is skipped (never used here).
    cpu=True keeps it off the GPU and out of torch.distributed."""
    from unittest import mock

    import green_score.green as upstream

    class NoJudge:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return mock.Mock()  # GREEN only calls .eval() on it

    check_threads()
    with mock.patch.object(upstream, "AutoModelForCausalLM", NoJudge):
        return upstream.GREEN(
            MODEL, output_dir="/tmp", cpu=True, compute_summary_stats=False
        )


def unparsed(scorer, text):
    """True if GREEN finds no section to parse for a score it computes (with
    for_reward=True its parser returns None there instead of counting 0)."""
    return any(
        scorer.parse_error_counts(text, c, for_reward=True)[0] is None
        for c in ("Clinically Significant Errors", "Matched Findings")
    )


def write_scores(scorer, ids, predictions, scores, results, out_csv):
    """out.csv (module docstring) from GREEN's (green_scores, results_df)."""
    import pandas as pd

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


def main(gt_csv, pred_csv, out_csv):
    if JUDGE_URL:
        score_served([(gt_csv, pred_csv, out_csv)])
        return
    ids, references, predictions = read_inputs(gt_csv, pred_csv)

    import torch

    scorer = load_scorer()
    start = time.time()
    _, _, scores, _, results = scorer(references, predictions)
    seconds = time.time() - start

    write_scores(scorer, ids, predictions, scores, results, out_csv)
    timing = {
        "reports": len(ids),
        "seconds": round(seconds, 1),
        "device": torch.cuda.get_device_name(0),
        "torch_threads": torch.get_num_threads(),
    }
    with open(os.path.splitext(out_csv)[0] + ".json", "w") as f:
        json.dump(timing, f, indent=2)
    print(f"scored {len(ids)} studies in {timing['seconds']} s on {timing['device']}")


def prompts_of(scorer, references, predictions):
    """GREEN's prompts, as its __call__ builds them (dataset map over make_prompt)."""
    from datasets import Dataset

    dataset = Dataset.from_dict({"reference": references, "prediction": predictions})
    return scorer.process_data(dataset)


def tokenize(scorer, prompt):
    """One prompt's token ids, as get_response builds and tokenizes a batch (of one)."""
    chat = [{"from": "human", "value": prompt}, {"from": "gpt", "value": ""}]
    tokens = scorer.tokenize_batch_as_chat([chat])["input_ids"][0].tolist()
    if len(tokens) >= MAX_LENGTH:
        raise SystemExit(
            f"a prompt of {len(tokens)} tokens leaves no room under "
            f"max_length={MAX_LENGTH}"
        )
    return tokens


def results_of(scorer, dataset, texts):
    """GREEN's (green_scores, results_df) for judge texts, by its own results code:
    what infer() sets before process_results(), then process_results()."""
    from green_score.utils import clean_responses

    scorer.dataset = dataset
    scorer.prompts = dataset["prompt"]
    scorer.completions = [clean_responses(t) for t in texts]
    _, _, scores, _, results = scorer.process_results()
    return scores, results


async def request_all(prompts, eos):
    """One greedy completion per prompt (token ids) from the served judge, in order:
    (generated token ids, finish reason) each, the server's version and its metrics."""
    import aiohttp
    from tqdm import tqdm

    gate = asyncio.Semaphore(CONCURRENCY)
    bar = tqdm(total=len(prompts), desc="judge", mininterval=30)
    async with aiohttp.ClientSession(
        JUDGE_URL,
        timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S),
        connector=aiohttp.TCPConnector(limit=CONCURRENCY),
    ) as session:

        async def one(tokens):
            body = {
                "prompt": tokens,
                "max_tokens": MAX_LENGTH - len(tokens),
                "temperature": 0,
                "n": 1,
                "stop_token_ids": [eos],
                "return_token_ids": True,
            }
            async with gate, session.post("/v1/completions", json=body) as r:
                if r.status != 200:
                    raise RuntimeError(f"judge: HTTP {r.status}: {await r.text()}")
                choice = (await r.json())["choices"][0]
            if choice.get("prompt_token_ids") != tokens:
                raise RuntimeError("judge: the server saw a different prompt")
            bar.update()
            return choice["token_ids"], choice["finish_reason"]

        try:
            outputs = await asyncio.gather(*(one(t) for t in prompts))
        finally:
            bar.close()
        async with session.get("/version") as r:
            version = (await r.json())["version"]
        async with session.get("/metrics") as r:
            metrics = await r.text()
    return outputs, version, metrics


def judge_served(scorer, sets):
    """Judge texts for several sets of (references, predictions), all in one queue.

    Returns per set (dataset, texts, run info) and the run's server metrics."""
    eos = scorer.tokenizer.eos_token_id
    assert scorer.max_length == MAX_LENGTH  # the tokenizer's truncation
    datasets = [prompts_of(scorer, refs, preds) for refs, preds in sets]
    prompts = [[tokenize(scorer, p) for p in d["prompt"]] for d in datasets]
    everything = [t for ps in prompts for t in ps]
    start = time.time()
    outputs, version, metrics = asyncio.run(request_all(everything, eos))
    seconds = time.time() - start
    print(f"judged {len(everything)} studies in {seconds:.1f} s (vLLM {version})")

    results, i = [], 0
    for dataset, ps in zip(datasets, prompts):
        part, i = outputs[i : i + len(ps)], i + len(ps)
        # get_response: decode the whole sequence (prompt + output; pad and eos are
        # special tokens, skipped); results_of applies clean_responses
        texts = scorer.tokenizer.batch_decode(
            [p + tokens for p, (tokens, _) in zip(ps, part)], skip_special_tokens=True
        )
        finish = {}
        for _, reason in part:
            finish[reason] = finish.get(reason, 0) + 1
        info = {
            "reports": len(ps),
            "judge": {"server": "vllm", "version": version},
            # the run's wall time over all its requests (one queue, one GPU)
            "seconds": round(seconds, 1),
            "requests_in_run": len(everything),
            "seconds_per_request": seconds / len(everything),
            "finish_reason": dict(sorted(finish.items())),
            "prompt_tokens": stats([len(p) for p in ps]),
            "completion_tokens": stats([len(tokens) for tokens, _ in part]),
        }
        results.append((dataset, texts, info))
    return results, metrics


def stats(values):
    import numpy as np

    v = np.asarray(values)
    return {
        "mean": float(v.mean()),
        "p50": float(np.percentile(v, 50)),
        "p95": float(np.percentile(v, 95)),
        "max": int(v.max()),
    }


def score_served(files):
    """Score sets of (gt.csv, pred.csv, out.csv) with the served judge."""
    scorer = load_text_scorer()
    inputs = [read_inputs(gt, pred) for gt, pred, _ in files]
    judged, metrics = judge_served(scorer, [(refs, preds) for _, refs, preds in inputs])
    for (_, _, out_csv), (ids, _, preds), (dataset, texts, info) in zip(
        files, inputs, judged
    ):
        scores, results = results_of(scorer, dataset, texts)
        write_scores(scorer, ids, preds, scores, results, out_csv)
        base = os.path.splitext(out_csv)[0]
        with open(base + ".json", "w") as f:
            json.dump(info, f, indent=2)
        with open(base + ".prom", "w") as f:
            f.write(metrics)
        print(f"{out_csv}: scored {len(ids)} studies, finish {info['finish_reason']}")


def repro_cases():
    """(refs, hyps, expected texts, epsilon), read from upstream's test file."""
    names = ("refs", "hyps", "expected_green_score_list", "epsilon")
    tree = ast.parse(open(REPRO_TEST).read())
    data = {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) in names
    }
    return tuple(data[n] for n in names)


def check_repro(scorer, texts, scores):
    """The test's two checks per case; exit 1 on a mismatch."""
    import difflib
    import math

    _, _, expected, epsilon = repro_cases()

    def norm(text):  # the test's normalization
        return text.strip().replace("</s>", "").replace(":\n", "")

    failed = 0
    for i, (text, score) in enumerate(zip(texts, scores, strict=True)):
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


def repro():
    """Upstream's reproduction test through this service's judge path."""
    refs, hyps, _, _ = repro_cases()
    if JUDGE_URL:
        scorer = load_text_scorer()
        [(dataset, texts, info)], _ = judge_served(scorer, [(refs, hyps)])
        print(f"finish: {info['finish_reason']}, vLLM {info['judge']['version']}")
        scores, results = results_of(scorer, dataset, texts)
    else:
        import torch

        scorer = load_scorer()
        _, _, scores, _, results = scorer(refs, hyps)
        print(f"device: {torch.cuda.get_device_name(0)}")
    check_repro(scorer, results["green_analysis"], scores)


if __name__ == "__main__":
    command, args = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else (None, [])
    if command == "--repro" and not args:
        repro()
    elif command == "score" and args and JUDGE_URL:
        score_served([(f"{d}/gt.csv", f"{d}/pred.csv", f"{d}/out.csv") for d in args])
    elif len(sys.argv) == 4:
        main(*sys.argv[1:])
    else:
        raise SystemExit(__doc__)
