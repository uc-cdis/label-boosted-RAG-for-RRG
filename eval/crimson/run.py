"""Score reports with CRIMSON, inside its image.

  docker compose -f eval/compose.yaml run --rm <service> <gt.csv> <pred.csv> <out.csv>
  docker compose -f eval/compose.yaml run --rm crimson-client score <dir>...

(service: crimson-native or crimson-client; paths inside the container: mount their dir)

Inputs: CSVs (study_id as integers, report), the same ids in both. out.csv per study:
crimson_score (in (-1, 1]); upstream's error_counts as crimson_<key>; crimson_failed (1
if upstream's evaluate_batch could not parse or score the judge's text: no score or
counts then); crimson_response (the judge's text). <out>.json: timing, failures, and how
many texts clean_report_text changed.

Both paths run upstream's CRIMSONScore.evaluate_batch (references first) with:
- the judge from /models as model_name, and include_guidelines=False: upstream drops the
  guidelines only for the Hub id (its judge was trained without them), so a local path
  needs the flag to get the same prompt;
- no patient context (docs/revision-plan.md §7);
- CRIMSON's clean_report_text on both texts first: ReXrank says it runs it; upstream's
  scoring path doesn't (§7);
- the judge's generation config (greedy, max_new_tokens 8192), as upstream uses it.

Two judge paths:
1. crimson-native: upstream's HF pipeline, one study per pass (upstream's default), bf16,
   one GPU. The reference.
2. crimson-client (CRIMSON_JUDGE_URL set): the judge served by vLLM (service
   crimson-vllm); no GPU here. Upstream's pipeline is built as in 1 (weights on the CPU,
   never run); ServedPipe runs its own parameter handling, preprocess (chat template,
   tokenizer) and postprocess (decoding), and swaps only the forward pass for a request
   with the prompt's ids, greedy, max_new_tokens and eos. One prompt per request, so no
   padding: the ids of upstream's batch of one. ``score`` judges each directory's set
   (gt.csv, pred.csv → out.csv) alone on the server, with its own cache_salt;
   <out>.start/end.prom hold the server's metrics around it.

OMP_NUM_THREADS must be set (unset, torch would use every core of the host).
"""

import asyncio
import json
import os
import sys
import time

MODEL = "/models/rajpurkarlab/medgemma-4b-it-crimson"
JUDGE_URL = os.environ.get("CRIMSON_JUDGE_URL")  # set by compose for crimson-client
CONCURRENCY = 256  # requests in flight; vLLM batches them
REQUEST_TIMEOUT_S = 3600
# vLLM drops idle connections after 5 s (VLLM_HTTP_TIMEOUT_KEEP_ALIVE); the client lets
# them go sooner, and retries a request whose reused connection was closed anyway
# (safe: greedy and batch-invariant, so the retry gets the same output).
KEEPALIVE_S = 2
DISCONNECT_RETRIES = 3

ERROR_COUNTS = [  # the keys of upstream's result["error_counts"], in its order
    "false_findings",
    "missing_findings",
    "attribute_errors",
    "location_errors",
    "severity_errors",
    "descriptor_errors",
    "measurement_errors",
    "certainty_errors",
    "unspecific_errors",
    "overinterpretation_errors",
    "temporal_errors",
]


def read_inputs(gt_csv, pred_csv):
    """Both CSVs, checked: integer study ids, the same unique ids in the same order, and
    no empty report (pandas reads "" as NaN). Returns (ids, references, predictions)."""
    import pandas as pd

    frames = {}
    for name, path in (("gt", gt_csv), ("pred", pred_csv)):
        df = pd.read_csv(path, keep_default_na=False, dtype={"report": str})
        if not pd.api.types.is_integer_dtype(df["study_id"]):
            raise SystemExit(f"{name}: study_id must be integers")
        if (df["report"].str.strip() == "").any():
            raise SystemExit(f"{name}: every report must be non-empty")
        if df["study_id"].duplicated().any():
            raise SystemExit(f"{name}: duplicate study ids")
        frames[name] = df
    if list(frames["gt"]["study_id"]) != list(frames["pred"]["study_id"]):
        raise SystemExit("gt and pred must hold the same study ids in the same order")
    return (
        list(frames["gt"]["study_id"]),
        list(frames["gt"]["report"]),
        list(frames["pred"]["report"]),
    )


def load_scorer():
    """Upstream's CRIMSONScore with its HF pipeline: on the one GPU (crimson-native), or
    on the CPU (crimson-client: no GPU; only the pipeline's text steps run)."""
    import torch
    from CRIMSON.generate_score import CRIMSONScore

    if not os.environ.get("OMP_NUM_THREADS", "").isdigit():
        raise SystemExit("set OMP_NUM_THREADS")
    gpus = torch.cuda.device_count()
    if gpus != (0 if JUDGE_URL else 1):
        raise SystemExit(f"{'no' if JUDGE_URL else 'one'} GPU expected, got {gpus}")
    scorer = CRIMSONScore(api="hf", model_name=MODEL)
    if not scorer._has_generation_config:
        raise SystemExit("the judge's generation_config.json was not found")
    return scorer


class Recorder:
    """The scorer's pipeline, recording the judge's text of each call's chats (what
    upstream reads from the output; evaluate_batch keeps only parsed results)."""

    def __init__(self, pipe):
        self._pipe = pipe
        self.texts = []

    def __getattr__(self, name):  # model, tokenizer: the pipeline's
        return getattr(self._pipe, name)

    def __call__(self, chats, **kwargs):
        outputs = self._pipe(chats, **kwargs)
        self.texts.extend(o[0]["generated_text"][-1]["content"] for o in outputs)
        return outputs


def clean_inputs(references, predictions):
    """Both lists through CRIMSON's clean_report_text, and how many texts it changed."""
    from CRIMSON.utils import clean_report_text

    refs = [clean_report_text(t) for t in references]
    preds = [clean_report_text(t) for t in predictions]
    changed = {
        "reference": sum(a != b for a, b in zip(refs, references)),
        "generated": sum(a != b for a, b in zip(preds, predictions)),
    }
    return refs, preds, changed


def evaluate(scorer, references, predictions, batch_size):
    """Upstream's evaluate_batch on cleaned texts; (results, judge texts, cleaning)."""
    refs, preds, changed = clean_inputs(references, predictions)
    pipe = scorer.pipe
    scorer.pipe = recorder = Recorder(pipe)
    try:
        results = scorer.evaluate_batch(
            refs, preds, include_guidelines=False, batch_size=batch_size
        )
    finally:
        scorer.pipe = pipe
    if len(results) != len(refs) or len(recorder.texts) != len(refs):
        raise SystemExit("results are missing for some studies")
    return results, recorder.texts, changed


def write_scores(ids, results, texts, out_csv):
    """out.csv (module docstring) from evaluate_batch's results (None = failed)."""
    import pandas as pd

    rows = []
    for study_id, result, text in zip(ids, results, texts, strict=True):
        row = {"study_id": study_id}
        if result is None:
            row["crimson_score"] = None
            row.update({f"crimson_{k}": None for k in ERROR_COUNTS})
        else:
            if set(result["error_counts"]) != set(ERROR_COUNTS):
                raise SystemExit(f"unexpected error counts: {result['error_counts']}")
            row["crimson_score"] = result["crimson_score"]
            row.update(
                {f"crimson_{k}": result["error_counts"][k] for k in ERROR_COUNTS}
            )
        row["crimson_failed"] = int(result is None)
        row["crimson_response"] = text
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def main(gt_csv, pred_csv, out_csv):
    if JUDGE_URL:
        score_served([(gt_csv, pred_csv, out_csv)])
        return
    ids, references, predictions = read_inputs(gt_csv, pred_csv)

    import torch

    scorer = load_scorer()
    start = time.time()
    results, texts, changed = evaluate(scorer, references, predictions, batch_size=1)
    seconds = time.time() - start

    write_scores(ids, results, texts, out_csv)
    info = {
        "reports": len(ids),
        "seconds": round(seconds, 1),
        "device": torch.cuda.get_device_name(0),
        "torch_threads": torch.get_num_threads(),
        "failed": sum(r is None for r in results),
        "clean_report_text_changed": changed,
    }
    with open(os.path.splitext(out_csv)[0] + ".json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"scored {len(ids)} studies in {info['seconds']} s on {info['device']}")


class ServedPipe:
    """Stands in for the scorer's HF pipeline: upstream's pipeline steps around a
    served forward pass (module docstring, 2). ``salt``: the cache salt of this set's
    requests. After a call: ``finish`` (finish reason per chat), ``seconds``, the
    server's ``version`` and its ``metrics`` before and after."""

    def __init__(self, pipe, salt):
        self._pipe = pipe
        self.salt = salt

    def __getattr__(self, name):  # model, tokenizer: the pipeline's
        return getattr(self._pipe, name)

    def __call__(self, chats, batch_size=None, **kwargs):
        import torch
        from transformers.pipelines.text_generation import Chat

        pipe = self._pipe
        # Pipeline.__call__'s parameter handling: this call's, over the pipeline's own
        pre, forward, post = pipe._sanitize_parameters(**kwargs)
        pre = {**pipe._preprocess_params, **pre}
        forward = {**pipe._forward_params, **forward}
        post = {**pipe._postprocess_params, **post}
        if set(forward) != {"generation_config"}:
            raise SystemExit(f"unexpected generate arguments: {sorted(forward)}")
        config = forward["generation_config"]
        inputs = [pipe.preprocess(Chat(chat), **pre) for chat in chats]
        prompts = []
        for x in inputs:
            if x["input_ids"].shape[0] != 1 or not bool(x["attention_mask"].all()):
                raise SystemExit("expected one unpadded prompt per chat")
            prompts.append(x["input_ids"][0].tolist())
        answers = asyncio.run(self.request_all(prompts, sampling(config)))
        self.finish = [reason for _, reason in answers]
        self.prompt_tokens = [len(p) for p in prompts]
        self.completion_tokens = [len(tokens) for tokens, _ in answers]
        outputs = []
        for x, prompt, (tokens, _) in zip(inputs, prompts, answers):
            # what _forward returns for a batch of one: the prompt and its continuation
            sequence = torch.tensor([[prompt + tokens]])
            model_outputs = {
                "generated_sequence": sequence,
                "input_ids": x["input_ids"],
                "prompt_text": x["prompt_text"],
            }
            outputs.append(pipe.postprocess(model_outputs, **post))
        return outputs

    async def request_all(self, prompts, params):
        """(generated token ids, finish reason) per prompt, from the served judge."""
        import httpx
        from tqdm import tqdm

        gate = asyncio.Semaphore(CONCURRENCY)
        limits = httpx.Limits(max_connections=CONCURRENCY, keepalive_expiry=KEEPALIVE_S)
        async with httpx.AsyncClient(
            base_url=JUDGE_URL, timeout=REQUEST_TIMEOUT_S, limits=limits
        ) as client:
            models = (await client.get("/v1/models")).raise_for_status().json()
            max_model_len = models["data"][0]["max_model_len"]
            longest = max(len(p) for p in prompts)
            if longest + params["max_tokens"] > max_model_len:
                raise SystemExit(
                    f"a prompt of {longest} tokens + max_new_tokens "
                    f"{params['max_tokens']} exceeds the server's {max_model_len}"
                )
            bar = tqdm(total=len(prompts), desc="judge", mininterval=30)

            async def post(body):
                for attempt in range(DISCONNECT_RETRIES + 1):
                    try:
                        return await client.post("/v1/completions", json=body)
                    except httpx.RemoteProtocolError:  # a closed kept-alive connection
                        if attempt == DISCONNECT_RETRIES:
                            raise
                        print("judge: server disconnected, retrying", file=sys.stderr)

            async def one(tokens):
                body = {
                    "prompt": tokens,
                    **params,
                    "n": 1,
                    "return_token_ids": True,
                    "cache_salt": self.salt,  # prefix-cache blocks stay within a set
                }
                async with gate:
                    r = await post(body)
                if r.status_code != 200:
                    raise RuntimeError(f"judge: HTTP {r.status_code}: {r.text}")
                choice = r.json()["choices"][0]
                if choice.get("prompt_token_ids") != tokens:
                    raise RuntimeError("judge: the server saw a different prompt")
                bar.update()
                return choice["token_ids"], choice["finish_reason"]

            self.metrics = [(await client.get("/metrics")).text]
            start = time.monotonic()
            try:
                answers = await asyncio.gather(*(one(p) for p in prompts))
            finally:
                bar.close()
            self.seconds = time.monotonic() - start
            self.metrics.append((await client.get("/metrics")).text)
            self.version = (await client.get("/version")).json()["version"]
        return answers


def sampling(config):
    """The request's sampling parameters for the judge's generation config: greedy to
    max_new_tokens, stopping at its eos token(s), as HF generate decodes with it."""
    defaults = {"num_beams": 1, "repetition_penalty": 1.0, "length_penalty": 1.0}
    if config.do_sample or any(
        getattr(config, k, v) not in (None, v) for k, v in defaults.items()
    ):
        raise SystemExit(f"unsupported generation config: {config}")
    eos = config.eos_token_id
    return {
        "max_tokens": config.max_new_tokens,
        "temperature": 0,
        "stop_token_ids": eos if isinstance(eos, list) else [eos],
    }


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
    """Score sets of (gt.csv, pred.csv, out.csv) with the served judge, one at a time."""
    scorer = load_scorer()
    inputs = [read_inputs(gt, pred) for gt, pred, _ in files]
    pipe = scorer.pipe
    for k, ((_, _, out_csv), (ids, refs, preds)) in enumerate(zip(files, inputs)):
        scorer.pipe = served = ServedPipe(pipe, salt=f"set-{k}")
        try:
            # one call of the pipeline for the whole set: every request in flight
            results, texts, changed = evaluate(scorer, refs, preds, len(refs))
        finally:
            scorer.pipe = pipe
        write_scores(ids, results, texts, out_csv)
        finish = {}
        for reason in served.finish:
            finish[reason] = finish.get(reason, 0) + 1
        info = {
            "reports": len(ids),
            "judge": {"server": "vllm", "version": served.version},
            "seconds": round(served.seconds, 1),
            "seconds_per_request": served.seconds / len(ids),
            "finish_reason": dict(sorted(finish.items())),
            "prompt_tokens": stats(served.prompt_tokens),
            "completion_tokens": stats(served.completion_tokens),
            "failed": sum(r is None for r in results),
            "clean_report_text_changed": changed,
        }
        base = os.path.splitext(out_csv)[0]
        with open(base + ".json", "w") as f:
            json.dump(info, f, indent=2)
        for when, text in zip(("start", "end"), served.metrics):
            with open(f"{base}.{when}.prom", "w") as f:
                f.write(text)
        print(
            f"{out_csv}: scored {len(ids)} studies in {info['seconds']} s "
            f"(vLLM {served.version}), finish {info['finish_reason']}, "
            f"failed {info['failed']}"
        )


if __name__ == "__main__":
    command, args = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else (None, [])
    if command == "score" and args and JUDGE_URL:
        score_served([(f"{d}/gt.csv", f"{d}/pred.csv", f"{d}/out.csv") for d in args])
    elif len(sys.argv) == 4:
        main(*sys.argv[1:])
    else:
        raise SystemExit(__doc__)
