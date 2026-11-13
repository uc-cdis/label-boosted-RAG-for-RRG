"""Stage ``generate``: one generation per prompt, from a model served by vLLM.

Reads prompt artifacts ``prompts/<path>/`` (stage build-prompts) and writes, per prompt
artifact, ``generations/<path>/<model>.<sampling>/<set>/generations.jsonl.gz``, one JSON
record per target study of that set, in prompt order:
- ``study_id``, ``set``;
- ``report``: the response text (the reasoning parser has removed the thinking); null
  if the output cap was hit before the model finished thinking;
- ``reasoning``: the thinking text (null without thinking);
- ``finish_reason``: ``stop`` or ``length`` (hit the output cap);
- ``prompt_tokens``, ``completion_tokens`` (from vLLM; completion includes thinking),
  ``reasoning_tokens`` (the thinking text re-tokenized; approximate);
- ``start_s`` (seconds after the job's first request) and ``latency_s``: when the
  request was sent and how long it took. Latency depends on the batch it shared, so it
  is not a cost; cost is the job's GPU time over its requests (manifest
  ``generation.gpu_seconds_per_request``, with the GPU model under ``server.gpus``).

Each artifact also holds ``server_metrics.prom``: vLLM's Prometheus metrics at the end
of the job (token totals, prefix-cache hits, preemptions, latency histograms). It is
job-level, the same file in every artifact of the job; its ``vllm:*_total`` counters
are summed into the manifest.

One server serves every prompt artifact of the job, and all their requests share one
queue, so a long-thinking tail is paid once, not once per artifact. Artifacts already
built with the same config are skipped; if all are, no server is started. No resume
yet: a failed job publishes nothing and is rerun whole.

Models: configs/models.yaml; sampling: configs/sampling.yaml; images:
configs/containers.yaml; weights: configs/weights.yaml.
"""

import asyncio
import gzip
import json
import time
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI
from tokenizers import Tokenizer
from tqdm import tqdm

from labrag import artifacts, paths, serving, weights

CONCURRENCY = 256  # requests in flight; vLLM batches them
REQUEST_TIMEOUT_S = 3 * 3600
# vLLM is served without authentication, but the OpenAI client requires some value.
NO_AUTH = "none"


def run(
    prompts: list[str],
    target_set: str,
    model: str,
    sampling: str,
    overwrite: bool = False,
) -> list[Path]:
    model_cfg = serving.load_config("models")[model]
    sampling_cfg = serving.load_config("sampling")[sampling]
    container = serving.load_config("containers")[model_cfg["container"]]

    jobs = []  # (prompt relpath, output relpath, config)
    for rel in prompts:
        rel = rel.strip("/").removeprefix("prompts/")
        config = {
            "stage": "generate",
            "prompts": f"prompts/{rel}",
            "set": target_set,
            "model": {"name": model, **model_cfg},
            "container": {k: container[k] for k in ("image", "tag", "digest")},
            "sampling": {"name": sampling, **sampling_cfg},
        }
        out = f"generations/{rel}/{model}.{sampling}/{target_set}"
        jobs.append((f"prompts/{rel}", out, config))
    todo = [j for j in jobs if not artifacts.is_built(j[1], j[2], overwrite)]
    for _, out, _ in jobs:
        if all(out != t[1] for t in todo):
            print(f"SKIP: {paths.SHARED_ROOT / out} already built with this config")
    if not todo:
        return [paths.SHARED_ROOT / out for _, out, _ in jobs]

    # Check every input before taking the GPU.
    model_dir, weights_record = weights.resolve(model_cfg["weights"])
    image = serving.image_record(model_cfg["container"])
    loaded = []
    for prompt_rel, out, config in todo:
        prompt_dir = paths.SHARED_ROOT / prompt_rel
        record = artifacts.artifact_record(prompt_dir)
        with gzip.open(prompt_dir / "prompts.jsonl.gz", "rt") as f:  # one NFS read
            rows = [json.loads(line) for line in f]
        rows = [r for r in rows if r["set"] == target_set]
        if not rows:
            raise SystemExit(f"{prompt_rel}: no targets in set {target_set!r}")
        loaded.append((out, config, record, rows))

    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    with serving.vllm_server(
        model, model_dir, image, model_cfg["serve_args"]
    ) as server:
        started = artifacts.now()
        t0 = time.monotonic()
        everything = [row for _, _, _, rows in loaded for row in rows]
        results = asyncio.run(
            _generate(server["base_url"], model, sampling_cfg, everything, tokenizer)
        )
        seconds = time.monotonic() - t0
        metrics = serving.metrics_text(server["root_url"])  # job-level, for costs

    built, i = [], 0
    for out, config, record, rows in loaded:
        part, i = results[i : i + len(rows)], i + len(rows)
        inputs = {
            "prompts": record,
            "weights": weights_record,
            "container": {k: image[k] for k in ("ref", "image_id")},
        }
        extra = {
            "server": {
                "vllm_version": server["version"],
                "devices": server["devices"],
                "gpus": server["gpus"],
                "startup_s": server["startup_s"],
                "concurrency": CONCURRENCY,
            },
            "generation": {
                "seconds": round(seconds),
                "n_requests_in_job": len(everything),
                # Cost: the job's GPU time shared over all its requests (one queue).
                "gpu_seconds": round(seconds * len(server["devices"])),
                "gpu_seconds_per_request": seconds
                * len(server["devices"])
                / len(everything),
                "vllm_counters": serving.counter_totals(metrics) if metrics else None,
            },
        }

        def build(dest: Path, part=part, extra=extra) -> dict:
            with open(dest / "generations.jsonl.gz", "wb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as f:
                    for r in part:
                        f.write((json.dumps(r, ensure_ascii=False) + "\n").encode())
            if metrics:  # the same job-level snapshot in every artifact of the job
                (dest / "server_metrics.prom").write_text(metrics)
            summary = {**extra, "summary": summarize(part)}
            print(out, json.dumps(summary["summary"], indent=2))
            return summary

        built.append(
            artifacts.publish(out, config, inputs, build, overwrite, started=started)
        )
    return built


async def _generate(
    base_url: str, model: str, sampling: dict, rows: list[dict], tokenizer: Tokenizer
) -> list[dict]:
    client = AsyncOpenAI(
        base_url=base_url,
        api_key=NO_AUTH,
        timeout=REQUEST_TIMEOUT_S,
        max_retries=0,
    )
    params = dict(sampling["params"])
    # Parameters the OpenAI client knows go as arguments; vLLM's own go in the body.
    known = {
        k: params.pop(k)
        for k in ("temperature", "top_p", "presence_penalty", "max_tokens")
        if k in params
    }
    body = {**params, "chat_template_kwargs": sampling["chat_template_kwargs"]}
    gate = asyncio.Semaphore(CONCURRENCY)
    t0 = time.monotonic()
    bar = tqdm(total=len(rows), desc="generations", mininterval=30)

    async def one(row: dict) -> dict:
        async with gate:
            sent = time.monotonic()
            resp = await client.chat.completions.create(
                model=model,
                messages=row["messages"],
                n=1,
                seed=sampling["seed"],
                extra_body=body,
                **known,
            )
            done = time.monotonic()
        bar.update()
        choice = resp.choices[0]
        reasoning = getattr(choice.message, "reasoning", None)
        if reasoning is None and choice.message.model_extra:
            reasoning = choice.message.model_extra.get("reasoning")
        return {
            "study_id": row["study_id"],
            "set": row["set"],
            "report": choice.message.content,
            "reasoning": reasoning,
            "finish_reason": choice.finish_reason,
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "start_s": round(sent - t0, 3),
            "latency_s": round(done - sent, 3),
            "reasoning_tokens": (
                len(tokenizer.encode(reasoning, add_special_tokens=False).ids)
                if reasoning
                else 0
            ),
        }

    try:
        return await asyncio.gather(*(one(r) for r in rows))
    finally:
        bar.close()
        await client.close()


def summarize(records: list[dict]) -> dict:
    """Counts and token statistics for the manifest."""

    def stats(values: list[int]) -> dict:
        v = np.asarray(values)
        return {
            "mean": float(v.mean()),
            "p50": float(np.percentile(v, 50)),
            "p95": float(np.percentile(v, 95)),
            "max": int(v.max()),
        }

    finish = {}
    for r in records:
        finish[r["finish_reason"]] = finish.get(r["finish_reason"], 0) + 1
    return {
        "n": len(records),
        "finish_reason": dict(sorted(finish.items())),
        "n_no_report": sum(not (r["report"] or "").strip() for r in records),
        "completion_tokens": stats([r["completion_tokens"] for r in records]),
        "reasoning_tokens": stats([r["reasoning_tokens"] for r in records]),
        "prompt_tokens": stats([r["prompt_tokens"] for r in records]),
    }
