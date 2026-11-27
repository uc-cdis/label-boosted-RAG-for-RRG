"""GRPO with LoRA on a vision LLM, CXRMate-2's reward, TRL's GRPOTrainer.

Run as ``run.py /work`` on the job's GPUs: re-launches itself under accelerate, one
process per visible GPU (DDP; each holds the frozen model), vLLM colocated on each.
/work holds:
- config.json: model (dir under /models), lora (peft.LoraConfig kwargs), grpo
  (GRPOConfig kwargs), reward_url, reward_weights {ratescore, cxrbert, bertscore, arn};
- train.jsonl: one row per study: study_id, prompt (chat messages, the image as
  {"type": "image"}), image (PNG path relative to /work, the pixels generate sends),
  reference (the report text scored);
and gets out/ (GRPOConfig.output_dir): LoRA checkpoints and trainer_state.json.

The reward scores the report after the last </think> as evaluate scores it (reports.py,
copied from labrag): headers removed, EMPTY if none (thinking cut by the length cap).
The four rewards come from the reward service, once per batch; the trainer z-scores
each per group and sums them with the weights, unscaled (trl.patch), as upstream.
"""

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

WORKER = (
    "--worker"  # a training process (accelerate's single-process launch sets no rank)
)
WORK = Path(sys.argv[-1])
REWARDS = ("ratescore", "cxrbert", "bertscore", "arn")  # upstream's order

if sys.argv[1:] != [WORKER, str(WORK)]:  # the launcher: one process per GPU
    import torch

    n = torch.cuda.device_count()
    cmd = [
        "accelerate", "launch", "--num_processes", str(n), "--num_machines", "1",
        "--mixed_precision", "bf16", "--dynamo_backend", "no",
        __file__, WORKER, str(WORK),
    ]  # fmt: skip
    print("launching:", " ".join(cmd), flush=True)
    sys.exit(subprocess.run(cmd).returncode)

import torch  # noqa: E402
from datasets import Dataset, Image  # noqa: E402
from peft import LoraConfig  # noqa: E402
from reports import EMPTY_REPORT, format_joint  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

config = json.loads((WORK / "config.json").read_text())
THINK_END = "</think>"


def report_text(completion: list[dict]) -> str:
    """The text the reward scores: the report after the thinking, as evaluate scores."""
    content = completion[0]["content"]
    report = content.rsplit(THINK_END, 1)[1] if THINK_END in content else ""
    return format_joint(report)[0] or EMPTY_REPORT


class Rewards:
    """The reward service's four rewards, fetched once for a batch of completions."""

    def __init__(self, url: str, group_size: int):
        self.url, self.group_size = url, group_size
        self.key, self.scores = None, None

    def get(self, name: str, completions: list, reference: list[str]) -> list[float]:
        predictions = [report_text(c) for c in completions]
        references = [" ".join(r.split()) for r in reference]
        key = (tuple(predictions), tuple(references))
        if key != self.key:
            body = json.dumps(
                {
                    "predictions": predictions,
                    "references": references,
                    "group_size": self.group_size,
                }
            ).encode()
            request = urllib.request.Request(
                f"{self.url}/score", body, {"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(request, timeout=3600) as r:
                self.scores = json.loads(r.read())
            self.key = key
        return self.scores[name]


rewards = Rewards(config["reward_url"], config["grpo"]["num_generations"])


def reward_func(name: str):
    def f(prompts, completions, reference, **kwargs):
        return rewards.get(name, completions, reference)

    f.__name__ = name  # TRL logs rewards/<name>/mean
    return f


rows = [json.loads(line) for line in open(WORK / "train.jsonl")]
for r in rows:
    r["image"] = str(WORK / r["image"])
dataset = Dataset.from_list(rows).cast_column("image", Image())

processor = AutoProcessor.from_pretrained(config["model"])
model = AutoModelForImageTextToText.from_pretrained(
    config["model"], dtype=torch.bfloat16, attn_implementation="sdpa"
)
args = GRPOConfig(
    output_dir=str(WORK / "out"),
    reward_weights=[config["reward_weights"][k] for k in REWARDS],
    **config["grpo"],
)
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_func(k) for k in REWARDS],
    args=args,
    train_dataset=dataset,
    processing_class=processor,
    peft_config=LoraConfig(**config["lora"]),
)
trainer.train(resume_from_checkpoint=config.get("resume"))
trainer.save_model(str(WORK / "out" / "final"))
