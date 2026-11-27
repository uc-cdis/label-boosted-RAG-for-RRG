# LLM fine-tuning

A LoRA adapter for the generator (Qwen3.5-27B with the image), trained with GRPO and
CXRMate-2's reward (Nicolson et al. 2026), so a tuned LaB-RAG can be compared with
CXRMate-2 trained by its own recipe. Recipes: [configs/tuning.yaml](../configs/tuning.yaml)
(which upstream settings we keep and which differ, and why).

## Services
- `grpo`: TRL's GRPOTrainer with PEFT LoRA, vLLM colocated for the rollouts, on the
  pinned vLLM image ([grpo/Dockerfile](grpo/Dockerfile)). [grpo/trl.patch](grpo/trl.patch)
  adds `normalize_then_sum_unscaled`: CXRMate-2's advantage (each reward z-scored
  within its group, weighted, summed, not rescaled over the batch). The reward scores
  the report after the thinking exactly as evaluate does (`src/labrag/reports.py`,
  copied into the image).
- `grpo-reward`: CXRMate-2's four rewards (RaTEScore, CXR-BERT, BERTScore, repeated
  3-grams) from upstream's own `rewards.py` and RaTEScore, unpatched, in CXRMate-2's
  env ([reward/requirements.txt](reward/requirements.txt)), over HTTP on a network only
  the trainer shares ([reward/server.py](reward/server.py)).

## How they run
`labrag tune --prompts <train prompt artifacts> --config <recipe>` checks the inputs
(weights by sha256), writes each study's image as generate sends it and the prompts to
node-local disk, builds both images from the checkout (tagged with the commit), runs
the trainer on every GPU of the job and publishes the adapter checkpoints to
`adapters/train/<datasets>/<section>/<condition>/<prompt>/<model>.<recipe>/`.
Containers need Docker, so they run in Slurm jobs, never in the dev container.

## Rules
As in eval/ and baselines/: upstream code is never forked or reimplemented. Weights come
from `/models`; with no network, the reward service maps upstream's hub ids to those
clones in an offline HF cache instead of changing upstream's code.
