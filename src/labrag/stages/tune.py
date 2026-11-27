"""Stage ``tune``: a LoRA adapter for a generator, GRPO with CXRMate-2's reward.

Writes ``adapters/train/<datasets>/<section>/<condition>/<prompt>/<model>.<config>/``
(datasets joined by ``-``, the prompt artifacts' path otherwise, as generations' depth):
checkpoints/<step>/ (adapter_model.safetensors, adapter_config.json: every saved
step, for choosing one on val), log_history.jsonl (TRL's per-step log: rewards, KL,
clip ratios, lengths).

``--prompts``: one train prompt artifact per dataset (build-prompts --experiment
train), the same section, condition and prompt version; their train targets are the
studies (``configs/tuning.yaml``'s ``studies``: all, or n drawn with the seed), each
with its prompt, its reference and its image prepared as generate sends it (the model
entry's short_edge, PNG bytes of generate.image_png).

The trainer and the reward service are tuning/compose.yaml's services, built from this
checkout and tagged with the commit: TRL's GRPOTrainer on every GPU of the job (DDP,
vLLM colocated), CXRMate-2's rewards next to it. Work happens on node-local disk; the
artifact is published once at the end. No resume across jobs.
"""

import concurrent.futures
import fcntl
import gzip
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path

import yaml

from labrag import artifacts, paths, serving, weights
from labrag.stages.generate import image_png

COMPOSE_FILE = paths.REPO_ROOT / "tuning" / "compose.yaml"
SERVICES = ("grpo", "grpo-reward")
REWARD_PORT = 8000  # compose.yaml's grpo-reward command
# what the reward service loads: upstream's models by hub id (server.py) + the ARN
# tokenizer
REWARD_WEIGHTS = (
    "grpo-reward.roberta-large",
    "grpo-reward.cxr-bert",
    "ratescore.rate-ner-deberta",
    "ratescore.biolord-2023-c",
    "grpo-reward.cxrmate-2-tokenizer",
)


def load_config() -> dict:
    return yaml.safe_load((paths.CONFIGS_DIR / "tuning.yaml").read_text())


def run(prompts: list[str], config: str, overwrite: bool = False) -> Path:
    recipe = load_config()[config]
    model_cfg = serving.load_config("models")[recipe["model"]]
    sampling = serving.load_config("sampling")[recipe["sampling"]]
    rels = [p.strip("/").removeprefix("prompts/") for p in prompts]
    parts = [r.split("/") for r in rels]
    if any(len(p) != 5 or p[0] != "train" for p in parts):
        raise SystemExit("tune: --prompts are prompts/train/<dataset>/<section>/...")
    datasets = [p[1] for p in parts]
    if len({tuple(p[2:]) for p in parts}) != 1 or len(set(datasets)) != len(parts):
        raise SystemExit("tune: one prompt artifact per dataset, the same otherwise")
    if "vision" not in model_cfg:
        raise SystemExit(
            f"tune: {recipe['model']} has no vision (the prompts show images)"
        )
    section, condition, prompt = parts[0][2:]
    rel = (
        f"adapters/train/{'-'.join(sorted(datasets))}/{section}/{condition}/{prompt}/"
        f"{recipe['model']}.{config}"
    )
    params = {k: v for k, v in sampling["params"].items() if k != "max_tokens"}
    config_record = {
        "stage": "tune",
        "prompts": [f"prompts/{r}" for r in sorted(rels)],
        "recipe": {"name": config, **recipe},
        "model": {"name": recipe["model"], **model_cfg},
        "sampling": {"name": recipe["sampling"], **sampling},
        "image": {"short_edge": model_cfg["vision"]["short_edge"], "as": "generate"},
        "images": image_sources(),
    }
    if artifacts.is_built(rel, config_record, overwrite):
        return paths.SHARED_ROOT / rel

    # Check every input before taking the GPUs.
    inputs = {"prompts": {}, "weights": {}}
    rows = []
    for r in sorted(rels):
        prompt_dir = paths.SHARED_ROOT / "prompts" / r
        inputs["prompts"][r] = artifacts.artifact_record(prompt_dir)
        with gzip.open(prompt_dir / "prompts.jsonl.gz", "rt") as f:
            rows += [x for line in f if (x := json.loads(line))["set"] == "train"]
    if recipe["studies"]:
        rng = random.Random(recipe["studies"]["seed"])
        rows = rng.sample(rows, recipe["studies"]["n"])
    model_dir, inputs["weights"]["model"] = weights.resolve(model_cfg["weights"])
    for name in REWARD_WEIGHTS:
        inputs["weights"][name] = weights.resolve(name)[1]
    missing = [x["image"] for x in rows if not (paths.CXR_ROOT / x["image"]).is_file()]
    if missing:
        raise SystemExit(f"tune: {len(missing)} images missing: {missing[:3]}")
    devices = serving.gpu_devices()

    def build(out: Path) -> dict:
        images = build_images()
        serving.raise_on_sigterm()
        ncpu = int(
            os.environ.get("SLURM_CPUS_PER_TASK") or len(os.sched_getaffinity(0))
        )
        paths.RUN_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=paths.RUN_ROOT, prefix="labrag-tune-"
        ) as work:
            work = Path(work)
            write_inputs(work, rows, model_cfg["vision"]["short_edge"], ncpu)
            grpo = {
                **recipe["grpo"],
                **params,
                "chat_template_kwargs": sampling["chat_template_kwargs"],
            }
            # vLLM's own sampling parameters go in generation_kwargs
            grpo["generation_kwargs"] = {
                "presence_penalty": grpo.pop("presence_penalty")
            }
            (work / "config.json").write_text(
                json.dumps(
                    {
                        "model": f"/models/{model_dir.relative_to(paths.MODELS_ROOT)}",
                        "lora": recipe["lora"],
                        "grpo": grpo,
                        "reward_url": f"http://grpo-reward:{REWARD_PORT}",
                        "reward_weights": recipe["reward_weights"],
                    },
                    indent=2,
                )
            )
            try:
                compose(
                    "run", "--rm", "--no-TTY", "--pull", "never",
                    "--user", f"{os.getuid()}:{os.getgid()}",
                    "--volume", f"{work}:/work",
                    "--env", f"OMP_NUM_THREADS={max(1, ncpu // (2 * len(devices)))}",
                    "grpo", "/work",
                    tag=images["tag"],
                )  # fmt: skip
            finally:
                compose("logs", "grpo-reward", tag=images["tag"], check=False)
                compose("down", "--timeout", "30", tag=images["tag"], check=False)
            return publish_outputs(work / "out", out, images, len(rows), devices)

    return artifacts.publish(rel, config_record, inputs, build, overwrite)


def write_inputs(work: Path, rows: list[dict], short_edge: int, ncpu: int) -> None:
    """train.jsonl and images/<i>.png (generate's bytes) under ``work``."""
    (work / "images").mkdir()

    def one(i: int) -> None:
        png = image_png(paths.CXR_ROOT / rows[i]["image"], short_edge)
        (work / "images" / f"{i}.png").write_bytes(png)

    with concurrent.futures.ThreadPoolExecutor(ncpu) as pool:
        list(pool.map(one, range(len(rows))))
    with open(work / "train.jsonl", "w") as f:
        for i, r in enumerate(rows):
            record = {
                "study_id": r["study_id"],
                "prompt": r["messages"],
                "image": f"images/{i}.png",
                "reference": r["reference"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def publish_outputs(
    trained: Path, out: Path, images: dict, n_studies: int, devices: list[str]
) -> dict:
    """Every checkpoint's adapter files and the trainer's log into ``out``."""
    steps = sorted(
        (int(d.name.rsplit("-", 1)[1]), d) for d in trained.glob("checkpoint-*")
    )
    if not steps:
        raise SystemExit("tune: the trainer saved no checkpoint")
    for step, d in steps:
        dest = out / "checkpoints" / str(step)
        dest.mkdir(parents=True)
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            (dest / name).write_bytes((d / name).read_bytes())
    state = json.loads((steps[-1][1] / "trainer_state.json").read_text())
    with open(out / "log_history.jsonl", "w") as f:
        for entry in state["log_history"]:
            f.write(json.dumps(entry) + "\n")
    return {
        "n_studies": n_studies,
        "devices": devices,
        "gpus": serving.gpu_names(devices),
        "checkpoints": [s for s, _ in steps],
        "global_step": state["global_step"],
        "images": images,
    }


def compose_service(name: str) -> dict:
    return yaml.safe_load(COMPOSE_FILE.read_text())["services"][name]


def image_sources() -> dict:
    """The git ids each image is built from (and the compose file it runs by)."""
    out = {
        "compose_blob": _git(
            "rev-parse", f"HEAD:{COMPOSE_FILE.relative_to(paths.REPO_ROOT)}"
        )
    }
    for name in SERVICES:
        context = COMPOSE_FILE.parent / compose_service(name)["build"]["context"]
        out[name] = _git(
            "rev-parse", f"HEAD:{context.resolve().relative_to(paths.REPO_ROOT)}"
        )
    out["upstream_commit"] = _git("rev-parse", "HEAD:external/cxrmate-2")
    out["reports_blob"] = _git("rev-parse", "HEAD:src/labrag/reports.py")
    return out


def build_images() -> dict:
    """Build both images from this checkout; their tag and image ids."""
    commit = artifacts.git_commit()
    tag = (
        "dev" if commit["dirty"] else commit["sha"]
    )  # a dirty tree builds what's on disk
    paths.RUN_ROOT.mkdir(parents=True, exist_ok=True)
    with open(paths.RUN_ROOT / "docker-build.lock", "w") as lock:  # one build per node
        fcntl.flock(lock, fcntl.LOCK_EX)
        compose("build", *SERVICES, tag=tag)
        ids = {}
        for name in SERVICES:
            ref = f"{compose_service(name)['image'].split(':')[0]}:{tag}"
            ids[name] = serving.image_id(ref)
            if ids[name] is None:
                raise SystemExit(f"{ref}: not found after the build")
    return {"tag": tag, "image_ids": ids}


def compose(*args: str, tag: str, check: bool = True) -> None:
    """docker compose as this job's own project, on the job's GPUs."""
    env = {
        **os.environ,
        "LABRAG_COMMIT": tag,
        "LABRAG_MODELS_ROOT": str(paths.MODELS_ROOT),
    }
    project = f"labrag-tuning-{os.environ.get('SLURM_JOB_ID') or os.getpid()}"
    cmd = ["docker", "compose", "-f", str(COMPOSE_FILE), "-p", project, *args]
    print("+", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, env=env)
    if check and result.returncode:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(paths.REPO_ROOT), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
