"""CXRMate-2 (Nicolson et al. 2026) on one dataset's test split, inside its image.

  docker compose -f baselines/compose.yaml run --rm cxrmate-2-prepare \
    prepare <dataset> <db_dir>
  docker compose -f baselines/compose.yaml run --rm cxrmate-2 \
    generate <dataset> <db_dir> <targets.txt> <out.csv>

<dataset>: mimic or rexgradient. ``prepare`` (CPU) runs upstream's prepare script on
the test split into <db_dir>: its Hugging Face dataset, with every study's images and
report sections. ``generate`` (one GPU) writes out.csv (study_id, findings, impression:
the columns of upstream's published outputs, and n_images: the study's current + prior
images) for the upstream study ids in targets.txt (one per line), in that order;
<out>.json holds the timing.

As upstream's test loop (stages_cxrmate2.py): CXRMate2Dataset with history 1 (the
latest earlier study's images and Findings/Impression), its test collate, batch size 1,
float32, greedy with the checkpoint's generation config (max_new_tokens 320). The
processor gets mimic_cxr_normalisation=False, as upstream's training and test did: the
prepare scripts normalise where needed (the checkpoint's processor_config.json says
True, for raw DICOMs). Two additions, for outputs that don't depend on the run:
- MIMIC-CXR: each study's images in dicom_id order (upstream's DuckDB LIST() has none);
  ReXGradient's are sorted by path in labrag.patch.
- random.seed(TRIAL) before each study: the processor samples 5 images of a study that
  has more (current + prior), so a shard's other studies don't change the choice. Such
  studies match the published outputs only by chance (upstream's random state is
  unknown).
OMP_NUM_THREADS must be set (prepare's workers, torch's threads).
"""

import json
import os
import random
import subprocess
import sys
import time

MODEL = "/models/aehrc/cxrmate-2"
UPSTREAM = "/opt/cxrmate-2"
# upstream's prepare script, its arguments and the dataset it saves, per dataset
PREPARE = {
    "mimic": (
        "prepare_mimic_cxr_jpg.py",
        ["--physionet_dir", "/physionet"],
        "mimic_cxr_jpg_dataset",
    ),
    "rexgradient": (
        "prepare_rexgradient.py",
        ["--rexgradient_dir", "/data/rexgradient"],
        "rexgradient_160k_dataset",
    ),
}
SPLIT = "test"
HISTORY = 1  # sft_public.yaml
TRIAL = 6  # the checkpoint's config.json; upstream seeds random with it
DEVICE = "cuda"  # "cpu" only for checks without a GPU


def threads() -> int:
    if not os.environ.get("OMP_NUM_THREADS", "").isdigit():
        raise SystemExit("set OMP_NUM_THREADS")
    return int(os.environ["OMP_NUM_THREADS"])


def prepare(dataset: str, db_dir: str) -> None:
    script, args, _ = PREPARE[dataset]
    cmd = [
        sys.executable,
        f"{UPSTREAM}/prepare_datasets/{script}",
        *args,
        "--database_dir",
        db_dir,
        "--splits",
        SPLIT,
        "--num_workers",
        str(threads()),
    ]
    start = time.time()
    # its directory first on sys.path: the scripts import their neighbours
    subprocess.run(cmd, check=True, cwd=f"{UPSTREAM}/prepare_datasets")
    print(f"prepared {dataset} {SPLIT} in {time.time() - start:.0f} s")


class ByDicomId:
    """A MIMIC-CXR split whose rows have their per-image lists in dicom_id order, sorted
    as read (no rewritten copy: the GPUs' containers share the prepared dataset)."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, key):
        study = self.dataset[key]
        if not isinstance(key, int):  # a column
            return study
        order = sorted(range(len(study["dicom_id"])), key=study["dicom_id"].__getitem__)
        for k in ("dicom_id", "images", "views"):
            study[k] = [study[k][i] for i in order]
        return study

    def __len__(self):
        return len(self.dataset)


def generate(dataset: str, db_dir: str, targets_txt: str, out_csv: str) -> None:
    torch_threads = threads()
    with open(targets_txt) as f:
        targets = [line.strip() for line in f if line.strip()]
    if len(set(targets)) != len(targets):
        raise SystemExit("duplicate target ids")

    import datasets
    import pandas as pd
    import torch
    import transformers

    sys.path.insert(0, UPSTREAM)
    from dataset import CXRMate2Dataset

    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise SystemExit("no GPU visible: run with the job's GPU")
    torch.set_num_threads(torch_threads)

    test_set = datasets.load_from_disk(os.path.join(db_dir, PREPARE[dataset][2]))[SPLIT]
    if dataset == "mimic":
        test_set = ByDicomId(test_set)
    ids = [str(i) for i in test_set["study_id"]]
    index = dict(zip(ids, range(len(ids)), strict=True))
    if missing := [t for t in targets if t not in index]:
        raise SystemExit(
            f"{len(missing)} targets not in the {SPLIT} split, e.g. {missing[0]}"
        )
    test_set = CXRMate2Dataset(test_set, HISTORY)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, dtype=torch.float32
    ).to(device=DEVICE)
    model.eval()
    generation_config = transformers.GenerationConfig.from_pretrained(MODEL)
    processor = transformers.AutoProcessor.from_pretrained(
        MODEL, trust_remote_code=True, mimic_cxr_normalisation=False
    )
    if processor.mimic_cxr_normalisation:
        raise SystemExit("the processor kept mimic_cxr_normalisation=True")

    rows = []
    start = time.time()
    for n, target in enumerate(targets, 1):
        example = test_set[index[target]]
        # upstream's test_collate_fn, for a batch of one
        batch = {k: [v] for k, v in example.items()}
        n_images = len(example["images"])
        random.seed(TRIAL)
        processed = processor(train=False, **batch).to(device=DEVICE)
        with torch.no_grad():
            generated_ids = model.generate(
                **processed, generation_config=generation_config
            )
        findings, impression = processor.split_and_decode_sections(generated_ids)
        rows.append(
            {
                "study_id": target,
                "findings": findings[0],
                "impression": impression[0],
                "n_images": n_images,
            }
        )
        if n % 100 == 0 or n == len(targets):
            rate = n / (time.time() - start)
            print(f"{n}/{len(targets)} studies, {rate:.2f}/s", flush=True)
    seconds = time.time() - start

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    timing = {
        "studies": len(rows),
        "seconds": round(seconds, 1),
        "device": torch.cuda.get_device_name(0) if DEVICE == "cuda" else DEVICE,
        "torch_threads": torch.get_num_threads(),
    }
    with open(os.path.splitext(out_csv)[0] + ".json", "w") as f:
        json.dump(timing, f, indent=2)
    print(
        f"generated {len(rows)} reports in {timing['seconds']} s on {timing['device']}"
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 3 and args[0] == "prepare" and args[1] in PREPARE:
        prepare(*args[1:])
    elif len(args) == 5 and args[0] == "generate" and args[1] in PREPARE:
        generate(*args[1:])
    else:
        raise SystemExit(__doc__)
