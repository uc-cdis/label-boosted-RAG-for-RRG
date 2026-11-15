"""Stage ``embed``: one image embedding per study.

Writes ``embeddings/<dataset>/<encoder>/``:
- ``embeddings.npy``: float32, shape (n_studies, dim), row i = study i of studies.csv;
- ``studies.csv``: study_id and the image_path that was embedded.

Every study of the dataset is embedded (all splits, with or without report text); later
stages pick their rows by study_id. The image is the study's preferred one, the same
row that gives its canonical report text (the adapter's load_studies).

``--feature tokens`` (with ``--section``) instead writes
``embeddings/<dataset>/<encoder>.tokens/<section>/tokens.npy``: the full last hidden
state (CLS + patch tokens), float16, shape (n_studies, tokens, dim), for the studies
with that section's text only (the full dataset would be ~4x larger).
"""

import os
from pathlib import Path

import numpy as np
import torch

from labrag import artifacts, paths, weights
from labrag.data import DATASETS
from labrag.encoders import raddino

# encoder name -> configs/weights.yaml entry
ENCODERS = {"raddino": "rad-dino"}
MAX_WORKERS = 32  # image decoding + resizing is the bottleneck, not the GPU
FEATURES = ("cls", "tokens")


def run(
    dataset: str,
    encoder: str = "raddino",
    feature: str = "cls",
    section: str | None = None,
    overwrite: bool = False,
    cxr_root: Path | None = None,
) -> Path:
    if (feature == "tokens") != (section is not None):
        raise SystemExit(
            "embed: --section goes with --feature tokens, and only with it"
        )
    cxr_root = cxr_root or paths.CXR_ROOT
    adapter = DATASETS[dataset]
    model_dir, weights_record = weights.resolve(ENCODERS[encoder])
    image_root = cxr_root / adapter.IMAGE_ROOT
    inputs = {
        "sources": [artifacts.file_record(cxr_root / adapter.SOURCE_CSV)],
        "images": {"root": str(image_root)},  # too many to hash; paths in studies.csv
        "weights": weights_record,
    }
    config = {
        "stage": "embed",
        "dataset": dataset,
        "encoder": encoder,
        "image": "preferred per study",
        "pixels": "min-max to 8-bit grayscale",
        "feature": "cls",
    }
    relpath = f"embeddings/{dataset}/{encoder}"
    if feature == "tokens":
        config |= {
            "feature": "tokens: last_hidden_state (CLS + patches), float16",
            "section": section,
            "studies": f"with {section} text",
        }
        relpath = f"embeddings/{dataset}/{encoder}.tokens/{section}"

    def build(out: Path) -> dict:
        studies = adapter.load_studies(cxr_root)
        if section is not None:
            studies = studies[studies[section].notna()].reset_index(drop=True)
        studies = studies[["study_id", "image_path"]]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_workers = min(MAX_WORKERS, max(1, len(os.sched_getaffinity(0)) - 1))
        model, processor = raddino.load_model(model_dir, device)
        tokens = None
        if feature == "tokens":
            tokens = np.lib.format.open_memmap(
                out / "tokens.npy",
                mode="w+",
                dtype=np.float16,
                shape=(len(studies), raddino.TOKENS, raddino.DIM),
            )
        emb = raddino.embed(
            [image_root / p for p in studies.image_path],
            model,
            processor,
            device,
            num_workers=num_workers,
            progress=True,
            tokens=tokens,
        )
        assert np.isfinite(emb).all(), "non-finite embeddings"
        if tokens is None:
            np.save(out / "embeddings.npy", emb)
        else:
            tokens.flush()
            del tokens
        studies.to_csv(out / "studies.csv", index=False)
        return {
            "n_studies": len(studies),
            "dim": emb.shape[1],
            "device": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
            "batch_size": raddino.BATCH_SIZE,
            "num_workers": num_workers,
        }

    return artifacts.publish(relpath, config, inputs, build, overwrite)
