"""Stage ``embed``: one image embedding per study.

Writes ``embeddings/<dataset>/<encoder>/``:
- ``embeddings.npy``: float32, shape (n_studies, dim), row i = study i of studies.csv;
- ``studies.csv``: study_id and the image_path that was embedded.

Every study of the dataset is embedded (all splits, with or without report text); later
stages pick their rows by study_id. The image is the study's preferred one, the same
row that gives its canonical report text (the adapter's load_studies).
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


def run(
    dataset: str,
    encoder: str = "raddino",
    overwrite: bool = False,
    cxr_root: Path | None = None,
) -> Path:
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

    def build(out: Path) -> dict:
        studies = adapter.load_studies(cxr_root)[["study_id", "image_path"]]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_workers = min(MAX_WORKERS, max(1, len(os.sched_getaffinity(0)) - 1))
        model, processor = raddino.load_model(model_dir, device)
        emb = raddino.embed(
            [image_root / p for p in studies.image_path],
            model,
            processor,
            device,
            num_workers=num_workers,
            progress=True,
        )
        assert np.isfinite(emb).all(), "non-finite embeddings"
        np.save(out / "embeddings.npy", emb)
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

    return artifacts.publish(
        f"embeddings/{dataset}/{encoder}", config, inputs, build, overwrite
    )
