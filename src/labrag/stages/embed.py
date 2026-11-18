"""Stage ``embed``: one image embedding per study.

Writes ``embeddings/<dataset>/<encoder>/``:
- ``embeddings.npy``: float32, shape (n_studies, dim), row i = study i of studies.csv;
- ``studies.csv``: study_id and the image_path that was embedded.

Every study of the dataset is embedded (all splits, with or without report text); later
stages pick their rows by study_id. The image is the study's preferred one, the same
row that gives its canonical report text (the adapter's load_studies).

``--feature tokens`` (with ``--section``) instead writes
``embeddings/<dataset>/<encoder>.tokens/<section>/``: the full last hidden state (CLS +
patch tokens) of the studies with that section's text, cast to float16 and stored
packed in TOKEN_FORMAT (quant.create_store): the same tokens classify gets by packing a
float16 store. Without ``--section``, ``embeddings/<dataset>/<encoder>.tokens/frontal/``
holds every study whose preferred image is frontal (FRONTAL_VIEWS), with or without
report text: the classifiers learn every section's labels from it. A labels-only
dataset (PadChest) takes no ``--section``: ``embeddings/<dataset>/<encoder>.tokens/``
holds every study (all frontal).
"""

import os
from pathlib import Path

import numpy as np
import torch

from labrag import artifacts, paths, weights
from labrag.classifiers import quant
from labrag.data import DATASETS, labels_only
from labrag.encoders import raddino

# encoder name -> configs/weights.yaml entry
ENCODERS = {"raddino": "rad-dino"}
MAX_WORKERS = 32  # image decoding + resizing is the bottleneck, not the GPU
FEATURES = ("cls", "tokens")
TOKEN_FORMAT = "nvfp4"  # the classifiers' format, ~1/3.5 the size of float16
TOKENS_FEATURE = (
    "tokens: last_hidden_state (CLS + patches) as float16, stored as "
    + quant.DESCRIPTION[TOKEN_FORMAT]
)


def run(
    dataset: str,
    encoder: str = "raddino",
    feature: str = "cls",
    section: str | None = None,
    overwrite: bool = False,
    cxr_root: Path | None = None,
) -> Path:
    if labels_only(dataset):
        if section is not None:
            raise SystemExit(f"embed: {dataset} has no report sections; drop --section")
    elif section is not None and feature != "tokens":
        raise SystemExit("embed: --section goes with --feature tokens only")
    cxr_root = cxr_root or paths.CXR_ROOT
    adapter = DATASETS[dataset]
    model_dir, weights_record = weights.resolve(ENCODERS[encoder])
    image_root = cxr_root / adapter.IMAGE_ROOT
    inputs = {
        "sources": [artifacts.file_record(cxr_root / f) for f in adapter.SOURCE_FILES],
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
    frontal = feature == "tokens" and section is None and not labels_only(dataset)
    if feature == "tokens" and labels_only(dataset):
        config["feature"] = TOKENS_FEATURE
        relpath = f"embeddings/{dataset}/{encoder}.tokens"
    elif frontal:
        config |= {
            "feature": TOKENS_FEATURE,
            "studies": f"preferred image frontal (view in {list(adapter.FRONTAL_VIEWS)})",
        }
        relpath = f"embeddings/{dataset}/{encoder}.tokens/frontal"
    elif feature == "tokens":
        config |= {
            "feature": TOKENS_FEATURE,
            "section": section,
            "studies": f"with {section} text",
        }
        relpath = f"embeddings/{dataset}/{encoder}.tokens/{section}"

    def build(out: Path) -> dict:
        studies = adapter.load_studies(cxr_root)
        if section is not None:
            studies = studies[studies[section].notna()].reset_index(drop=True)
        if frontal:
            studies = studies[studies.view.isin(adapter.FRONTAL_VIEWS)]
            studies = studies.reset_index(drop=True)
        studies = studies[["study_id", "image_path"]]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_workers = min(MAX_WORKERS, max(1, len(os.sched_getaffinity(0)) - 1))
        model, processor = raddino.load_model(model_dir, device)
        store = on_tokens = None
        if feature == "tokens":
            shape = (len(studies), raddino.TOKENS, raddino.DIM)
            store = quant.create_store(out, TOKEN_FORMAT, shape)

            def on_tokens(start: int, hidden: torch.Tensor) -> None:
                half = hidden.half()
                assert torch.isfinite(half).all(), "non-finite tokens"
                quant.write_rows(store, start, quant.pack(TOKEN_FORMAT, half))

        emb = raddino.embed(
            [image_root / p for p in studies.image_path],
            model,
            processor,
            device,
            num_workers=num_workers,
            progress=True,
            on_tokens=on_tokens,
        )
        assert np.isfinite(emb).all(), "non-finite embeddings"
        if store is None:
            np.save(out / "embeddings.npy", emb)
        else:
            for part in store.values():
                part.flush()
            del store
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
