"""Stage ``labels``: reference labels on each study's report section.

Writes ``labels/<dataset>/<section>/<labeler>/labels.csv``: one row per study with that
section, columns study_id + one per condition (1 positive, 0 negative, -1 uncertain,
blank = not mentioned).
"""

import subprocess
from pathlib import Path

import torch

from labrag import artifacts, paths, weights
from labrag.data import DATASETS
from labrag.labelers import chexbert

SECTIONS = ("findings",)
LABELERS = ("chexbert14",)


def run(
    dataset: str,
    section: str,
    labeler: str = "chexbert14",
    overwrite: bool = False,
    cxr_root: Path | None = None,
) -> Path:
    cxr_root = cxr_root or paths.CXR_ROOT
    adapter = DATASETS[dataset]
    chexbert_dir, chexbert_record = weights.resolve("chexbert")
    bert_dir, bert_record = weights.resolve("bert-base-uncased")
    code = subprocess.run(
        ["git", "-C", str(chexbert.UPSTREAM_SRC.parent), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    inputs = {
        "sources": [artifacts.file_record(cxr_root / adapter.SOURCE_CSV)],
        "chexbert_weights": chexbert_record,
        "bert_weights": bert_record,
        "chexbert_code": {"path": "external/CheXbert", "commit": code},
    }
    config = {
        "stage": "labels",
        "dataset": dataset,
        "section": section,
        "labeler": labeler,
    }

    def build(out: Path) -> dict:
        studies = adapter.load_studies(cxr_root)
        studies = studies[studies[section].notna()].reset_index(drop=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, tokenizer = chexbert.load_model(
            chexbert_dir / "chexbert.pth", bert_dir, device
        )
        classes = chexbert.predict(studies[section].tolist(), model, tokenizer, device)
        labels = chexbert.to_chexpert(classes)
        labels.insert(0, "study_id", studies.study_id)
        labels.to_csv(out / "labels.csv", index=False)
        return {"n_studies": len(labels), "device": str(device)}

    return artifacts.publish(
        f"labels/{dataset}/{section}/{labeler}", config, inputs, build, overwrite
    )
