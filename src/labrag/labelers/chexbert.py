"""CheXbert-14 labels (Smit et al. 2020) with the authors' code in external/CheXbert.

The model class and condition list are imported unmodified from the submodule. The
inference loop mirrors upstream ``src/label.py`` (``label`` + ``save_preds``), which
can't be called directly: it loads BERT by hub name, wraps the model in DataParallel,
and its tokenizer helper uses ``encode_plus``, removed in transformers 5. Our normalize()
and encode() were checked token for token against upstream's helpers run under
transformers 4.46 on all 46,759 CheXpert Plus Findings (2026-11-11: no differences);
tests/test_chexbert.py reproduces upstream's own labeled_reports.csv.
"""

import importlib.util
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from labrag import paths

UPSTREAM_SRC = paths.REPO_ROOT / "external" / "CheXbert" / "src"
MAX_TOKENS = 512  # upstream cuts every report at 512 tokens, keeping the final [SEP]
# Upstream constants.BATCH_SIZE. Batching only pads (padding is masked out), so it
# changes results at most through float rounding.
BATCH_SIZE = 18


def _load_upstream(module: str):
    """Import one upstream file under a private name (its src/ isn't a package, and
    its module names, e.g. ``datasets``, would shadow installed packages)."""
    path = UPSTREAM_SRC / f"{module}.py"
    spec = importlib.util.spec_from_file_location(
        f"chexbert_upstream_{path.stem}", path
    )
    mod = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        # bert_tokenizer.py writes the regex '\s+' without r"" (harmless)
        warnings.simplefilter("ignore", SyntaxWarning)
        spec.loader.exec_module(mod)
    return mod


CONDITIONS: list[str] = _load_upstream("constants").CONDITIONS


def normalize(texts: pd.Series) -> pd.Series:
    """Upstream bert_tokenizer.get_impressions_from_csv, minus the CSV read."""
    imp = texts.str.strip()
    imp = imp.replace("\n", " ", regex=True)
    imp = imp.replace(r"\s+", " ", regex=True)
    return imp.str.strip()


def encode(text: str, tokenizer) -> list[int]:
    """Upstream bert_tokenizer.tokenize for one report.

    There, ``encode_plus`` on a list of wordpieces maps them to ids and adds [CLS] and
    [SEP]; an empty report becomes [CLS, SEP]; over-long ones are cut to 511 + [SEP].
    """
    pieces = tokenizer.tokenize(text)
    ids = [tokenizer.cls_token_id]
    ids += tokenizer.convert_tokens_to_ids(pieces) + [tokenizer.sep_token_id]
    if len(ids) > MAX_TOKENS:
        ids = ids[: MAX_TOKENS - 1] + [tokenizer.sep_token_id]
    return ids


def load_model(chexbert_pth: Path, bert_dir: Path, device: torch.device):
    """The upstream bert_labeler with the CheXbert checkpoint, plus its tokenizer."""
    from transformers import BertTokenizer

    bert_labeler = _load_upstream("models/bert_labeler").bert_labeler
    model = bert_labeler(pretrain_path=str(bert_dir))
    checkpoint = torch.load(chexbert_pth, map_location="cpu", weights_only=True)
    state = {
        k.removeprefix("module."): v for k, v in checkpoint["model_state_dict"].items()
    }
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, BertTokenizer.from_pretrained(str(bert_dir))


@torch.inference_mode()
def predict(texts: Sequence[str], model, tokenizer, device, batch_size=BATCH_SIZE):
    """Class index per report and condition, shape (n, 14), in upstream's class order:
    0 blank, 1 positive, 2 negative, 3 uncertain (No Finding: 0 blank, 1 positive)."""
    encoded = [torch.tensor(encode(t, tokenizer)) for t in normalize(pd.Series(texts))]
    preds = []
    for start in range(0, len(encoded), batch_size):
        batch = encoded[start : start + batch_size]
        ids = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
        mask = torch.zeros(ids.shape, dtype=torch.float)
        for i, seq in enumerate(batch):
            mask[i, : len(seq)] = 1
        out = model(ids.to(device), mask.to(device))
        preds.append(torch.stack([head.argmax(dim=1) for head in out], dim=1).cpu())
    return torch.cat(preds).numpy() if preds else np.zeros((0, len(CONDITIONS)), int)


def to_chexpert(classes: np.ndarray) -> pd.DataFrame:
    """Upstream save_preds coding: 1 positive, 0 negative, -1 uncertain, NaN blank."""
    df = pd.DataFrame(classes, columns=CONDITIONS).astype(float)
    # Same order as upstream: blank -> NaN before negative (2) takes the value 0.
    return df.replace(0, np.nan).replace(3, -1).replace(2, 0)
