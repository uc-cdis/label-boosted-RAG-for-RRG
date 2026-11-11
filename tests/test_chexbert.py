"""CheXbert labeler: parity with the authors' code and their own labelled examples."""

import numpy as np
import pandas as pd
import pytest

from labrag import paths, weights
from labrag.labelers import chexbert

UPSTREAM = chexbert.UPSTREAM_SRC
BERT_DIR = paths.MODELS_ROOT / "google-bert" / "bert-base-uncased"
CHEXBERT_PTH = paths.MODELS_ROOT / "StanfordAIMI" / "RRG_scorers" / "chexbert.pth"

needs_weights = pytest.mark.skipif(
    not (BERT_DIR.is_dir() and CHEXBERT_PTH.is_file()),
    reason="model weights not mounted",
)


def test_normalize_matches_upstream(tmp_path):
    texts = pd.Series([" 1. Left  effusion.\n\n2. No edema. ", "Clear\tlungs.\n", "x"])
    csv = tmp_path / "reports.csv"
    pd.DataFrame({"Report Impression": texts}).to_csv(csv, index=False)
    upstream = chexbert._load_upstream("bert_tokenizer").get_impressions_from_csv(csv)
    assert chexbert.normalize(texts).tolist() == upstream.tolist()


def test_to_chexpert_coding():
    classes = np.array([[0, 1, 2, 3] + [0] * 9 + [1]])
    row = chexbert.to_chexpert(classes).iloc[0]
    assert row.iloc[:4].tolist()[1:] == [1.0, 0.0, -1.0] and np.isnan(row.iloc[0])
    assert row["No Finding"] == 1.0
    assert list(chexbert.to_chexpert(classes).columns) == chexbert.CONDITIONS


@needs_weights
def test_encode_special_tokens_and_truncation():
    from transformers import BertTokenizer

    tok = BertTokenizer.from_pretrained(str(BERT_DIR))
    assert chexbert.encode("", tok) == [tok.cls_token_id, tok.sep_token_id]
    ids = chexbert.encode("No edema.", tok)
    assert tok.convert_ids_to_tokens(ids) == [
        "[CLS]",
        "no",
        "ed",
        "##ema",
        ".",
        "[SEP]",
    ]
    long = chexbert.encode("Heart size is normal. " * 300, tok)
    assert len(long) == chexbert.MAX_TOKENS and long[-1] == tok.sep_token_id


@needs_weights
def test_reproduces_upstream_labeled_reports():
    """The repo's labeled_reports.csv is upstream label.py's output on sample_reports.csv."""
    import torch

    device = torch.device("cpu")
    model, tok = chexbert.load_model(CHEXBERT_PTH, BERT_DIR, device)
    reports = pd.read_csv(UPSTREAM / "sample_reports.csv")["Report Impression"]
    expected = pd.read_csv(UPSTREAM / "labeled_reports.csv")[chexbert.CONDITIONS]
    got = chexbert.to_chexpert(chexbert.predict(reports.tolist(), model, tok, device))
    pd.testing.assert_frame_equal(got, expected)


@needs_weights
def test_configured_weights_resolve():
    for name in ("chexbert", "bert-base-uncased"):
        model_dir, record = weights.resolve(name)
        assert model_dir.is_dir() and record["files"]
