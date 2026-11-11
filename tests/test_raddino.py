"""RAD-DINO encoder: parity with the authors' inference path (rad_dino package)."""

import numpy as np
import pytest
import torch

from labrag import paths
from labrag.encoders import raddino

MODEL_DIR = paths.MODELS_ROOT / "microsoft" / "rad-dino"
IMAGES = paths.CXR_ROOT / "chexpertplus" / "PNG" / "train"
FRONTAL = IMAGES / "patient00001" / "study1" / "view1_frontal.png"
LATERAL = IMAGES / "patient00002" / "study1" / "view2_lateral.png"

# CLS embeddings of FRONTAL and LATERAL from the authors' path (RadDino.extract_cls_token:
# AutoImageProcessor use_fast=False + AutoModel, last_hidden_state[:, 0]) under
# transformers 4.46.3 and torch 2.15 on CPU, one image at a time (2026-11-11): the first
# six values and the L2 norm. On 100 CheXpert Plus images that day, our embed() gave
# bit-identical pixels and CLS embeddings.
UPSTREAM = {
    FRONTAL: (
        [0.20607009, 0.30027437, 0.06580535, -0.02850654, -0.34228405, -0.85442424],
        11.32204437,
    ),
    LATERAL: (
        [-0.40681791, -0.25216761, 0.44210935, -0.02140008, -0.08878567, 0.55302155],
        19.27184868,
    ),
}

needs_weights_and_images = pytest.mark.skipif(
    not (MODEL_DIR / "model.safetensors").is_file() or not FRONTAL.is_file(),
    reason="RAD-DINO weights or CheXpert Plus images not mounted",
)


@pytest.fixture(scope="module")
def model_and_processor():
    return raddino.load_model(MODEL_DIR, torch.device("cpu"))


@needs_weights_and_images
def test_matches_upstream_values(model_and_processor):
    model, processor = model_and_processor
    emb = raddino.embed([FRONTAL, LATERAL], model, processor, torch.device("cpu"))
    assert emb.shape == (2, raddino.DIM) and emb.dtype == np.float32
    for row, (head, norm) in zip(emb, UPSTREAM.values()):
        np.testing.assert_allclose(row[:6], head, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(row), norm, rtol=1e-5)


@needs_weights_and_images
def test_batching_and_order(model_and_processor):
    model, processor = model_and_processor
    paths_ = [LATERAL, FRONTAL, LATERAL]
    batched = raddino.embed(paths_, model, processor, torch.device("cpu"), batch_size=2)
    single = raddino.embed(paths_, model, processor, torch.device("cpu"), batch_size=1)
    np.testing.assert_allclose(batched, single, atol=1e-5)
    np.testing.assert_array_equal(batched[0], batched[2])
