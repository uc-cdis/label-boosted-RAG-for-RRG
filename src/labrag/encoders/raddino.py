"""Public RAD-DINO (Pérez-García et al. 2025, microsoft/rad-dino): the CLS embedding.

Reference: the authors' inference code, the ``rad_dino`` package in the HF repo
(src/rad_dino/__init__.py) at the revision whose weights we pin (110cbc18, see
configs/weights.yaml). The model card's usage section is just a call into it:
``RadDino().extract_features(image)``. Its steps, which this module repeats:
- the repo's image processor in its PIL backend (``use_fast=False``; transformers 5
  calls it ``backend="pil"``): convert to RGB, resize the shorter side to 518
  (bicubic), center-crop 518 x 518, normalize with the repo's mean/std;
- ``Dinov2Model`` in float32, keeping ``last_hidden_state[:, 0]`` (the CLS token).
Verified 2026-11-11 on CheXpert Plus images: bit-identical to the package run under our
transformers 5 (30 images) and to the same path under transformers 4.46 (100 images).
tests/test_raddino.py pins two images' reference values, so drift fails the tests.

Why not call the package: it loads weights by hub name, whereas jobs use pre-pulled
files checked by sha256 and never download; and it preprocesses in the calling process,
whereas here DataLoader workers do it (decoding and resizing ~3000 px PNGs is the
bottleneck). Why the backend is explicit: with torchvision installed, transformers 5
defaults to the torchvision backend, which differs slightly (CLS max |diff| 0.003 on the
same 30 images). The model card's older usage section (before Aug 2025) called the
processor that way, with ``pooler_output`` (identical to ``last_hidden_state[:, 0]``).

Later revisions of the model card or package are deliberately not followed: the
weights, the reference code and its outputs are pinned together, and a change of
preprocessing would be a new encoder setting, not a silent update.

One step added before the reference path: every image is first min-max scaled to 8-bit
grayscale (encoders/images.py), as RAD-DINO's training images were (model card,
"Preprocessing"). The reference path expects 8-bit input and silently clips 16-bit
PNGs (PadChest, ReXGradient). On 8-bit images that already span 0-255 (every sampled
CheXpert Plus and MIMIC-CXR image) the step is an exact no-op, so the parity above
holds there. Still different from training, as for anyone using the reference: training
resized to 518 with B-spline interpolation, the reference processor uses bicubic.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from labrag.encoders.images import load_8bit

DIM = 768
# Batching changes nothing here (checked on CPU: batch 1 vs 8, and vs 32, identical).
BATCH_SIZE = 32


def load_model(model_dir: Path, device: torch.device):
    """The model (float32, eval mode, on ``device``) and its image processor."""
    from transformers import AutoImageProcessor, AutoModel

    model = AutoModel.from_pretrained(str(model_dir)).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(str(model_dir), backend="pil")
    return model, processor


class _Images(Dataset):
    """Decodes and preprocesses one image per item (in DataLoader workers)."""

    def __init__(self, paths: Sequence[Path], processor):
        self.paths = list(paths)
        self.processor = processor

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        image = load_8bit(self.paths[i])
        return self.processor(image, return_tensors="pt")["pixel_values"][0]


@torch.inference_mode()
def embed(
    paths: Sequence[Path],
    model,
    processor,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
    progress: bool = False,
) -> np.ndarray:
    """CLS embedding of each image, float32, shape (len(paths), 768), in input order."""
    loader = DataLoader(
        _Images(paths, processor),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    if progress:
        from tqdm import tqdm

        loader = tqdm(loader, desc="rad-dino", unit="batch", mininterval=30)
    out = np.empty((len(paths), DIM), dtype=np.float32)
    start = 0
    for pixels in loader:
        cls = model(pixel_values=pixels.to(device)).last_hidden_state[:, 0]
        out[start : start + len(cls)] = cls.float().cpu().numpy()
        start += len(cls)
    assert start == len(paths)
    return out
