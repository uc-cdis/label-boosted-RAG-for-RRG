"""Public RAD-DINO (Pérez-García et al. 2025, microsoft/rad-dino): the CLS embedding.

Repeats the authors' ``rad_dino`` package at the pinned revision (110cbc18): the repo's
image processor in its PIL backend (resize shorter side to 518, bicubic, center-crop,
normalize), then ``Dinov2Model`` float32 ``last_hidden_state[:, 0]``. Bit-identical to
the package (tests/test_raddino.py pins two images). Not called directly: it loads by
hub name and preprocesses in-process (here DataLoader workers do it). The backend is
explicit because transformers 5 defaults to torchvision, which differs slightly. Later
card/package revisions would be a new encoder setting, not a silent update.

Added first: min-max to 8-bit (encoders/images.py), as RAD-DINO's training images were;
a no-op on full-range 8-bit images. Training resized with B-spline, the processor uses
bicubic (as for anyone using the reference).
"""

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from labrag.encoders.images import load_8bit

DIM = 768
TOKENS = 1 + (518 // 14) ** 2  # CLS + the 37 x 37 patch grid
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
    on_tokens: Callable[[int, torch.Tensor], None] | None = None,
) -> np.ndarray:
    """CLS embedding of each image, float32, shape (len(paths), 768), in input order.

    If ``on_tokens`` is given, it is called per batch with the batch's first row index
    and its full ``last_hidden_state`` (CLS + patches; float32, on ``device``).
    """
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
        hidden = model(pixel_values=pixels.to(device)).last_hidden_state
        cls = hidden[:, 0]
        out[start : start + len(cls)] = cls.float().cpu().numpy()
        if on_tokens is not None:
            on_tokens(start, hidden)
        start += len(cls)
    assert start == len(paths)
    return out
