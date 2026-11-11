"""Reading a released CXR image file into the 8-bit grayscale image encoders take.

Every image is min-max scaled to the full 8-bit range [0, 255], whatever its stored bit
depth. This is RAD-DINO's own training preprocessing ("min-max scaled to [0, 255]",
model card), and it is needed: the released files differ in bit depth (sampled
2026-11-11, 150-500 images each):
- CheXpert Plus PNGs and MIMIC-CXR-JPG JPEGs: 8-bit grayscale, every sampled image
  already spanning exactly 0-255, so scaling changes nothing for them;
- PadChest (so also PadChest-GR) and ReXGradient PNGs: 16-bit grayscale. PIL's RGB
  conversion inside the encoders' processors clips these at 255, which turned 70-90% of
  each sampled image pure white. Scaling first avoids that.
Unknown image modes are refused, so a new dataset's format gets looked at, not guessed.
"""

from pathlib import Path

import numpy as np
from PIL import Image

GRAYSCALE_MODES = frozenset({"L", "I;16", "I;16B", "I;16L", "I"})


def load_8bit(path: Path) -> Image.Image:
    """The image at ``path`` as mode "L", min-max scaled to [0, 255] (a constant image
    becomes all zeros)."""
    with Image.open(path) as image:
        if image.mode not in GRAYSCALE_MODES:
            raise ValueError(f"{path}: unexpected image mode {image.mode!r}")
        pixels = np.asarray(image)
    return Image.fromarray(min_max_8bit(pixels), mode="L")


def min_max_8bit(pixels: np.ndarray) -> np.ndarray:
    """Min-max scale to uint8 [0, 255], rounding to nearest. Exact identity on uint8
    input that already spans 0-255."""
    lo, hi = int(pixels.min()), int(pixels.max())
    if hi == lo:
        return np.zeros(pixels.shape, dtype=np.uint8)
    scaled = (pixels.astype(np.float64) - lo) * (255.0 / (hi - lo))
    return np.rint(scaled).astype(np.uint8)
