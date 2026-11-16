"""Reading a released CXR image into the 8-bit grayscale image encoders take.

Every image is min-max scaled to [0, 255] (RAD-DINO's training preprocessing). Needed:
PadChest(-GR) and ReXGradient PNGs are 16-bit, which the processors' RGB conversion
clips to mostly white; CheXpert Plus and MIMIC are full-range 8-bit (a no-op). Unknown
image modes are refused.

``resize_short_edge`` is the resize of RAD-DINO's processor (shorter side to a size,
bicubic) without its center crop.
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


def resize_short_edge(image: Image.Image, size: int) -> Image.Image:
    """``image`` scaled so its shorter side is ``size`` (aspect kept), bicubic."""
    w, h = image.size
    scale = size / min(w, h)
    return image.resize((round(w * scale), round(h * scale)), Image.Resampling.BICUBIC)
