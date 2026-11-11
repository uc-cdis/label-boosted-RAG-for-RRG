"""Shared image loading: min-max scaling to 8-bit grayscale."""

import numpy as np
import pytest
from PIL import Image

from labrag import paths
from labrag.encoders import images

CHEXPERTPLUS = (
    paths.CXR_ROOT / "chexpertplus/PNG/train/patient00001/study1/view1_frontal.png"
)
PADCHEST_DIR = paths.CXR_ROOT / "padchest" / "images"


def test_full_range_uint8_is_unchanged():
    pixels = np.random.default_rng(0).integers(0, 256, (37, 41), dtype=np.uint8)
    pixels[0, 0], pixels[0, 1] = 0, 255
    np.testing.assert_array_equal(images.min_max_8bit(pixels), pixels)


def test_16bit_is_scaled_to_full_range():
    pixels = np.array([[1000, 2000], [3000, 65000]], dtype=np.uint16)
    out = images.min_max_8bit(pixels)
    assert out.dtype == np.uint8 and out.min() == 0 and out.max() == 255
    np.testing.assert_array_equal(out, [[0, 4], [8, 255]])


def test_constant_image_is_zeros():
    out = images.min_max_8bit(np.full((3, 3), 700, dtype=np.uint16))
    assert out.dtype == np.uint8 and not out.any()


def test_16bit_png_round_trip(tmp_path):
    path = tmp_path / "x.png"
    pixels = np.linspace(500, 60000, 64 * 64).reshape(64, 64).astype(np.uint16)
    Image.fromarray(pixels).save(path)
    with Image.open(path) as im:
        assert im.mode.startswith("I")  # stored as 16-bit
    out = images.load_8bit(path)
    assert out.mode == "L"
    np.testing.assert_array_equal(np.asarray(out), images.min_max_8bit(pixels))


def test_unknown_mode_is_refused(tmp_path):
    path = tmp_path / "rgb.png"
    Image.new("RGB", (4, 4)).save(path)
    with pytest.raises(ValueError, match="unexpected image mode"):
        images.load_8bit(path)


@pytest.mark.skipif(not CHEXPERTPLUS.is_file(), reason="CheXpert Plus not mounted")
def test_chexpertplus_image_is_unchanged():
    with Image.open(CHEXPERTPLUS) as im:
        raw = np.asarray(im)
    np.testing.assert_array_equal(np.asarray(images.load_8bit(CHEXPERTPLUS)), raw)


@pytest.mark.skipif(not PADCHEST_DIR.is_dir(), reason="PadChest not mounted")
def test_padchest_16bit_image_spans_8bit_range():
    path = next(PADCHEST_DIR.glob("*.png"))
    out = np.asarray(images.load_8bit(path))
    assert out.dtype == np.uint8 and out.min() == 0 and out.max() == 255
    assert (out == 255).mean() < 0.1  # not clipped white
