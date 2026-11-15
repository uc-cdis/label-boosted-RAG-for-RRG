"""Token storage formats: sizes, exact values, error bounds, row access."""

import pytest
import torch

from labrag.classifiers import quant


def _tokens(n=3, t=5, d=64, seed=0):
    return torch.randn(n, t, d, generator=torch.Generator().manual_seed(seed)) * 0.7


@pytest.mark.parametrize("fmt,bits", [("fp16", 16), ("fp8", 8.5), ("nvfp4", 5.0)])
def test_bits_per_value(fmt, bits):
    x = _tokens(d=768)
    parts = quant.pack(fmt, x)
    stored = sum(v.numel() * v.element_size() for v in parts.values()) * 8
    assert stored / x.numel() <= bits


@pytest.mark.parametrize(
    "fmt,max_rel", [("fp16", 1e-3), ("fp8", 0.04), ("nvfp4", 0.12)]
)
def test_roundtrip_error_and_determinism(fmt, max_rel):
    x = _tokens()
    y = quant.unpack(fmt, quant.pack(fmt, x))
    assert y.dtype == torch.float32 and y.shape == x.shape
    assert ((y - x).norm() / x.norm()).item() < max_rel
    assert torch.equal(y, quant.unpack(fmt, quant.pack(fmt, x)))


def test_nvfp4_grid_values_are_exact():
    # Each block holds e2m1 values times the token's step, its largest magnitude 6.
    values = torch.cat([quant.FP4_VALUES, -quant.FP4_VALUES])  # 16 = one block
    x = torch.stack([values * 0.125, values.flip(0) * 0.125])[None]  # (1, 2, 16)
    x = torch.cat([x / 4, x], dim=-1)  # first block 4x smaller: block scale 448 / 4
    assert torch.equal(quant.unpack("nvfp4", quant.pack("nvfp4", x)), x)


@pytest.mark.parametrize("fmt", quant.FORMATS)
def test_zero_tokens_stay_zero(fmt):
    x = torch.zeros(2, 3, 32)
    assert torch.equal(quant.unpack(fmt, quant.pack(fmt, x)), x)


@pytest.mark.parametrize("fmt", quant.FORMATS)
def test_take_rows_of_packed_storage(fmt):
    x = _tokens(n=6)
    store = quant.empty(fmt, tuple(x.shape), torch.device("cpu"))
    for k, v in quant.pack(fmt, x).items():
        store.parts[k][:] = v
    assert len(store) == 6 and store.dim == x.shape[-1]
    rows = torch.tensor([4, 0, 4])
    assert torch.equal(store.take(rows), quant.unpack(fmt, quant.pack(fmt, x))[rows])
