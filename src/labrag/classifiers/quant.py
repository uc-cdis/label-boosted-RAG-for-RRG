"""Storage formats for patch tokens: fp16, fp8 (e4m3) and NVFP4-style 4-bit.

Each format packs float tokens of shape (n, tokens, dim) into a few tensors with a
first axis of n, and unpacks any rows of them back to float32. Packing is deterministic,
so a classifier trained on fp16 tokens packed as fp8 sees exactly what it would see if
the tokens had been stored as fp8.

- ``fp16``: the embed stage's original store (one ``tokens.npy``).
- ``fp8``: e4m3, one fp32 scale per token (its max |value| maps to 448, e4m3's max).
- ``nvfp4``: e2m1 values in blocks of 16 along dim, each block with an e4m3 scale, and
  one fp32 scale per token (NVFP4's two-level scaling, with the per-tensor scale taken
  per token so rows pack independently). Rounds to the nearest e2m1 value, ties toward
  zero. 4.54 bits per value.

A packed store on disk is one ``tokens.<part>.npy`` per part (fp8 parts as uint8 bytes,
since numpy has no fp8), written row by row as the embed stage goes.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

FORMATS = ("fp16", "fp8", "nvfp4")
DESCRIPTION = {
    "fp16": "float16",
    "fp8": "float8 e4m3, fp32 scale per token (max |value| -> 448)",
    "nvfp4": "e2m1 in blocks of 16, e4m3 block scales, fp32 scale per token; "
    "nearest value, ties toward zero",
}
FP8_MAX = 448.0
FP4_MAX = 6.0
BLOCK = 16
# e2m1 magnitudes; a code is the magnitude's index, plus 8 when negative.
FP4_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FP4_BOUNDS = (FP4_VALUES[1:] + FP4_VALUES[:-1]) / 2
FP4_TABLE = torch.cat([FP4_VALUES, -FP4_VALUES])


@dataclass
class Packed:
    """Packed tokens: tensors whose first axis is the row (study)."""

    fmt: str
    parts: dict[str, torch.Tensor]

    def __len__(self) -> int:
        return len(next(iter(self.parts.values())))

    @property
    def dim(self) -> int:
        if self.fmt == "nvfp4":
            return self.parts["codes"].shape[-1] * 2
        return self.parts["x"].shape[-1]

    def take(self, rows: torch.Tensor) -> torch.Tensor:
        """Rows ``rows`` unpacked to float32, shape (len(rows), tokens, dim)."""
        return unpack(self.fmt, {k: v[rows] for k, v in self.parts.items()})


def empty(fmt: str, shape: tuple[int, int, int], device: torch.device) -> Packed:
    """Uninitialized storage for ``shape`` = (n, tokens, dim) in format ``fmt``."""
    n, t, d = shape
    shapes = {
        "fp16": {"x": ((n, t, d), torch.float16)},
        "fp8": {
            "x": ((n, t, d), torch.float8_e4m3fn),
            "scale": ((n, t, 1), torch.float32),
        },
        "nvfp4": {
            "codes": ((n, t, d // 2), torch.uint8),
            "block": ((n, t, d // BLOCK), torch.float8_e4m3fn),
            "scale": ((n, t, 1), torch.float32),
        },
    }[fmt]
    return Packed(
        fmt,
        {k: torch.empty(s, dtype=dt, device=device) for k, (s, dt) in shapes.items()},
    )


def pack(fmt: str, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Pack float tokens ``x`` of shape (..., dim) into format ``fmt``."""
    x = x.float()
    if fmt == "fp16":
        return {"x": x.half()}
    amax = x.abs().amax(-1, keepdim=True)
    if fmt == "fp8":
        scale = torch.where(amax > 0, amax / FP8_MAX, 1.0)
        return {"x": (x / scale).to(torch.float8_e4m3fn), "scale": scale}
    if fmt == "nvfp4":
        assert x.shape[-1] % BLOCK == 0, "dim must be a multiple of the block size"
        scale = torch.where(amax > 0, amax / (FP4_MAX * FP8_MAX), 1.0)
        blocks = x.unflatten(-1, (-1, BLOCK))
        block = (blocks.abs().amax(-1) / FP4_MAX / scale).to(torch.float8_e4m3fn)
        step = (block.float() * scale)[..., None]
        y = torch.where(step > 0, blocks / step, 0.0).clamp(-FP4_MAX, FP4_MAX)
        index = torch.bucketize(y.abs(), FP4_BOUNDS.to(y.device))
        codes = (index + 8 * (y < 0)).to(torch.uint8).flatten(-2)
        return {
            "codes": codes[..., 0::2] | (codes[..., 1::2] << 4),
            "block": block,
            "scale": scale,
        }
    raise ValueError(f"unknown token format {fmt!r}")


def unpack(fmt: str, parts: dict[str, torch.Tensor]) -> torch.Tensor:
    """Float32 tokens from ``pack(fmt, ...)``'s parts (any leading rows of them)."""
    if fmt == "fp16":
        return parts["x"].float()
    if fmt == "fp8":
        return parts["x"].float() * parts["scale"]
    if fmt == "nvfp4":
        packed = parts["codes"]
        codes = torch.stack([packed & 15, packed >> 4], -1).flatten(-2)
        values = FP4_TABLE.to(packed.device)[codes.long()]
        step = parts["block"].float() * parts["scale"]
        return (values.unflatten(-1, (-1, BLOCK)) * step[..., None]).flatten(-2)
    raise ValueError(f"unknown token format {fmt!r}")


def create_store(directory: Path, fmt: str, shape: tuple[int, int, int]) -> dict:
    """Writable memmaps, one per part, for ``shape`` = (n, tokens, dim) in ``fmt``."""
    layout = empty(fmt, shape, torch.device("meta")).parts
    return {
        k: np.lib.format.open_memmap(
            directory / f"tokens.{k}.npy",
            mode="w+",
            dtype=np.uint8 if v.dtype == torch.float8_e4m3fn else _NUMPY[v.dtype],
            shape=tuple(v.shape),
        )
        for k, v in layout.items()
    }


def write_rows(store: dict, start: int, parts: dict[str, torch.Tensor]) -> None:
    """Write ``pack``'s parts for rows ``start``... into a store."""
    for k, v in parts.items():
        v = v.cpu()
        if v.dtype == torch.float8_e4m3fn:
            v = v.view(torch.uint8)
        store[k][start : start + len(v)] = v.numpy()


def stored_format(directory: Path) -> str | None:
    """The format of the tokens stored in ``directory``: fp16 (``tokens.npy``), a
    packed store's format, or None if there is none."""
    if (directory / "tokens.npy").exists():
        return "fp16"
    if (directory / "tokens.codes.npy").exists():
        return "nvfp4"
    return None


def open_store(directory: Path, fmt: str) -> dict[str, np.ndarray]:
    """Read-only memmaps of a packed store's parts."""
    parts = empty(fmt, (0, 0, BLOCK), torch.device("meta")).parts
    return {k: np.load(directory / f"tokens.{k}.npy", mmap_mode="r") for k in parts}


def from_store(array: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    """A store part's rows as a tensor of ``like``'s dtype (fp8 from its bytes)."""
    t = torch.from_numpy(np.ascontiguousarray(array))
    return t.view(like.dtype) if like.dtype == torch.float8_e4m3fn else t


_NUMPY = {torch.uint8: np.uint8, torch.float32: np.float32, torch.float16: np.float16}
