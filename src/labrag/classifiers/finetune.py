"""RAD-DINO fine-tuned end to end with the attention head on its tokens (``Model``).

Two phases (LP-FT, Kumar et al. 2022): the head alone on the frozen backbone (linear
probe phase), then everything, the backbone at a lower learning rate that decays by
layer (early layers change least), so the random head's first gradients don't distort
the pretrained features. The head and its loss are attention.Head's (one query per
label, a readout per section, BCE over each study's labelled entries).

Images: the embed preprocessing (encoders/raddino.py: min-max 8-bit, short side to
518, center crop, normalize); training adds a random crop and a small rotation, drawn
per (seed, epoch, row) so they don't depend on the number of workers or GPUs. No flips:
left and right matter.

One node, one process per GPU (DDP). Deterministic for a given seed and number of GPUs:
deterministic kernels, eager attention, seeded head init, batch order and augmentation.
The epoch (of either phase) with the best ``select_metric`` on val is kept.
"""

import copy
import math
import os
import socket
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from labrag.classifiers import attention
from labrag.encoders.images import load_8bit, resize_short_edge

SIZE = 518
PARAMS = {
    "backbone": "RAD-DINO ViT-B/14 (Dinov2Model), eager attention",
    "head": {k: attention.PARAMS[k] for k in ("heads", "ffn", "dropout")},
    "lp_epochs": 2,
    "lp_lr": 1e-4,
    "ft_epochs": 10,
    "head_lr": 1e-4,
    "backbone_lr": 2e-5,
    "layer_decay": 0.75,
    "warmup": 0.05,
    "schedule": "fine-tuning: linear warmup (fraction of its steps), cosine to 0",
    "optimizer": "AdamW; backbone biases, norms and embeddings without weight decay",
    "weight_decay": 0.05,
    "batch_size": 128,
    "precision": "bfloat16 autocast",
    "crop_scale": [0.5, 1.0],
    "crop_ratio": [0.75, 4 / 3],
    "rotate_degrees": 10,
    "loss": "BCE over each study's labelled entries",
    "select": "epoch with the best mean over datasets of their mean val AUROC",
}
PREDICT_BATCH = 64
UNUSED = {"embeddings.mask_token"}  # masked-image pretraining only; stays frozen
RESULT = ".finetune-result.pt"


class Model(nn.Module):
    """Backbone tokens (CLS + patches, or CLS only) into the attention head."""

    def __init__(self, backbone: nn.Module, head: attention.Head, cls_only: bool):
        super().__init__()
        self.backbone, self.head, self.cls_only = backbone, head, cls_only

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(pixel_values=pixels).last_hidden_state
        return self.head(hidden[:, :1] if self.cls_only else hidden)


def build(backbone: nn.Module, n_labels: int, readouts: int, cls_only: bool) -> Model:
    h = PARAMS["head"]
    dim = backbone.config.hidden_size
    head = attention.Head(n_labels, dim, h["heads"], h["ffn"], h["dropout"], readouts)
    return Model(backbone, head, cls_only)


def load_backbone(model_dir: Path) -> nn.Module:
    from transformers import AutoModel

    return AutoModel.from_pretrained(str(model_dir), attn_implementation="eager")


def load_processor(model_dir: Path):
    from transformers import AutoImageProcessor

    return AutoImageProcessor.from_pretrained(str(model_dir), backend="pil")


def augment(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Short side to SIZE, a random crop (area fraction and aspect drawn as
    torchvision's RandomResizedCrop does) resized to SIZE x SIZE, then a random
    rotation; bicubic throughout."""
    image = resize_short_edge(image, SIZE)
    w, h = image.size
    lo, hi = np.log(PARAMS["crop_ratio"])
    box = (0, 0, w, h)
    for _ in range(10):
        area = w * h * rng.uniform(*PARAMS["crop_scale"])
        ratio = math.exp(rng.uniform(lo, hi))
        cw, ch = round(math.sqrt(area * ratio)), round(math.sqrt(area / ratio))
        if 0 < cw <= w and 0 < ch <= h:
            x, y = int(rng.integers(0, w - cw + 1)), int(rng.integers(0, h - ch + 1))
            box = (x, y, x + cw, y + ch)
            break
    image = image.resize((SIZE, SIZE), Image.Resampling.BICUBIC, box=box)
    degrees = PARAMS["rotate_degrees"]
    return image.rotate(rng.uniform(-degrees, degrees), Image.Resampling.BICUBIC)


class Images(Dataset):
    """Item i: image i's pixel values; augmented when ``seed`` is set (drawn from
    (seed, epoch, i)). Eval items are exactly the embed preprocessing."""

    def __init__(self, paths: Sequence[Path], processor, seed: int | None = None):
        self.paths, self.processor = list(paths), processor
        self.seed, self.epoch = seed, 0

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        image = load_8bit(self.paths[i])
        if self.seed is not None:
            image = augment(image, np.random.default_rng([self.seed, self.epoch, i]))
        return self.processor(image, return_tensors="pt")["pixel_values"][0]


def param_groups(model: Model) -> list[dict]:
    """AdamW groups for fine-tuning: the head at head_lr; backbone embeddings (depth
    0), layer i (depth i + 1) and final norm (depth L + 1) at backbone_lr *
    layer_decay ** (L + 1 - depth); no weight decay on the backbone's 1-D parameters
    and embeddings."""
    n_layers = model.backbone.config.num_hidden_layers
    groups = {}
    for name, p in model.backbone.named_parameters():
        if name in UNUSED:
            continue
        if name.startswith("embeddings."):
            depth = 0
        elif name.startswith("encoder.layer."):
            depth = int(name.split(".")[2]) + 1
        else:
            depth = n_layers + 1
        no_decay = p.ndim < 2 or depth == 0
        group = groups.setdefault(
            (depth, no_decay),
            {
                "params": [],
                "lr": PARAMS["backbone_lr"]
                * PARAMS["layer_decay"] ** (n_layers + 1 - depth),
                "weight_decay": 0.0 if no_decay else PARAMS["weight_decay"],
            },
        )
        group["params"].append(p)
    head = {
        "params": list(model.head.parameters()),
        "lr": PARAMS["head_lr"],
        "weight_decay": PARAMS["weight_decay"],
    }
    return [groups[k] for k in sorted(groups)] + [head]


def lr_factor(step: int, total: int) -> float:
    """Fine-tuning schedule: linear warmup over PARAMS["warmup"] of the steps, then
    cosine to 0."""
    warm = max(1, round(PARAMS["warmup"] * total))
    if step < warm:
        return (step + 1) / warm
    return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def select_metric(
    Y: np.ndarray, P: np.ndarray, mask: np.ndarray, group: np.ndarray
) -> tuple[float, dict]:
    """Mean over groups (datasets) of each group's mean AUROC over its labelled
    entries (every readout); also each group's."""
    by = {
        str(g): attention.mean_auroc(Y[group == g], P[group == g], mask[group == g])
        for g in np.unique(group)
    }
    return float(np.mean(list(by.values()))), by


def train(
    model_dir: Path | None,
    paths: Sequence[Path],
    Y: np.ndarray,
    mask: np.ndarray,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    group: np.ndarray,
    readouts: int,
    seed: int,
    workdir: Path,
    cls_only: bool = False,
    world: int | None = None,
    workers: int = 0,
    backbone: nn.Module | None = None,
    processor=None,
    params: dict | None = None,
) -> tuple[dict, np.ndarray, list[dict]]:
    """Fine-tune on ``train_rows``, keep the epoch with the best ``select_metric`` on
    ``val_rows`` (boolean masks over the rows of paths, Y and mask; ``group`` = each
    row's dataset). One process per GPU (``world``: default all visible GPUs; one
    process on CPU). Returns the kept state dict (CPU), its val probabilities (the
    val_rows in order) and one history row per epoch.
    ``backbone``/``processor`` replace RAD-DINO's and ``params`` overrides PARAMS
    entries (tests)."""
    world = world or max(1, torch.cuda.device_count())
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(_free_port())
    args = (world, model_dir, list(paths), Y, mask, train_rows, val_rows, group)
    args += (readouts, seed, workdir, cls_only, workers, backbone, processor)
    args += ({**PARAMS, **(params or {})},)
    if world == 1:
        _worker(0, *args)
    else:
        torch.multiprocessing.spawn(_worker, args=args, nprocs=world, join=True)
    saved = torch.load(workdir / RESULT, weights_only=False)
    (workdir / RESULT).unlink()
    return saved["state_dict"], saved["val_probs"], saved["history"]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(
    rank,
    world,
    model_dir,
    paths,
    Y,
    mask,
    train_rows,
    val_rows,
    group,
    readouts,
    seed,
    workdir,
    cls_only,
    workers,
    backbone,
    processor,
    params,
):
    defaults = dict(PARAMS)
    PARAMS.update(params)  # spawned workers start from the module's PARAMS
    attention.deterministic()
    cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{rank}" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if cuda else "gloo", rank=rank, world_size=world)
    try:
        _fit(
            rank,
            world,
            device,
            model_dir,
            paths,
            Y,
            mask,
            train_rows,
            val_rows,
            group,
            readouts,
            seed,
            workdir,
            cls_only,
            workers,
            backbone,
            processor,
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()
        PARAMS.clear()
        PARAMS.update(defaults)


def _fit(
    rank,
    world,
    device,
    model_dir,
    paths,
    Y,
    mask,
    train_rows,
    val_rows,
    group,
    readouts,
    seed,
    workdir,
    cls_only,
    workers,
    backbone,
    processor,
):
    cuda = device.type == "cuda"
    torch.manual_seed(seed)  # the head's init, the same on every rank
    # a passed-in backbone reaches spawned ranks in shared memory: each needs its own
    backbone = (
        copy.deepcopy(backbone) if backbone is not None else load_backbone(model_dir)
    )
    processor = processor if processor is not None else load_processor(model_dir)
    model = build(backbone, Y.shape[1] // readouts, readouts, cls_only).to(device)
    targets = torch.as_tensor(Y, dtype=torch.float32)
    weights = torch.as_tensor(mask, dtype=torch.float32)
    train_idx, val_idx = np.flatnonzero(train_rows), np.flatnonzero(val_rows)
    batch = PARAMS["batch_size"]
    per_rank = batch // world
    assert per_rank * world == batch, "batch_size must divide by the number of GPUs"
    steps = len(train_idx) // batch  # the last partial batch is dropped
    assert steps > 0, "fewer train rows than one batch"
    data = Images([paths[i] for i in train_idx], processor, seed)
    shards = np.array_split(val_idx, world)
    val_data = Images([paths[i] for i in shards[rank]], processor)
    autocast = torch.autocast(device.type, dtype=torch.bfloat16)

    history, best, best_state, best_P, epoch = [], -np.inf, None, None, 0
    for phase, n_epochs in (("lp", PARAMS["lp_epochs"]), ("ft", PARAMS["ft_epochs"])):
        lp = phase == "lp"
        for name, p in model.backbone.named_parameters():
            p.requires_grad_(not lp and name not in UNUSED)
        ddp = nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if cuda else None
        )
        if lp:
            opt = torch.optim.AdamW(
                model.head.parameters(),
                lr=PARAMS["lp_lr"],
                weight_decay=PARAMS["weight_decay"],
            )
        else:
            opt = torch.optim.AdamW(param_groups(model))
            base = [g["lr"] for g in opt.param_groups]
            step, total = 0, n_epochs * steps
        for _ in range(n_epochs):
            epoch += 1
            model.train()
            data.epoch = epoch
            order = torch.Generator().manual_seed(seed * 1000 + epoch)
            perm = torch.randperm(len(train_idx), generator=order)[: steps * batch]
            mine = perm.view(steps, world, per_rank)[:, rank].reshape(-1).numpy()
            loader = DataLoader(
                Subset(data, mine.tolist()),
                batch_size=per_rank,
                num_workers=workers,
                pin_memory=cuda,
            )
            loss_sum = torch.zeros((), dtype=torch.float64, device=device)
            for b, pixels in enumerate(loader):
                rows = train_idx[mine[b * per_rank : (b + 1) * per_rank]]
                y, w = targets[rows].to(device), weights[rows].to(device)
                denom = w.sum()
                dist.all_reduce(denom)  # labelled entries of the global batch
                with autocast:
                    logits = ddp(pixels.to(device))
                each = nn.functional.binary_cross_entropy_with_logits(
                    logits.float(), y, reduction="none"
                )
                # the global batch's mean; DDP averages gradients over the ranks
                loss = (each * w).sum() / denom * world
                if not lp:
                    for g, lr in zip(opt.param_groups, base):
                        g["lr"] = lr * lr_factor(step, total)
                    step += 1
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                loss_sum += loss.detach().double() / world
            dist.all_reduce(loss_sum)
            P = _gather(predict_proba(model, val_data, device, workers), shards, device)
            row = {
                "epoch": epoch,
                "phase": phase,
                "train_loss": loss_sum.item() / steps,
            }
            if rank == 0:
                metric, by = select_metric(Y[val_idx], P, mask[val_idx], group[val_idx])
                row |= {"val_auroc": metric} | {
                    f"val_auroc.{g}": v for g, v in by.items()
                }
                history.append(row)
                print(
                    f"epoch {epoch} ({phase}): train loss {row['train_loss']:.4f}, "
                    f"val AUROC {metric:.4f} {by}",
                    flush=True,
                )
                if metric > best:
                    best, best_P = metric, P
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }
        del ddp
    if rank == 0:
        result = {"state_dict": best_state, "val_probs": best_P, "history": history}
        torch.save(result, workdir / RESULT)


def _gather(P: np.ndarray, shards: list, device) -> np.ndarray | None:
    """Every rank's shard of predictions, in row order, on rank 0 (None elsewhere)."""
    world = len(shards)
    if world == 1:
        return P
    width = P.shape[1]
    longest = max(len(s) for s in shards)
    padded = torch.zeros((longest, width), dtype=torch.float64, device=device)
    padded[: len(P)] = torch.as_tensor(P, device=device)
    gathered = [torch.zeros_like(padded) for _ in range(world)]
    dist.all_gather(gathered, padded)
    if dist.get_rank() != 0:
        return None
    return torch.cat([g[: len(s)].cpu() for g, s in zip(gathered, shards)]).numpy()


@torch.no_grad()
def predict_proba(
    model: Model, images: Images, device: torch.device, workers: int = 0
) -> np.ndarray:
    """Probabilities (len(images), readouts * labels), float64, in order, under the
    training's bfloat16 autocast."""
    model.eval()
    loader = DataLoader(images, batch_size=PREDICT_BATCH, num_workers=workers)
    out = [np.zeros((0, model.head.bias.shape[0]))]
    for pixels in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16):
            logits = model(pixels.to(device))
        out.append(torch.sigmoid(logits.float()).double().cpu().numpy())
    return np.concatenate(out)
