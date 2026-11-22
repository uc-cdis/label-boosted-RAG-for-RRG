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

``classes`` > 1 (multiclass): each label and section gets one logit per class
(MULTICLASS: CheXbert's positive, negative, uncertain, blank), softmax cross-entropy
over the classes instead of BCE; the epoch is still picked on the positive class's
AUROC (one vs rest).

``condition`` (per row, an index < ``conditions``): the head's vector for it, e.g. the
report's reader (attention.Head).

Several images per row (``paths`` items that are sequences of paths, StudyImages): the
backbone encodes each image alone and the head attends over all of a study's tokens
(padding masked out). The backbone then recomputes its activations in the backward
pass (gradient checkpointing): a batch holds a variable number of images.
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
MULTICLASS = ("positive", "negative", "uncertain", "unmentioned")
MULTICLASS_CODES = (1.0, 0.0, -1.0, None)  # CheXbert's value per class; None = blank
PREDICT_BATCH = 64
UNUSED = {"embeddings.mask_token"}  # masked-image pretraining only; stays frozen
RESULT = ".finetune-result.pt"


class Model(nn.Module):
    """Backbone tokens (CLS + patches, or CLS only) into the attention head; with
    ``classes`` > 1 the head's readouts are section-major, then class. With ``counts``
    (images per study, the pixels' rows in study order) the head reads each study's
    tokens from all its images."""

    def __init__(
        self,
        backbone: nn.Module,
        head: attention.Head,
        cls_only: bool,
        classes: int = 1,
    ):
        super().__init__()
        self.backbone, self.head, self.cls_only = backbone, head, cls_only
        self.classes = classes

    def forward(
        self,
        pixels: torch.Tensor,
        condition: torch.Tensor | None = None,
        counts: list[int] | None = None,
    ) -> torch.Tensor:
        hidden = self.backbone(pixel_values=pixels).last_hidden_state
        if self.cls_only:
            hidden = hidden[:, :1]
        if counts is None:
            return self.head(hidden, condition)
        tokens, mask = by_study(hidden, counts)
        return self.head(tokens, condition, mask)


def by_study(
    hidden: torch.Tensor, counts: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Images' tokens (images, T, D), consecutive per study, as (studies, max tokens,
    D) zero-padded, and the mask of the real tokens."""
    per = [h.flatten(0, 1) for h in hidden.split(counts)]
    tokens = nn.utils.rnn.pad_sequence(per, batch_first=True)
    length = torch.tensor([len(p) for p in per], device=hidden.device)
    mask = torch.arange(tokens.shape[1], device=hidden.device) < length[:, None]
    return tokens, mask


def build(
    backbone: nn.Module,
    n_labels: int,
    readouts: int,
    cls_only: bool,
    classes: int = 1,
    conditions: int = 0,
) -> Model:
    h = PARAMS["head"]
    dim = backbone.config.hidden_size
    head = attention.Head(
        n_labels,
        dim,
        h["heads"],
        h["ffn"],
        h["dropout"],
        readouts * classes,
        conditions,
    )
    return Model(backbone, head, cls_only, classes)


def class_index(codes: np.ndarray) -> np.ndarray:
    """CheXbert values (1, 0, -1, NaN for blank) as MULTICLASS indices."""
    codes = np.asarray(codes, dtype=float)
    index = np.full(codes.shape, MULTICLASS.index("unmentioned"))
    for i, code in enumerate(MULTICLASS_CODES):
        if code is not None:
            index[codes == code] = i
    return index


def positive_probs(P: np.ndarray, readouts: int, classes: int) -> np.ndarray:
    """(N, readouts * labels) probabilities of the positive class, from the
    (N, readouts * classes * labels) outputs of a multiclass model (as is for
    classes = 1)."""
    if classes == 1:
        return P
    n = len(P)
    return P.reshape(n, readouts, classes, -1)[:, :, 0].reshape(n, -1)


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


class StudyImages(Images):
    """Item i: study i's images' pixel values (images, 3, SIZE, SIZE), each augmented
    when ``seed`` is set (drawn from (seed, epoch, i, image))."""

    def __getitem__(self, i: int) -> torch.Tensor:
        out = []
        for k, path in enumerate(self.paths[i]):
            image = load_8bit(path)
            if self.seed is not None:
                rng = np.random.default_rng([self.seed, self.epoch, i, k])
                image = augment(image, rng)
            out.append(self.processor(image, return_tensors="pt")["pixel_values"][0])
        return torch.stack(out)


def collate_studies(items: list[torch.Tensor]) -> tuple[torch.Tensor, list[int]]:
    """StudyImages items as one stack of images and each study's count."""
    return torch.cat(items), [len(x) for x in items]


def multi_image(paths: Sequence) -> bool:
    """True if each row has a sequence of paths (StudyImages), not one path."""
    return len(paths) > 0 and not isinstance(paths[0], (str, os.PathLike))


def image_data(paths: Sequence, processor, seed: int | None = None) -> Images:
    """Images, or StudyImages for rows of several paths."""
    return (StudyImages if multi_image(paths) else Images)(paths, processor, seed)


def loader(
    images: Images, batch_size: int, workers: int, rows=None, pin_memory=False
) -> DataLoader:
    """Batches of pixels (of ``rows``, if given, in order), or (pixels, counts) for
    StudyImages."""
    collate = collate_studies if isinstance(images, StudyImages) else None
    return DataLoader(
        images if rows is None else Subset(images, rows),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=pin_memory,
        collate_fn=collate,
    )


def unpack(batch) -> tuple[torch.Tensor, list[int] | None]:
    """A loader batch as (pixels, counts or None)."""
    return batch if isinstance(batch, tuple) else (batch, None)


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
    classes: int = 1,
    condition: np.ndarray | None = None,
    conditions: int = 0,
) -> tuple[dict, np.ndarray, list[dict]]:
    """Fine-tune on ``train_rows``, keep the epoch with the best ``select_metric`` on
    ``val_rows`` (boolean masks over the rows of paths, Y and mask; ``group`` = each
    row's dataset). One process per GPU (``world``: default all visible GPUs; one
    process on CPU). Returns the kept state dict (CPU), its val probabilities (the
    val_rows in order; multiclass: every class, as predict_proba) and one history row
    per epoch. Multiclass (``classes`` > 1): Y holds MULTICLASS indices.
    ``condition``: each row's index into the head's ``conditions`` vectors.
    ``backbone``/``processor`` replace RAD-DINO's and ``params`` overrides PARAMS
    entries (tests)."""
    world = world or max(1, torch.cuda.device_count())
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(_free_port())
    args = (world, model_dir, list(paths), Y, mask, train_rows, val_rows, group)
    args += (readouts, seed, workdir, cls_only, workers, backbone, processor)
    args += ({**PARAMS, **(params or {})}, classes, condition, conditions)
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
    classes,
    condition,
    conditions,
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
            classes,
            condition,
            conditions,
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
    classes,
    condition,
    conditions,
):
    cuda = device.type == "cuda"
    torch.manual_seed(seed)  # the head's init, the same on every rank
    # a passed-in backbone reaches spawned ranks in shared memory: each needs its own
    backbone = (
        copy.deepcopy(backbone) if backbone is not None else load_backbone(model_dir)
    )
    processor = processor if processor is not None else load_processor(model_dir)
    n_labels = Y.shape[1] // readouts
    model = build(backbone, n_labels, readouts, cls_only, classes, conditions)
    model = model.to(device)
    if multi_image(paths):  # a batch's image count varies; bound its activations
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    cond = None if condition is None else torch.as_tensor(condition, dtype=torch.long)
    if classes == 1:
        targets = torch.as_tensor(Y, dtype=torch.float32)
        Y_auroc = Y
    else:  # one-hot (rows, readouts, labels, classes): elementwise, deterministic
        index = torch.as_tensor(Y, dtype=torch.long).view(len(Y), readouts, n_labels)
        targets = nn.functional.one_hot(index, classes).float()
        Y_auroc = (Y == MULTICLASS.index("positive")).astype(int)
    weights = torch.as_tensor(mask, dtype=torch.float32)
    train_idx, val_idx = np.flatnonzero(train_rows), np.flatnonzero(val_rows)
    batch = PARAMS["batch_size"]
    per_rank = batch // world
    assert per_rank * world == batch, "batch_size must divide by the number of GPUs"
    steps = len(train_idx) // batch  # the last partial batch is dropped
    assert steps > 0, "fewer train rows than one batch"
    data = image_data([paths[i] for i in train_idx], processor, seed)
    shards = np.array_split(val_idx, world)
    val_data = image_data([paths[i] for i in shards[rank]], processor)
    val_cond = None if cond is None else cond[shards[rank]]
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
            batches = loader(data, per_rank, workers, mine.tolist(), cuda)
            loss_sum = torch.zeros((), dtype=torch.float64, device=device)
            for b, items in enumerate(batches):
                pixels, counts = unpack(items)
                rows = train_idx[mine[b * per_rank : (b + 1) * per_rank]]
                y, w = targets[rows].to(device), weights[rows].to(device)
                denom = w.sum()
                dist.all_reduce(denom)  # labelled entries of the global batch
                c = None if cond is None else cond[rows].to(device)
                with autocast:
                    logits = ddp(pixels.to(device), c, counts)
                if classes == 1:
                    each = nn.functional.binary_cross_entropy_with_logits(
                        logits.float(), y, reduction="none"
                    )
                else:
                    z = logits.float().view(len(y), readouts, classes, n_labels)
                    logp = z.log_softmax(2).transpose(2, 3)
                    each = -(logp * y).sum(-1).flatten(1)
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
            P = predict_proba(model, val_data, device, workers, val_cond)
            P = _gather(P, shards, device)
            row = {
                "epoch": epoch,
                "phase": phase,
                "train_loss": loss_sum.item() / steps,
            }
            if rank == 0:
                Pp = positive_probs(P, readouts, classes)
                metric, by = select_metric(
                    Y_auroc[val_idx], Pp, mask[val_idx], group[val_idx]
                )
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
    model: Model,
    images: Images,
    device: torch.device,
    workers: int = 0,
    condition: torch.Tensor | np.ndarray | None = None,
) -> np.ndarray:
    """Probabilities (len(images), readouts * labels), float64, in order (one row per
    item: an image, or a study of StudyImages), under the training's bfloat16 autocast. Multiclass: (len(images), readouts * classes *
    labels), section-major then class, softmax over the classes. ``condition``: each
    image's index into the head's vectors."""
    model.eval()
    out = [np.zeros((0, model.head.bias.shape[0]))]
    c = model.classes
    if condition is not None:
        condition = torch.as_tensor(condition, dtype=torch.long)
    start = 0
    for items in loader(images, PREDICT_BATCH, workers):
        pixels, counts = unpack(items)
        n = len(pixels) if counts is None else len(counts)
        cond = None
        if condition is not None:
            cond = condition[start : start + n].to(device)
        start += n
        with torch.autocast(device.type, dtype=torch.bfloat16):
            logits = model(pixels.to(device), cond, counts).float()
        if c == 1:
            p = torch.sigmoid(logits)
        else:
            b = len(logits)
            p = logits.view(b, model.head.readouts // c, c, -1).softmax(2).flatten(1)
        out.append(p.double().cpu().numpy())
    return np.concatenate(out)
