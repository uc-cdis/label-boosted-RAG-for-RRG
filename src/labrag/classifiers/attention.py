"""Per-label attention pooling over frozen patch tokens (one cross-attention layer).

One learned query per label cross-attends over a study's tokens (CLS + patches), then a
residual feed-forward block and one logit per label: each label pools its own image
regions. The queries don't attend to each other (unlike Query2Label's decoder); labels
share only the projection and feed-forward weights. Given only the CLS token, the
attention passes it through and the head is a jointly trained MLP on CLS. Trained jointly on all labels with BCE, AdamW, early stopping on mean val
AUROC.

``readouts`` > 1 gives each label several logits (one per report section) from the same
query: the sections share where to look, and only the last linear layer learns what
each section reports. Logits are readout-major; a mask leaves out each study's
sections without labels, in the loss and in the val AUROC. Deterministic for a given seed and device: deterministic torch algorithms, the
attention written out (no fused kernels), a seeded generator for the batch order.
"""

import copy
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from labrag.classifiers.quant import Packed

PARAMS = {
    "heads": 8,
    "ffn": 3072,
    "dropout": 0.1,
    "optimizer": "AdamW",
    "lr": 1e-4,
    "weight_decay": 0.05,
    "batch_size": 256,
    "max_epochs": 30,
    "patience": 5,
    "loss": "BCE, unweighted",
    "select": "epoch with the best mean val AUROC",
}
PREDICT_BATCH = 512


class Head(nn.Module):
    def __init__(
        self,
        n_labels: int,
        dim: int,
        heads: int,
        ffn: int,
        dropout: float,
        readouts: int = 1,
    ):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.readouts = heads, readouts
        self.query = nn.Parameter(torch.randn(n_labels, dim) * 0.02)
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim) for _ in range(4))
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn, dim)
        )
        self.drop = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.zeros(readouts * n_labels, dim))
        self.bias = nn.Parameter(torch.zeros(readouts * n_labels))

    def attention(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pooled features (B, labels, dim) and weights (B, heads, labels, tokens)."""
        b, t, d = x.shape
        h = self.heads
        q = self.q(self.query).view(-1, h, d // h)
        k = self.k(x).view(b, t, h, d // h)
        v = self.v(x).view(b, t, h, d // h)
        weights = (torch.einsum("lhe,bthe->bhlt", q, k) / (d // h) ** 0.5).softmax(-1)
        pooled = torch.einsum("bhlt,bthe->blhe", self.drop(weights), v).flatten(-2)
        return self.o(pooled), weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits (B, readouts * labels) for tokens x (B, tokens, dim)."""
        pooled, _ = self.attention(x)
        z = self.norm1(self.query + self.drop(pooled))
        z = self.norm2(z + self.drop(self.ffn(z)))
        if self.readouts == 1:  # the single-readout heads' exact arithmetic
            return (z * self.weight).sum(-1) + self.bias
        r = self.readouts
        logits = (z.unsqueeze(1) * self.weight.view(r, *z.shape[1:])).sum(-1)
        return (logits + self.bias.view(r, -1)).flatten(1)


def deterministic() -> None:
    """Deterministic torch kernels; call before the first CUDA call."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def predict_proba(model: Head, tokens: Packed, rows: torch.Tensor) -> np.ndarray:
    """Probabilities (len(rows), labels), float64."""
    model.eval()
    out = [
        torch.sigmoid(model(tokens.take(batch))).double().cpu()
        for batch in rows.split(PREDICT_BATCH)
    ]
    return torch.cat(out).numpy()


def mean_auroc(Y: np.ndarray, P: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean AUROC over the labels with both classes present (each label on the rows
    where ``mask`` is true, if given)."""
    if mask is None:
        mask = np.ones(Y.shape, dtype=bool)
    scores = [
        roc_auc_score(y[m], p[m])
        for y, p, m in zip(Y.T, P.T, mask.T)
        if 0 < y[m].sum() < m.sum()
    ]
    assert scores, "no label has both classes"
    return float(np.mean(scores))


def fit(
    tokens: Packed,
    Y: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
    seed: int,
    log=print,
    mask: np.ndarray | None = None,
    readouts: int = 1,
) -> tuple[Head, list[dict]]:
    """Train on rows ``train`` of (tokens, Y); pick the epoch by mean AUROC on ``val``.
    Y has ``readouts`` blocks of label columns; ``mask`` (Y's shape) marks the entries
    with a label (all, if not given). Returns the selected model and one history row
    per epoch."""
    device = next(iter(tokens.parts.values())).device
    torch.manual_seed(seed)
    order = torch.Generator().manual_seed(seed)
    model = Head(
        Y.shape[1] // readouts,
        tokens.dim,
        PARAMS["heads"],
        PARAMS["ffn"],
        PARAMS["dropout"],
        readouts,
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=PARAMS["lr"], weight_decay=PARAMS["weight_decay"]
    )
    targets = torch.as_tensor(Y, dtype=torch.float32, device=device)
    weights = None
    if mask is not None:
        weights = torch.as_tensor(mask, dtype=torch.float32, device=device)
    train_rows = torch.as_tensor(np.flatnonzero(train), device=device)
    val_rows = torch.as_tensor(np.flatnonzero(val), device=device)
    loss_fn = nn.BCEWithLogitsLoss()
    best, best_state, history, stale = -np.inf, None, [], 0
    for epoch in range(1, PARAMS["max_epochs"] + 1):
        model.train()
        perm = train_rows[torch.randperm(len(train_rows), generator=order).to(device)]
        total = 0.0
        for batch in perm.split(PARAMS["batch_size"]):
            logits = model(tokens.take(batch))
            if weights is None:
                loss = loss_fn(logits, targets[batch])
            else:  # mean over the labelled entries only
                w = weights[batch]
                each = nn.functional.binary_cross_entropy_with_logits(
                    logits, targets[batch], reduction="none"
                )
                loss = (each * w).sum() / w.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        P = predict_proba(model, tokens, val_rows)
        auroc = mean_auroc(Y[val], P, None if mask is None else mask[val])
        history.append(
            {"epoch": epoch, "train_loss": total / len(perm), "val_auroc": auroc}
        )
        log(f"epoch {epoch}: train loss {total / len(perm):.4f}, val AUROC {auroc:.4f}")
        if auroc > best:
            best, best_state, stale = auroc, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= PARAMS["patience"]:
                break
    model.load_state_dict(best_state)
    return model, history
