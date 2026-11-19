"""The CUB-200 test set for accuracy checks of every optimized model variant.

Pre-processing is done ONCE and cached: each test photo is decoded and resized to 224x224 exactly as in training
(`patchvision.data.build_transforms(..., train=False)` = Resize(224, antialias) -> scale to [0,1] -> Normalize).
Resizing happens on uint8, so the uint8 cache is bit-identical to what the original pipeline feeds the model;
the float conversion + normalization is applied on the GPU by `normalize()`.

This keeps JPEG decoding / data-loading out of the accuracy loop, so evaluating a new variant takes seconds.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms.v2 as T

from .models import REPO
from patchvision.data import CUBDataset  # noqa: E402  (sys.path set up by .models)

PREP = REPO / "DataPreprocessing_Results"
CUB_ROOT = REPO / "CUB_200_2011"
CACHE = Path(__file__).resolve().parents[1] / "cache"


def norm_stats() -> Tuple[list, list]:
    s = json.loads((PREP / "norm_stats.json").read_text())
    return s["mean"], s["std"]


def load_test_set(split: str = "test", img_size: int = 224) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (images uint8 [N,3,H,W], labels int64 [N]) on the CPU, building the cache on first use."""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{split}_{img_size}_uint8.pt"
    if f.exists():
        d = torch.load(f, weights_only=True)
        return d["x"], d["y"]
    if not (CUB_ROOT / "images").is_dir():
        raise FileNotFoundError(f"CUB images not found under {CUB_ROOT}. Extract CUB_200_2011.tgz into the repo root.")
    resize = T.Compose([T.Resize((img_size, img_size), antialias=True)])
    ds = CUBDataset(PREP / f"{split}.csv", lambda img, m: (resize(img), m), bboxes=None, cub_root=CUB_ROOT)
    xs = torch.empty(len(ds), 3, img_size, img_size, dtype=torch.uint8)
    ys = torch.empty(len(ds), dtype=torch.int64)
    for i in range(len(ds)):
        img, _, y, _ = ds[i]
        xs[i], ys[i] = torch.as_tensor(img), y
    torch.save({"x": xs, "y": ys}, f)
    return xs, ys


class Normalizer:
    """uint8 [B,3,H,W] (any device) -> normalized float32, identical to ToDtype(scale=True) + Normalize."""

    def __init__(self, device="cuda"):
        mean, std = norm_stats()
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)

    def __call__(self, x_u8: torch.Tensor) -> torch.Tensor:
        return (x_u8.float() / 255.0 - self.mean) / self.std


@torch.inference_mode()
def evaluate_logits(fn, x_u8: torch.Tensor, device="cuda", batch_size: int = 128) -> torch.Tensor:
    """Runs `fn(normalized_batch) -> logits` over the whole set; returns float32 logits [N, C] on the CPU."""
    norm = Normalizer(device)
    out = []
    for i in range(0, len(x_u8), batch_size):
        xb = norm(x_u8[i:i + batch_size].to(device, non_blocking=True))
        out.append(fn(xb).float().cpu())
    return torch.cat(out)


def accuracy_metrics(logits: torch.Tensor, y: torch.Tensor) -> dict:
    """top-1, top-5, macro-F1 (in %) - same definitions as the training code's MetricTracker."""
    pred = logits.argmax(1)
    top5 = logits.topk(5, dim=1).indices
    n_cls = logits.shape[1]
    acc1 = (pred == y).float().mean().item() * 100
    acc5 = (top5 == y[:, None]).any(1).float().mean().item() * 100
    cm = torch.bincount(y * n_cls + pred, minlength=n_cls * n_cls).view(n_cls, n_cls).double()
    tp = cm.diag()
    prec = tp / cm.sum(0).clamp_min(1)
    rec = tp / cm.sum(1).clamp_min(1)
    f1 = torch.where(prec + rec > 0, 2 * prec * rec / (prec + rec), torch.zeros_like(tp))
    return {"top1": acc1, "top5": acc5, "macro_f1": f1.mean().item() * 100}


def prediction_agreement(logits_a: torch.Tensor, logits_b: torch.Tensor) -> dict:
    """How closely an optimized variant reproduces the reference model, photo by photo."""
    pa, pb = logits_a.argmax(1), logits_b.argmax(1)
    pa_s, pb_s = logits_a.float().softmax(1), logits_b.float().softmax(1)
    return {"top1_agreement": (pa == pb).float().mean().item() * 100,
            "max_abs_logit_diff": (logits_a.float() - logits_b.float()).abs().max().item(),
            "max_abs_prob_diff": (pa_s - pb_s).abs().max().item()}
