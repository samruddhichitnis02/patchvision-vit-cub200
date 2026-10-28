"""Plots saved to disk: curves, confusion matrix, reliability diagram, attention overlays, augmentation preview."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def _finish(fig, path: Optional[Path], show: bool):
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)


def _unnormalize(img: torch.Tensor, mean, std) -> np.ndarray:
    m, s = torch.tensor(mean)[:, None, None], torch.tensor(std)[:, None, None]
    return (img.cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


def plot_curves(h: pd.DataFrame, path, title: str = "", show: bool = False):
    """Loss / accuracy / overfitting-gap / lr / attention-in-box / EMA-vs-raw curves."""
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    e = h["epoch"]
    ax[0, 0].plot(e, h["train_loss"], label="train total")
    ax[0, 0].plot(e, h["train_ce"], label="train CE", ls=":")
    ax[0, 0].plot(e, h["val_loss"], label="val CE")
    ax[0, 0].set_title("loss")
    ax[0, 0].legend()
    ax[0, 1].plot(e, h["train_acc1"], label="train (augmented, approx.)", alpha=.6)
    if "clean_acc1" in h:
        c = h.dropna(subset=["clean_acc1"])
        ax[0, 1].plot(c["epoch"], c["clean_acc1"], "o-", ms=3, label="train (clean images)")
    ax[0, 1].plot(e, h["val_acc1"], label="val", lw=2)
    if "valraw_acc1" in h and h["valraw_acc1"].notna().any():
        ax[0, 1].plot(e, h["valraw_acc1"], label="val (raw weights)", ls="--", alpha=.7)
    ax[0, 1].set_title("top-1 accuracy (%)")
    ax[0, 1].legend()
    ax[0, 2].plot(e, h["val_macro_f1"], label="val macro-F1")
    ax[0, 2].plot(e, h["val_acc5"], label="val top-5")
    ax[0, 2].set_title("validation F1 / top-5 (%)")
    ax[0, 2].legend()
    ax[1, 0].plot(e, h["train_lr_first"])
    ax[1, 0].set_title("learning rate")
    if "val_attn_in_box" in h and h["val_attn_in_box"].notna().any():
        ax[1, 1].plot(e, h["val_attn_in_box"], label="attention inside bird box")
        ax[1, 1].plot(e, h["val_box_area_frac"], label="box area (= chance)", ls="--")
        ax[1, 1].set_title("validation attention inside the bird box")
        ax[1, 1].legend()
    if "clean_acc1" in h and h["clean_acc1"].notna().any():
        c = h.dropna(subset=["clean_acc1"])
        ax[1, 2].plot(c["epoch"], c["clean_acc1"] - c["val_acc1"], "o-", ms=3, label="clean train - val")
        ax[1, 2].legend()
    ax[1, 2].axhline(0, c="k", lw=.5)
    ax[1, 2].set_title("overfitting gap (top-1 points)")
    for a in ax.ravel():
        a.set_xlabel("epoch")
        a.grid(alpha=.3)
    fig.suptitle(title)
    _finish(fig, path, show)


def plot_confusion(conf: np.ndarray, path, title: str = "", show: bool = False):
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(np.log1p(conf), cmap="viridis")
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(title + " (log scale)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    _finish(fig, path, show)


def plot_reliability(rows: Sequence[dict], ece: float, path, title: str = "", show: bool = False):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.plot([r["confidence"] for r in rows], [r["accuracy"] for r in rows], "o-", label="model")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(f"{title}\nECE = {ece:.2f}%")
    ax.legend()
    ax.grid(alpha=.3)
    _finish(fig, path, show)


@torch.no_grad()
def attention_overlays(model, dataset, indices: Sequence[int], device, grid: int, mean, std, path,
                       title: str = "", ncols: int = 8, show: bool = False):
    """Per image: (1) the image with the bird box (green) and pred/true label, (2) attention received by each
    patch in the LAST pre-pooling block, (3) the mean over all pre-pooling blocks."""
    was_training = model.training
    model.eval()
    items = [dataset[i] for i in indices]
    x = torch.stack([it[0] for it in items]).to(device)
    logits, aux = model(x)
    probs = torch.softmax(logits.float(), -1)
    conf, pred = probs.max(1)
    maps = aux["attn_maps"]
    last = maps[-1].float().view(-1, 1, grid, grid)
    mean_map = torch.stack([m.float() for m in maps]).mean(0).view(-1, 1, grid, grid)
    size = x.shape[-2:]
    up = lambda t: F.interpolate(t, size=size, mode="bicubic", align_corners=False).clamp_min(0).cpu()
    last_up, mean_up = up(last), up(mean_map)
    names = getattr(dataset, "class_names", None)
    nrows_groups = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(3 * nrows_groups, ncols, figsize=(2.3 * ncols, 2.3 * 3 * nrows_groups), squeeze=False)
    for a in axes.ravel():
        a.axis("off")
    for p, (img, mask, y, _) in enumerate(items):
        r0, c = (p // ncols) * 3, p % ncols
        rgb = _unnormalize(img, mean, std)
        ok = int(pred[p]) == y
        axes[r0, c].imshow(rgb)
        if mask.max() > 0.5:
            axes[r0, c].contour(mask[0].numpy(), levels=[0.5], colors="lime", linewidths=1.5)
        t = (names[y][:14] if names else str(y))
        pn = (names[int(pred[p])][:14] if names else str(int(pred[p])))
        axes[r0, c].set_title(f"{t}\n-> {pn} {conf[p]:.2f}", fontsize=7, color="green" if ok else "red")
        for k, heat in enumerate((last_up[p, 0], mean_up[p, 0]), start=1):
            axes[r0 + k, c].imshow(rgb)
            axes[r0 + k, c].imshow(heat.numpy(), cmap="jet", alpha=0.5)
            if mask.max() > 0.5:
                axes[r0 + k, c].contour(mask[0].numpy(), levels=[0.5], colors="white", linewidths=0.8)
        if c == 0:
            axes[r0 + 1, c].set_title("last block", fontsize=7)
            axes[r0 + 2, c].set_title("mean of blocks", fontsize=7)
    fig.suptitle(title)
    _finish(fig, path, show)
    model.train(was_training)


def augmentation_preview(dataset, mean, std, path, n: int = 12, show: bool = False, title: str = ""):
    """Augmented training images with their bird box (green): the box must stay on the bird."""
    ncols = 6
    fig, axes = plt.subplots(int(np.ceil(n / ncols)), ncols, figsize=(3 * ncols, 3.2 * np.ceil(n / ncols)), squeeze=False)
    for a in axes.ravel():
        a.axis("off")
    for ax, i in zip(axes.ravel(), range(n)):
        img, m, y, _ = dataset[i]
        ax.imshow(_unnormalize(img, mean, std))
        if m.max() > 0.5:
            ax.contour(m[0].numpy(), levels=[0.5], colors="lime", linewidths=2)
        ax.set_title(dataset.class_names[y][:18], fontsize=8)
    fig.suptitle(title)
    _finish(fig, path, show)
