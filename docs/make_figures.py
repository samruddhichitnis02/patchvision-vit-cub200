"""
Builds every figure (docs/figures/*.png) and animation (docs/animations/*.gif) used by README.md.

All numbers and pictures come from files that already exist in the repo (results/full, results/base,
DataPreprocessing_Results, CUB_200_2011) or from the real trained models (results/*/seed42/best_model.pt).
Nothing is drawn by hand except the architecture sketch.

    python docs/make_figures.py            # everything
    python docs/make_figures.py arch lr    # only the named parts (see PARTS at the bottom)
"""
from __future__ import annotations

import json
import random
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from patchvision.config import PathsConfig, TrainConfig  # noqa: E402
from patchvision.data import CUBDataset, build_transforms, load_data_info  # noqa: E402
from patchvision.experiment import load_for_inference  # noqa: E402
from patchvision.losses import flip_tokens, sym_kl_probs  # noqa: E402
from patchvision.models import grid_scale_masks  # noqa: E402
from patchvision.optim import build_scheduler  # noqa: E402

FIG, ANI = ROOT / "docs" / "figures", ROOT / "docs" / "animations"
FIG.mkdir(parents=True, exist_ok=True)
ANI.mkdir(parents=True, exist_ok=True)

from matplotlib.colors import ListedColormap  # noqa: E402

NB_CMAP = ListedColormap(["#e4e4e4", "#2b7bba"])   # grey = token cannot be seen, blue = token may be looked at
COL = {"full": "#1f77b4", "base": "#ff7f0e", "clean": "#2ca02c", "grey": "#7f7f7f", "red": "#d62728"}
NAME = {"full": "full (PatchVision-X)", "base": "base (additions off)"}
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 100})

PATHS = PathsConfig(root=str(ROOT))
INFO = load_data_info(PATHS)
EV_TF = build_transforms("none", INFO.img_size, INFO.mean, INFO.std, train=False)
H = {n: pd.read_csv(ROOT / f"results/{n}/seed42/history.csv") for n in ("full", "base")}
TM = {n: json.loads((ROOT / f"results/{n}/seed42/test_metrics.json").read_text()) for n in ("full", "base")}
GRID = INFO.img_size // 16


def save(fig, name):
    fig.savefig(FIG / name, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  figure   ", name)


def save_gif(anim, name, fps, dpi=80):
    anim.save(ANI / name, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(anim._fig)
    print(f"  animation {name}  ({(ANI / name).stat().st_size / 1e6:.2f} MB)")


def unnorm(img):
    m, s = torch.tensor(INFO.mean)[:, None, None], torch.tensor(INFO.std)[:, None, None]
    return (img.cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


def show(ax, img, mask=None, color="lime", lw=1.6):
    ax.imshow(unnorm(img))
    if mask is not None and mask.max() > 0.5:
        ax.contour(mask[0].numpy(), levels=[0.5], colors=color, linewidths=lw)
    ax.axis("off")


def ds_of(csv, tf=EV_TF, **kw):
    return CUBDataset(csv, tf, INFO.bboxes, INFO.cub_root, **kw)


def idx_of_class(ds, name):
    for i, r in enumerate(ds.records):
        if r["class_name"] == name:
            return i
    return 0


# ============================================================================================
# 1. data
# ============================================================================================
def part_dataset():
    ds = ds_of(INFO.train_csv)
    labels = np.array([r["label"] for r in ds.records])
    classes = np.linspace(0, 199, 12).astype(int)
    fig, axes = plt.subplots(2, 6, figsize=(15, 5.6))
    for ax, c in zip(axes.ravel(), classes):
        img, m, y, _ = ds[int(np.where(labels == c)[0][0])]
        show(ax, img, m)
        ax.set_title(ds.class_names[y], fontsize=10)
    fig.suptitle("12 of the 200 bird species (green box = where the bird is, given by the dataset)", fontsize=13)
    save(fig, "dataset_samples.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1.5]})
    ax = axes[0]
    sizes = {"train": len(ds_of(INFO.train_csv)), "val": len(ds_of(INFO.val_csv)), "test": len(ds_of(INFO.test_csv))}
    bars = ax.bar(list(sizes), list(sizes.values()), color=[COL["full"], COL["clean"], COL["grey"]])
    for b, v in zip(bars, sizes.values()):
        ax.text(b.get_x() + b.get_width() / 2, v + 80, f"{v:,}", ha="center", fontsize=11)
    ax.set_title("How the 11,788 pictures are split")
    ax.set_ylabel("number of pictures")
    ax.set_ylim(0, 6800)
    ax = axes[1]
    ref = {"ImageNet-1k": 1281, "CIFAR-100": 500, "TinyImageNet": 500, "Oxford Pets": 99.5,
           "Flowers102": 20, "CUB-200 (ours)": len(ds_of(INFO.train_csv)) / 200}
    names, vals = list(ref), list(ref.values())
    cols = [COL["red"] if "ours" in n else COL["grey"] for n in names]
    ax.barh(names[::-1], vals[::-1], color=cols[::-1])
    ax.set_xscale("log")
    for i, v in enumerate(vals[::-1]):
        ax.text(v * 1.08, i, f"{v:,.0f}", va="center")
    ax.set_xlim(8, 4000)
    ax.set_title("Training pictures PER SPECIES (log scale): we have very few")
    fig.tight_layout(w_pad=3)
    save(fig, "dataset_sizes.png")


def part_augment():
    ds_eval = ds_of(INFO.train_csv)
    i = idx_of_class(ds_eval, "Painted Bunting")
    strong = build_transforms("strong", INFO.img_size, INFO.mean, INFO.std, train=True)
    ds_s = ds_of(INFO.train_csv, strong, train=True, box_crop_prob=0.0)
    ds_c = ds_of(INFO.train_csv, strong, train=True, box_crop_prob=1.0, box_crop_margin=(0.1, 0.4))
    random.seed(3)
    torch.manual_seed(3)
    items = [("original", ds_eval[i])] + [(f"strong augmentation {k + 1}", ds_s[i]) for k in range(7)] + \
            [(f"box-guided crop + augmentation {k + 1}", ds_c[i]) for k in range(4)]
    fig, axes = plt.subplots(3, 4, figsize=(12.5, 9.8))
    for ax, (t, (img, m, y, _)) in zip(axes.ravel(), items):
        show(ax, img, m)
        ax.set_title(t, fontsize=10)
    fig.suptitle("ONE training picture, many versions: the network never sees the same picture twice", fontsize=13)
    save(fig, "augmentations.png")

    # mixup / cutmix
    a = ds_eval[i]
    b = ds_eval[idx_of_class(ds_eval, "Green Jay")]
    lam = 0.6
    mix = lam * a[0] + (1 - lam) * b[0]
    cut = a[0].clone()
    y0, y1, x0, x1 = 20, 120, 100, 200
    cut[:, y0:y1, x0:x1] = b[0][:, y0:y1, x0:x1]
    lam_cut = 1 - (y1 - y0) * (x1 - x0) / 224 ** 2
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, (im, t) in zip(axes, [(a[0], "picture A: Painted Bunting"), (b[0], "picture B: Green Jay"),
                                  (mix, f"MIXUP\ntarget = {lam:.1f} x A + {1 - lam:.1f} x B"),
                                  (cut, f"CUTMIX\ntarget = {lam_cut:.2f} x A + {1 - lam_cut:.2f} x B")]):
        show(ax, im)
        ax.set_title(t, fontsize=11)
    fig.suptitle("Mixup and CutMix: blend two pictures and ask the model for a blended answer", fontsize=13)
    save(fig, "mixup_cutmix.png")


# ============================================================================================
# 2. how a ViT sees a picture
# ============================================================================================
def part_patches():
    ds = ds_of(INFO.test_csv)
    img, m, y, _ = ds[idx_of_class(ds, "Painted Bunting")]
    rgb = unnorm(img)
    fig = plt.figure(figsize=(15, 4.6))
    ax = fig.add_axes([0.0, 0.0, 0.25, 0.9])
    ax.imshow(rgb)
    ax.axis("off")
    ax.set_title("1. the photo (224 x 224 pixels)")
    ax = fig.add_axes([0.27, 0.0, 0.25, 0.9])
    ax.imshow(rgb)
    for k in range(0, 225, 16):
        ax.axhline(k - 0.5, color="white", lw=0.8)
        ax.axvline(k - 0.5, color="white", lw=0.8)
    ax.axis("off")
    ax.set_title("2. cut into 14 x 14 = 196 squares")
    ax = fig.add_axes([0.56, 0.05, 0.43, 0.8])
    ax.axis("off")
    ax.set_title("3. each square becomes one 'token' (a row of numbers)")
    order = [(r, c) for r in (4, 5) for c in range(3, 11)]
    for n, (r, c) in enumerate(order):
        tile = rgb[r * 16:(r + 1) * 16, c * 16:(c + 1) * 16]
        col, row = n % 8, 1 - n // 8
        ax.imshow(tile, extent=(col * 1.2, col * 1.2 + 1, row * 1.5 + 0.4, row * 1.5 + 1.4))
        ax.text(col * 1.2 + 0.5, row * 1.5 + 0.2, f"token {r * 14 + c}", ha="center", fontsize=7)
    ax.text(4.8, -0.5, "... and so on, 196 tokens in total", ha="center", fontsize=10, style="italic")
    ax.set_aspect("equal")
    ax.set_xlim(-0.1, 9.7)
    ax.set_ylim(-0.8, 3.1)
    save(fig, "patches.png")


def draw_box(ax, cx, cy, w, h, text, fc, ec="#444444", fs=10, ls="-", bold=False):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.4, ls=ls))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")


def arrow(ax, p, q, ls="-", color="#333333"):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, ls=ls))


def part_arch():
    fig, ax = plt.subplots(figsize=(12.5, 13.6))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 16.3)
    ax.axis("off")
    ax.grid(False)
    X = 4.3
    draw_box(ax, X, 15.55, 6.6, 0.9, "Bird photo\n224 x 224 pixels", "#fff2cc", bold=True)
    arrow(ax, (X, 15.05), (X, 14.35))
    draw_box(ax, X, 13.8, 6.6, 1.1, "Conv tokenizer (two small convolution layers)\n"
             "-> 14 x 14 = 196 tokens, each token = 256 numbers", "#dae8fc", bold=True)
    arrow(ax, (X, 13.2), (X, 12.55))
    ax.add_patch(Rectangle((0.7, 7.75), 7.2, 4.8, fc="#f7f7f7", ec="#888888", ls="--", lw=1.3))
    ax.text(0.9, 12.28, "STAGE 1  (196 tokens: fine details)   x 4 blocks", fontsize=10.5, fontweight="bold", color="#555555")
    draw_box(ax, X, 11.35, 6.4, 0.95, "Local Feature block (LFE)\n'who are my neighbours?'  (replaces position numbers)", "#d5e8d4")
    arrow(ax, (X, 10.87), (X, 10.55))
    draw_box(ax, X, 9.95, 6.4, 1.1, "Multi-scale graph attention\n8 heads in 4 groups: look at 3x3, 5x5, 9x9, or ALL tokens\n+ sparse gate (switch off useless links)", "#ffe6cc")
    arrow(ax, (X, 9.4), (X, 9.08))
    draw_box(ax, X, 8.55, 6.4, 0.85, "MLP (feed-forward)  'think about what I gathered'", "#e1d5e7")
    arrow(ax, (X, 7.75), (X, 7.2))
    draw_box(ax, X, 6.75, 6.6, 0.9, "Graph pooling: 196 tokens -> 49 tokens\n(merge similar squares)", "#dae8fc", bold=True)
    arrow(ax, (X, 6.3), (X, 5.8))
    ax.add_patch(Rectangle((0.7, 3.6), 7.2, 2.2, fc="#f7f7f7", ec="#888888", ls="--", lw=1.3))
    ax.text(0.9, 5.55, "STAGE 2  (49 tokens: the big picture)   x 4 blocks", fontsize=10.5, fontweight="bold", color="#555555")
    draw_box(ax, X, 4.5, 6.4, 1.1, "Multi-scale graph attention  +  MLP\n(same idea, now on 49 merged tokens)", "#ffe6cc")
    arrow(ax, (X, 3.6), (X, 3.05))
    draw_box(ax, X, 2.65, 6.6, 0.8, "Average of the 49 tokens = one summary of the photo", "#fff2cc")
    arrow(ax, (X, 2.25), (X, 1.7))
    draw_box(ax, X, 1.3, 6.6, 0.8, "Linear classifier -> 200 species scores", "#f8cecc", bold=True)

    ax.text(10.1, 15.6, "ONLY USED WHILE TRAINING", ha="center", fontsize=11, fontweight="bold", color="#aa4400")
    draw_box(ax, 10.1, 13.2, 3.9, 1.5, "Flipped copy of the photo\n-> consistency loss:\nsame answer, mirrored attention", "#fce5cd", ls="--", fs=9.5)
    arrow(ax, (8.15, 13.35), (7.65, 13.6), ls="--", color="#aa4400")
    draw_box(ax, 10.1, 9.95, 3.9, 1.5, "Bird box from the dataset\n-> bounding-box attention loss:\n'look at the bird!'", "#fce5cd", ls="--", fs=9.5)
    arrow(ax, (8.15, 9.95), (7.65, 9.95), ls="--", color="#aa4400")
    draw_box(ax, 10.1, 4.0, 3.9, 1.5, "EMA copy of the weights\n(slow-moving average)\nthis copy is used to test", "#fce5cd", ls="--", fs=9.5)
    arrow(ax, (8.15, 3.5), (7.65, 1.5), ls="--", color="#aa4400")
    ax.set_title("PatchVision-X: what happens to one photo", fontsize=14, fontweight="bold")
    save(fig, "architecture.png")


def part_scales():
    masks = grid_scale_masks(GRID, (1, 2, 4, 0))
    c = 6 * GRID + 6
    titles = ["head group 1\n3 x 3 neighbours", "head group 2\n5 x 5 neighbours", "head group 3\n9 x 9 neighbours", "head group 4\nALL tokens (global)"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    for g, ax in enumerate(axes):
        arr = masks[g, c].view(GRID, GRID).numpy()
        ax.imshow(arr, cmap=NB_CMAP, vmin=0, vmax=1)
        for s in ax.spines.values():
            s.set_visible(True)
        ax.scatter([6], [6], c="red", s=70, marker="*", zorder=3)
        ax.set_title(f"{titles[g]}\n({int(arr.sum())} of {GRID * GRID} tokens)", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle("Each head group lets the red star token look at a different amount of its surroundings", fontsize=13)
    save(fig, "multiscale_neighbourhoods.png")

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4))
    ims = []
    for g, ax in enumerate(axes):
        ims.append(ax.imshow(np.zeros((GRID, GRID)), cmap=NB_CMAP, vmin=0, vmax=1))
        for s in ax.spines.values():
            s.set_visible(True)
        ax.scatter([0], [0], c="red", s=60, marker="*", zorder=3)
        ax.set_xlim(-0.5, GRID - 0.5)          # fixed view: the star must not rescale the picture
        ax.set_ylim(GRID - 0.5, -0.5)
        ax.set_title(titles[g], fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    stars = [ax.collections[0] for ax in axes]
    path = [(r, 2) for r in range(2, 12)] + [(11, c_) for c_ in range(3, 12)] + [(r, 11) for r in range(10, 1, -1)] + \
           [(2, c_) for c_ in range(10, 2, -1)]
    path += [(r, c_) for r, c_ in [(4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (7, 5), (6, 3), (5, 7), (4, 9), (6, 10), (9, 6), (10, 4)]]
    txt = fig.suptitle("")

    def upd(k):
        r, c_ = path[k]
        for g in range(4):
            ims[g].set_data(masks[g, r * GRID + c_].view(GRID, GRID).numpy())
            stars[g].set_offsets([[c_, r]])
        txt.set_text(f"token at row {r}, column {c_}: the blue area is what each head group may look at")
        return ims

    save_gif(FuncAnimation(fig, upd, frames=len(path), interval=120), "neighbourhoods.gif", fps=8, dpi=75)


# ============================================================================================
# 3. training
# ============================================================================================
def part_lr():
    tc = TrainConfig()
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=tc.lr)
    sch = build_scheduler(opt, tc, 84)
    lrs = []
    for _ in range(tc.epochs * 84):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sch.step()
    ep = np.arange(len(lrs)) / 84
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(ep, lrs, color=COL["full"], lw=2)
    ax.axvspan(0, 10, color="#ffe9a8", alpha=0.6, label="warm-up (10 epochs): start gently")
    for n, c in (("full", COL["full"]), ("base", COL["base"])):
        e = int(H[n].epoch.max())
        ax.axvline(e, color=c, ls="--")
        ax.text(e + 2, tc.lr * 0.8, f"{n}: early stop\nat epoch {e}", color=c, fontsize=9)
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.set_title("Learning rate: gentle start, then slowly smaller (cosine). Early stopping ends runs before 300 epochs")
    ax.legend(loc="center right")
    save(fig, "lr_schedule.png")


def part_curves():
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4))
    ax = axes[0, 0]
    for n in ("base", "full"):
        ax.plot(H[n].epoch, H[n].val_acc1, color=COL[n], label=NAME[n])
        e = int(H[n].loc[H[n].val_acc1.idxmax(), "epoch"])
        ax.scatter([e], [H[n].val_acc1.max()], color=COL[n], zorder=3, s=40)
    ax.set_title("Accuracy on NEW pictures (validation): higher is better")
    ax.set_ylabel("top-1 accuracy (%)")
    ax.legend()
    ax = axes[0, 1]
    for n in ("base", "full"):
        ax.plot(H[n].epoch, H[n].val_loss, color=COL[n], label=NAME[n])
    ax.set_title("Validation loss: lower is better")
    ax.set_ylabel("cross-entropy")
    ax = axes[1, 0]
    for n in ("base", "full"):
        c = H[n].dropna(subset=["clean_acc1"])
        ax.plot(c.epoch, c.clean_acc1, color=COL[n], ls="--", label=f"{n}: TRAINING pictures")
        ax.plot(H[n].epoch, H[n].val_acc1, color=COL[n], label=f"{n}: NEW pictures")
    ax.set_title("Training vs new pictures: the gap is the memorizing")
    ax.set_ylabel("top-1 accuracy (%)")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    for n in ("base", "full"):
        ax.plot(H[n].epoch, H[n].val_attn_in_box, color=COL[n], label=NAME[n])
    ax.axhline(H["full"].val_box_area_frac.mean(), color=COL["grey"], ls=":", label="chance (box area / image area)")
    ax.set_title("Share of the model's attention that lands on the bird")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    for a in axes.ravel():
        a.set_xlabel("epoch")
    fig.tight_layout(h_pad=2.5, w_pad=2.5)
    save(fig, "training_curves.png")


def part_gap():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, n in zip(axes, ("base", "full")):
        c = H[n].dropna(subset=["clean_acc1"])
        v = H[n].set_index("epoch").val_acc1.reindex(c.epoch).values
        ax.plot(c.epoch, c.clean_acc1, color=COL["clean"], lw=2, label="TRAINING pictures (clean)")
        ax.plot(c.epoch, v, color=COL[n], lw=2, label="NEW pictures (validation)")
        ax.fill_between(c.epoch, v, c.clean_acc1, color="#f4b6b6", alpha=0.6, label="the memorizing gap")
        gap = c.clean_acc1.iloc[-1] - v[-1]
        ax.annotate(f"gap at the end = {gap:.0f} points", (c.epoch.iloc[-1], (c.clean_acc1.iloc[-1] + v[-1]) / 2), ha="right", fontsize=11, color=COL["red"], fontweight="bold")
        ax.set_title(NAME[n])
        ax.set_xlabel("epoch")
        ax.legend(loc="center left", fontsize=8)
    axes[0].set_ylabel("top-1 accuracy (%)")
    fig.suptitle("Both models memorize the 5,394 training pictures (about 100%) but do far worse on pictures they have never seen", fontsize=12)
    save(fig, "overfitting_gap.png")


def part_anim_curves():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.9))
    for a, t, yl in ((a1, "Accuracy on NEW pictures (validation)", 52), (a2, "Accuracy on TRAINING pictures (clean)", 105)):
        a.set_xlim(0, 250)
        a.set_ylim(0, yl)
        a.set_title(t, fontsize=10)
        a.set_xlabel("epoch")
    a1.set_ylabel("top-1 (%)")
    lines = {}
    for n in ("base", "full"):
        lines[(n, "val")], = a1.plot([], [], color=COL[n], lw=2, label=NAME[n])
        lines[(n, "clean")], = a2.plot([], [], color=COL[n], lw=2, marker="o", ms=2.5)
    a1.legend(loc="lower right", fontsize=8)
    txt = fig.suptitle("")
    frames = list(range(3, 247, 3)) + [246]

    def upd(k):
        for n in ("base", "full"):
            h = H[n]
            m = h.epoch <= k
            lines[(n, "val")].set_data(h.epoch[m], h.val_acc1[m])
            c = h[m].dropna(subset=["clean_acc1"])
            lines[(n, "clean")].set_data(c.epoch, c.clean_acc1)
        txt.set_text(f"epoch {k}")
        return list(lines.values())

    save_gif(FuncAnimation(fig, upd, frames=frames, interval=100), "training_progress.gif", fps=12, dpi=75)


def part_anim_gap():
    h = H["full"]
    c = h.dropna(subset=["clean_acc1"]).reset_index(drop=True)
    v = h.set_index("epoch").val_acc1.reindex(c.epoch).values
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.set_xlim(0, 215)
    ax.set_ylim(0, 105)
    ax.set_xlabel("epoch")
    ax.set_ylabel("top-1 accuracy (%)")
    l1, = ax.plot([], [], color=COL["clean"], lw=2.4, label="TRAINING pictures (it memorizes them)")
    l2, = ax.plot([], [], color=COL["full"], lw=2.4, label="NEW pictures (what we care about)")
    ax.legend(loc="lower right", fontsize=8)
    state = {"fill": None}
    txt = ax.text(5, 92, "", fontsize=12, color=COL["red"], fontweight="bold")

    def upd(i):
        if state["fill"] is not None:
            state["fill"].remove()
        e, cc, vv = c.epoch[:i + 1], c.clean_acc1[:i + 1], v[:i + 1]
        l1.set_data(e, cc)
        l2.set_data(e, vv)
        state["fill"] = ax.fill_between(e, vv, cc, color="#f4b6b6", alpha=0.7)
        txt.set_text(f"epoch {int(c.epoch[i])}:  gap = {cc.iloc[-1] - vv[-1]:.0f} points")
        return l1, l2

    save_gif(FuncAnimation(fig, upd, frames=len(c), interval=140), "memorizing_gap.gif", fps=7, dpi=80)


def part_anim_snapshots():
    files = sorted((ROOT / "results/full/seed42/attention_snapshots").glob("epoch_*.png"))
    files = [f for f in files if f.stem != "epoch_213"]
    imgs = [np.asarray(Image.open(f).convert("RGB").crop((0, 52, 1110, 845)).resize((740, 529))) for f in files]
    fig, ax = plt.subplots(figsize=(7.4, 5.5))
    fig.subplots_adjust(0, 0, 1, 0.93)
    ax.axis("off")
    ax.grid(False)
    im = ax.imshow(imgs[0])
    txt = fig.suptitle("", y=0.985, fontsize=11)

    def upd(k):
        im.set_data(imgs[k])
        e = int(files[k].stem.split("_")[1])
        txt.set_text(f"where the model looks, training epoch {e}")
        return im,

    save_gif(FuncAnimation(fig, upd, frames=len(files), interval=500), "attention_over_training.gif", fps=2, dpi=75)


def part_anim_aug():
    ds_eval = ds_of(INFO.train_csv)
    i = idx_of_class(ds_eval, "Painted Bunting")
    strong = build_transforms("strong", INFO.img_size, INFO.mean, INFO.std, train=True)
    ds_s = ds_of(INFO.train_csv, strong, train=True, box_crop_prob=0.3)
    random.seed(11)
    torch.manual_seed(11)
    items = [ds_eval[i]] + [ds_s[i] for _ in range(23)]
    fig, ax = plt.subplots(figsize=(3.6, 3.9))
    fig.subplots_adjust(0, 0, 1, 0.9)
    ax.axis("off")
    ax.grid(False)
    im = ax.imshow(unnorm(items[0][0]))
    cont = {"c": None}
    txt = fig.suptitle("", fontsize=10)

    def upd(k):
        img, m, _, _ = items[k]
        im.set_data(unnorm(img))
        if cont["c"] is not None:
            try:
                cont["c"].remove()                      # matplotlib >= 3.8
            except Exception:
                for coll in cont["c"].collections:
                    coll.remove()
        cont["c"] = ax.contour(m[0].numpy(), levels=[0.5], colors="lime", linewidths=2) if m.max() > 0.5 else None
        txt.set_text("original picture" if k == 0 else f"training version {k}")
        return im,

    save_gif(FuncAnimation(fig, upd, frames=len(items), interval=500), "augmentations.gif", fps=2, dpi=80)


# ============================================================================================
# 4. results
# ============================================================================================
def part_results():
    keys = [("test_acc1", "top-1"), ("test_acc5", "top-5"), ("test_macro_f1", "macro-F1")]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    w = 0.36
    for j, n in enumerate(("base", "full")):
        vals = [TM[n]["row"][k] for k, _ in keys]
        bars = ax.bar(np.arange(3) + (j - 0.5) * w, vals, w, color=COL[n], label=NAME[n])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", fontsize=10)
    ax.set_xticks(range(3))
    ax.set_xticklabels([l for _, l in keys])
    ax.set_ylabel("%")
    ax.set_ylim(0, 85)
    ax.set_title("Test results on 5,794 never-seen pictures (seed 42)")
    ax.legend()
    save(fig, "results_bars.png")


def part_extra_losses():
    h = H["full"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    axes[0].plot(h.epoch, h.train_ce, color=COL["full"])
    axes[0].set_title("main loss: cross-entropy\n(guess the species)")
    axes[1].plot(h.epoch, h.train_attn, color=COL["base"])
    axes[1].set_title("bounding-box attention loss\n(lower = more attention on the bird)")
    axes[2].plot(h.epoch, h.train_cons_logit, label="prediction differs when flipped")
    axes[2].plot(h.epoch, h.train_cons_attn, label="attention differs when flipped")
    axes[2].set_title("flip-consistency loss\n(lower = more mirror-consistent)")
    axes[2].legend(fontsize=8)
    for a in axes:
        a.set_xlabel("epoch")
    save(fig, "extra_losses.png")


def part_ema():
    h = H["full"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(h.epoch, h.valraw_acc1, color=COL["grey"], label="raw weights (current)")
    ax.plot(h.epoch, h.val_acc1, color=COL["full"], lw=2, label="EMA weights (slow average)")
    late = h[h.epoch >= 50]
    gain = (late.val_acc1 - late.valraw_acc1).mean()
    ax.set_title(f"EMA helps a little: on average {gain:+.1f} points over the raw weights (epochs 50 to {int(h.epoch.max())})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation top-1 (%)")
    ax.legend()
    save(fig, "ema_effect.png")
    print(f"    EMA mean gain epochs>=50: {gain:.2f} points")


def part_per_class():
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    for n in ("base", "full"):
        pc = pd.read_csv(ROOT / f"results/{n}/seed42/test_per_class.csv")
        ax.plot(np.arange(200), np.sort(pc.f1.values)[::-1], color=COL[n], lw=2, label=f"{NAME[n]} (median {pc.f1.median():.0f})")
    ax.set_xlabel("species, from easiest to hardest")
    ax.set_ylabel("F1 score (%)")
    ax.set_title("Every one of the 200 species, sorted by how well it is recognised")
    ax.legend()
    ax = axes[1]
    pc = pd.read_csv(ROOT / "results/full/seed42/test_per_class.csv")
    conf = np.load(ROOT / "results/full/seed42/test_confusion_matrix.npy").copy()
    np.fill_diagonal(conf, 0)
    top = np.dstack(np.unravel_index(np.argsort(-conf.ravel())[:8], conf.shape))[0]
    labels = [f"{pc.class_name[i]}\n-> {pc.class_name[j]}" for i, j in top][::-1]
    ax.barh(labels, [conf[i, j] for i, j in top][::-1], color=COL["full"])
    ax.set_xlabel("test pictures confused")
    ax.set_title("Most common mistakes of the full model\n(true species -> what it said)")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout(w_pad=4)
    save(fig, "per_class_and_confusions.png")


def part_calibration():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, n in zip(axes, ("base", "full")):
        r = pd.read_csv(ROOT / f"results/{n}/seed42/test_reliability.csv")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        ax.plot(r.confidence, r.accuracy, "o-", color=COL[n], label="model")
        ax.set_xlabel("how sure the model says it is")
        ax.set_ylabel("how often it is actually right")
        ax.set_title(f"{NAME[n]}\nECE = {TM[n]['row']['test_ece']:.2f}%")
        ax.legend()
    fig.suptitle("Calibration: are the confidence numbers honest? (closer to the dashed line is better)", fontsize=12)
    save(fig, "calibration.png")


# ============================================================================================
# 5. what the trained models actually look at
# ============================================================================================
def _models():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    for n in ("base", "full"):
        m, _ = load_for_inference(ROOT / f"results/{n}/seed42", dev)
        out[n] = m
    return out, dev


@torch.no_grad()
def _heat(model, x, dev):
    logits, aux = model(x.to(dev))
    conf, pred = torch.softmax(logits.float(), -1).max(1)
    a = aux["attn_maps"][-1].float().view(-1, 1, GRID, GRID)
    a = F.interpolate(a, size=x.shape[-2:], mode="bicubic", align_corners=False).clamp_min(0).cpu()
    return a, pred.cpu(), conf.cpu(), aux


def part_attention_compare():
    models, dev = _models()
    ds = ds_of(INFO.test_csv)
    idx = np.linspace(0, len(ds) - 1, 8).astype(int)
    items = [ds[int(i)] for i in idx]
    x = torch.stack([it[0] for it in items])
    res = {n: _heat(models[n], x, dev) for n in models}
    fig, axes = plt.subplots(3, 8, figsize=(19, 7.6))
    for p, (img, m, y, _) in enumerate(items):
        show(axes[0, p], img, m)
        axes[0, p].set_title(ds.class_names[y][:20], fontsize=9)
        for r, n in ((1, "base"), (2, "full")):
            heat, pred, conf, _ = res[n]
            ax = axes[r, p]
            ax.imshow(unnorm(img))
            ax.imshow(heat[p, 0].numpy(), cmap="jet", alpha=0.5)
            ax.contour(m[0].numpy(), levels=[0.5], colors="white", linewidths=0.8)
            ax.axis("off")
            ok = int(pred[p]) == y
            ax.set_title(f"{ds.class_names[int(pred[p])][:18]} {conf[p]:.2f}", fontsize=8.5, color="green" if ok else "red")
    axes[0, 0].text(-0.06, 0.5, "photo\n(green = bird box)", transform=axes[0, 0].transAxes, rotation=90, va="center", ha="right", fontsize=10)
    axes[1, 0].text(-0.06, 0.5, "base model:\nwhere it looks", transform=axes[1, 0].transAxes, rotation=90, va="center", ha="right", fontsize=10, color=COL["base"])
    axes[2, 0].text(-0.06, 0.5, "full model:\nwhere it looks", transform=axes[2, 0].transAxes, rotation=90, va="center", ha="right", fontsize=10, color=COL["full"])
    fig.suptitle("Same 8 test photos (evenly spaced, NOT cherry-picked). Titles = predicted species and confidence, green = correct, red = wrong", fontsize=12)
    save(fig, "attention_base_vs_full.png")


def part_flip():
    models, dev = _models()
    ds = ds_of(INFO.test_csv)
    sub = np.linspace(0, len(ds) - 1, 256).astype(int)
    stats = {}
    for n, model in models.items():
        kl_last, kl_mean, agree = [], [], []
        with torch.no_grad():
            for b in range(0, len(sub), 64):
                x = torch.stack([ds[int(i)][0] for i in sub[b:b + 64]]).to(dev)
                l1, a1 = model(x)
                l2, a2 = model(x.flip(-1))
                kl_last.append(sym_kl_probs(a1["attn_maps"][-1], flip_tokens(a2["attn_maps"][-1], GRID)).item())
                kl_mean.append(np.mean([sym_kl_probs(p, flip_tokens(q, GRID)).item() for p, q in zip(a1["attn_maps"], a2["attn_maps"])]))
                agree.append((l1.argmax(1) == l2.argmax(1)).float().mean().item())
        stats[n] = {"attn_mismatch_last_block": float(np.mean(kl_last)), "attn_mismatch_mean_blocks": float(np.mean(kl_mean)),
                    "same_prediction_when_flipped": float(np.mean(agree)), "n_images": len(sub)}
    (FIG / "flip_stats.json").write_text(json.dumps(stats, indent=2))
    print("    flip stats:", json.dumps(stats))

    img, m, y, _ = ds[int(sub[37])]
    x = img[None]
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.2))
    for r, n in enumerate(("base", "full")):
        h1, p1, c1, _ = _heat(models[n], x, dev)
        h2, p2, c2, _ = _heat(models[n], x.flip(-1), dev)
        panels = [(unnorm(img), None, "photo"), (unnorm(img), h1[0, 0].numpy(), f"attention\n(predicts {ds.class_names[int(p1)][:16]})"),
                  (unnorm(img.flip(-1)), None, "mirrored photo"), (unnorm(img.flip(-1)), h2[0, 0].numpy(), f"attention on mirror\n(predicts {ds.class_names[int(p2)][:16]})"),
                  (unnorm(img), h2[0, 0].flip(-1).numpy(), "mirror attention flipped back\n(should match column 2)")]
        for ax, (im, heat, t) in zip(axes[r], panels):
            ax.imshow(im)
            if heat is not None:
                ax.imshow(heat, cmap="jet", alpha=0.5)
            ax.set_title(t, fontsize=9)
            ax.axis("off")
        axes[r, 0].text(-0.05, 0.5, n, transform=axes[r, 0].transAxes, rotation=90, va="center", ha="right", fontsize=12, color=COL[n], fontweight="bold")
    fig.suptitle(f"Flip test. Attention mismatch over {len(sub)} test photos (lower = more consistent): "
                 f"base {stats['base']['attn_mismatch_last_block']:.4f}, full {stats['full']['attn_mismatch_last_block']:.4f}", fontsize=11)
    save(fig, "flip_consistency.png")


PARTS = {
    "dataset": part_dataset, "augment": part_augment, "patches": part_patches, "arch": part_arch, "scales": part_scales,
    "lr": part_lr, "curves": part_curves, "gap": part_gap, "anim_curves": part_anim_curves, "anim_gap": part_anim_gap,
    "anim_snap": part_anim_snapshots, "anim_aug": part_anim_aug, "results": part_results, "losses": part_extra_losses,
    "ema": part_ema, "perclass": part_per_class, "calib": part_calibration, "attn": part_attention_compare, "flip": part_flip,
}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(PARTS)
    for k in wanted:
        print(f"[{k}]")
        try:
            PARTS[k]()
        except Exception:
            traceback.print_exc()
