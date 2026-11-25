"""Figures, GIF and facts for CNN_Baseline/README.md: round 01 (strong augmentation + Mixup/CutMix) vs round 02
(auto augmentation, no Mixup/CutMix, regularization search) for CNN small and CNN large.

Reads only finished results: 01_strongaug_mixup_cutmix/results/<model>/seed42/ and
02_autoaug_nomix_regularization/results/<model>/<run>/seed42/. Run from anywhere:
    python CNN_Baseline/docs/make_comparison.py
"""
from __future__ import annotations

import json
import random
import sys
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

HERE = Path(__file__).resolve().parents[1]                       # CNN_Baseline/
ROOT = HERE.parent
R01, R02 = HERE / "01_strongaug_mixup_cutmix" / "results", HERE / "02_autoaug_nomix_regularization" / "results"
FIG, ANI = HERE / "docs" / "figures", HERE / "docs" / "animations"
for d in (FIG, ANI):
    d.mkdir(parents=True, exist_ok=True)
CUB = ROOT / "CUB_200_2011"

INK, MUTED, GRID = "#222222", "#666666", "#e6e6e6"
C_SMALL, C_LARGE = "#ff7f0e", "#2ca02c"                          # same model colours as the round READMEs
LIGHT = {C_SMALL: "#ffd2a8", C_LARGE: "#b5e0b5"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100,
                     "axes.edgecolor": "#999999", "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.titleweight": "bold", "axes.titlesize": 11.5})
MODELS = {"cnn_small": ("CNN small", C_SMALL), "cnn_large": ("CNN large", C_LARGE)}
FINAL = "r8_combined"                                            # round 02 final run of both models
FACTS: list[str] = []


def fact(s):
    FACTS.append(s)
    print("  fact:", s)


def savefig(fig, name):
    fig.savefig(FIG / name, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", FIG / name)


def header(fig, title, sub, top=0.88):
    fig.suptitle(title, x=0.01, y=0.995, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.93, sub, fontsize=9, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0, 1, top))


def grid(a, axis="y"):
    a.grid(axis=axis, color=GRID, lw=0.8)
    a.set_axisbelow(True)


def rdir(model, rnd, run=FINAL):
    return R01 / model / "seed42" if rnd == 1 else R02 / model / run / "seed42"


def load(model, rnd, run=FINAL):
    d = rdir(model, rnd, run)
    m = json.loads((d / "test_metrics.json").read_text())
    r = m["row"]
    preds = pd.read_csv(d / "test_predictions.csv").sort_values("image_id").reset_index(drop=True)
    return {"model": model, "round": rnd, "run": run, "best_epoch": r["best_epoch"], "epochs": r["epochs_run"],
            "val": r["val_acc1_at_best"], "gap": r["overfit_gap_clean_minus_val"], "top1": r["test_acc1"],
            "top5": r["test_acc5"], "f1": r["test_macro_f1"], "ece": r["test_ece"], "min": r["train_time_s"] / 60,
            "cfg": m["config"], "hist": pd.read_csv(d / "history.csv"), "preds": preds,
            "pc": pd.read_csv(d / "test_per_class.csv"), "rel": pd.read_csv(d / "test_reliability.csv")}


def mcnemar(a, b):
    """a, b = loaded runs. Returns (a minus b in points, only a right, only b right, exact two-sided p)."""
    ca = (a["preds"].pred == a["preds"].label).values
    cb = (b["preds"].pred == b["preds"].label).values
    n10, n01 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    n, k = n10 + n01, min(n10, n01)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return (ca.mean() - cb.mean()) * 100, n10, n01, p


# ------------------------------------------------------------------------------------------------------------
# 1. headline: round 01 vs round 02 final, five measures
# ------------------------------------------------------------------------------------------------------------
def fig_headline(D):
    meas = [("top1", "Test top-1, %", True), ("top5", "Test top-5, %", True), ("f1", "Test macro-F1, %", True),
            ("gap", "Memorizing gap, points", False), ("ece", "Calibration error (ECE), %", False)]
    fig, ax = plt.subplots(1, 5, figsize=(19, 4.6))
    for a, (k, t, up) in zip(ax, meas):
        for i, m in enumerate(MODELS):
            c = MODELS[m][1]
            for j, rnd in enumerate((1, 2)):
                v = D[m][rnd][k]
                x = i * 2.6 + j
                a.bar(x, v, 0.9, color=c if rnd == 2 else LIGHT[c], hatch="" if rnd == 2 else "//",
                      edgecolor="white")
                a.text(x, v + 0.6, f"{v:.1f}" if k in ("gap", "ece") else f"{v:.2f}", ha="center", fontsize=9,
                       fontweight="bold" if rnd == 2 else "normal", color=INK)
        a.set_xticks([0, 1, 2.6, 3.6])
        a.set_xticklabels(["round\n01", "round\n02", "round\n01", "round\n02"], fontsize=8.5)
        for i, m in enumerate(MODELS):
            a.text(i * 2.6 + 0.5, -0.2, MODELS[m][0], transform=a.get_xaxis_transform(), ha="center",
                   color=MODELS[m][1], fontweight="bold")
        a.set_title(t + ("  (higher is better)" if up else "  (lower is better)"), fontsize=10, loc="left")
        top = max(D[m][r][k] for m in MODELS for r in (1, 2))
        a.set_ylim(0, top * 1.15)
        grid(a)
    header(fig, "Round 01 vs round 02, both CNNs, same 5,794 test photos",
           "Hatched = round 01 (strong augmentation + Mixup/CutMix).  Solid = round 02 final "
           "(auto augmentation, no Mixup/CutMix, weight decay 0.2 + label smoothing 0.2 + drop-path 0.2, "
           "+ SAM for CNN large).")
    savefig(fig, "x01_round01_vs_round02.png")


# ------------------------------------------------------------------------------------------------------------
# 2. what one training photo looks like in each round (the project's real transforms)
# ------------------------------------------------------------------------------------------------------------
def fig_photos():
    import torch
    from PIL import Image
    from torchvision import tv_tensors
    from torchvision.transforms import v2 as T
    sys.path.insert(0, str(ROOT / "Vision_Transformer"))
    from patchvision.data import MixupCutmix, build_transforms
    mean = [0.4859701097011566, 0.49921685457229614, 0.43137678503990173]
    std = [0.22647330164909363, 0.2219720035791397, 0.2609000504016876]
    train = pd.read_csv(ROOT / "DataPreprocessing_Results" / "train.csv")

    def img(name):
        fp = train[train.class_name == name].iloc[0].filepath.replace("\\", "/").split("CUB_200_2011/", 1)[1]
        return tv_tensors.Image(T.functional.pil_to_tensor(Image.open(CUB / fp).convert("RGB")))

    def show(t):
        if t.dtype == torch.uint8:
            return (t.float() / 255).permute(1, 2, 0).numpy()
        m, s = torch.tensor(mean)[:, None, None], torch.tensor(std)[:, None, None]
        return (t.float() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()

    bird, other = img("Painted Bunting"), img("Blue Jay")
    strong, auto = (build_transforms(t, 224, mean, std, True) for t in ("strong", "auto"))
    torch.manual_seed(3)
    random.seed(3)
    np.random.seed(3)
    r1 = [strong(bird) for _ in range(4)]
    pair = torch.stack([strong(bird), strong(other)])
    dummy = torch.zeros(2, 1, 224, 224)
    mixed = []
    for pc in (0.0, 1.0):                                    # one Mixup and one CutMix example, as in training
        np.random.seed(11 if pc == 0 else 5)
        x, t, _, info = MixupCutmix(200, 0.2, 1.0, pc, 0.1)(pair, torch.tensor([0, 1]), dummy)
        mixed.append((x[0], info))
    bmask = tv_tensors.Mask(torch.zeros(1, *bird.shape[-2:]))     # the bird-box mask is not used by the CNN
    r2 = [auto(bird, bmask)[0] for _ in range(6)]
    fig, ax = plt.subplots(2, 7, figsize=(19, 7.6))
    ax[0, 0].imshow(show(T.Resize((224, 224))(bird)))
    ax[0, 0].set_title("original photo", fontsize=9.5)
    ax[1, 0].imshow(show(T.Resize((224, 224))(bird)))
    ax[1, 0].set_title("original photo", fontsize=9.5)
    for i, t in enumerate(r1):
        ax[0, i + 1].imshow(show(t))
        ax[0, i + 1].set_title(f"strong view {i + 1}", fontsize=9.5)
    for j, (x, info) in enumerate(mixed):
        lam = info["lam"]
        ax[0, 5 + j].imshow(show(x))
        ax[0, 5 + j].set_title(f"+ {'Mixup' if info['type'] == 'mixup' else 'CutMix'} with a Blue Jay\n"
                               f"label {lam * 100:.0f} % bunting / {100 - lam * 100:.0f} % jay", fontsize=9,
                               color="#c0392b")
    for i, t in enumerate(r2):
        ax[1, i + 1].imshow(show(t))
        ax[1, i + 1].set_title(f"auto view {i + 1}", fontsize=9.5)
    for a in ax.ravel():
        a.axis("off")
    fig.subplots_adjust(left=0.035, right=0.995, top=0.80, bottom=0.01, hspace=0.32, wspace=0.05)
    fig.text(0.012, 0.62, "ROUND 01", rotation=90, fontsize=13, fontweight="bold", color=INK, va="center")
    fig.text(0.012, 0.20, "ROUND 02", rotation=90, fontsize=13, fontweight="bold", color=INK, va="center")
    fig.suptitle("What the model is shown: the same training photo in each round", x=0.01, y=0.99, ha="left",
                 fontsize=14, fontweight="bold")
    fig.text(0.01, 0.925, "Round 01: crop + flip + rotate/shift/zoom + colour change + erase, ALL stacked on every "
             "photo, then every batch is mixed (Mixup or CutMix).  Round 02: crop + flip + ONE random TrivialAugment "
             "operation + erase, never mixed.", fontsize=9, color=MUTED, ha="left")
    savefig(fig, "x02_training_photos.png")


# ------------------------------------------------------------------------------------------------------------
# 3. where the gain came from: round 01 -> r0 (new recipe) -> r6 (weight decay 0.2) -> r8 (all winners)
# ------------------------------------------------------------------------------------------------------------
STEPS = [(1, None, "round 01\nstrong aug +\nMixup/CutMix"), (2, "r0_reference", "r0\nauto aug,\nno mixing"),
         (2, "r6_wd0.2", "r6\n+ weight decay\n0.2"), (2, "r8_combined", "r8\n+ the other\nwinners")]


def fig_steps(D):
    fig, ax = plt.subplots(2, 2, figsize=(15, 8.6))
    for col, m in enumerate(MODELS):
        name, c = MODELS[m]
        runs = [D[m][1] if rnd == 1 else load(m, 2, run) for rnd, run, _ in STEPS]
        for row, (k, t) in enumerate((("top1", "Test top-1, %"), ("gap", "Memorizing gap, points"))):
            a = ax[row, col]
            vals = [r[k] for r in runs]
            cols = [LIGHT[c]] + [c] * 3
            a.bar(range(4), vals, 0.6, color=cols, edgecolor="white", hatch=None)
            for i, v in enumerate(vals):
                a.text(i, v + 0.3, f"{v:.2f}" if k == "top1" else f"{v:.1f}", ha="center", fontsize=9.5,
                       fontweight="bold")
                if i:
                    d = v - vals[i - 1]
                    txt = f"{d:+.2f}"
                    if k == "top1":
                        _, n10, n01, p = mcnemar(runs[i], runs[i - 1])
                        txt += f"\np = {p:.2g}" if p >= 1e-4 else "\np < 0.0001"
                    a.annotate(txt, xy=(i - 0.5, max(vals[i - 1], v) + (2.2 if k == "top1" else 1.6)),
                               ha="center", fontsize=8.5, color="#1a7f37" if (d > 0) == (k == "top1") else
                               "#c0392b")
            a.set_xticks(range(4))
            a.set_xticklabels([s[2] for s in STEPS], fontsize=8.5)
            lo = min(vals) - (6 if k == "top1" else 4)
            a.set_ylim(lo, max(vals) + (5 if k == "top1" else 3.5))
            a.set_title(f"{name}: {t}", loc="left", color=c)
            grid(a)
            if row == 0:
                for i in range(1, 4):
                    _, n10, n01, p = mcnemar(runs[i], runs[i - 1])
                    fact(f"{name} step {STEPS[i - 1][1] or 'round01'} -> {STEPS[i][1]}: "
                         f"{runs[i]['top1'] - runs[i - 1]['top1']:+.2f}, {n10}/{n01}, p={p:.2g}")
    header(fig, "Where the round 02 gain came from, step by step",
           "Each bar adds one thing to the bar on its left. Number between bars = change (green better, red worse) (p = exact paired"
           " test on the same test photos). The y-axes do not start at zero.", top=0.9)
    savefig(fig, "x03_where_the_gain_came_from.png")


# ------------------------------------------------------------------------------------------------------------
# 4. training curves: round 01 (dashed) vs round 02 final (solid)
# ------------------------------------------------------------------------------------------------------------
def fig_curves(D):
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.8))
    for m in MODELS:
        name, c = MODELS[m]
        for rnd, ls, lw in ((1, "--", 1.6), (2, "-", 2.2)):
            h = D[m][rnd]["hist"].set_index("epoch")
            lab = f"{name}, round 0{rnd}"
            ax[0].plot(h.index, h.val_acc1.rolling(7, center=True, min_periods=1).mean(), color=c, ls=ls, lw=lw,
                       label=lab)
            cl = h.clean_acc1.dropna()
            ax[1].plot(cl.index, cl.values, color=c, ls=ls, lw=lw, label=lab)
            g = h.gap_acc1.dropna() if "gap_acc1" in h else None
            gg = (cl - h.val_acc1.loc[cl.index]).dropna()
            ax[2].plot(gg.index, gg.values, color=c, ls=ls, lw=lw, label=lab)
            ax[3].plot(h.index, h.val_loss, color=c, ls=ls, lw=lw, label=lab)
            if rnd == 1:
                e = D[m][1]["epochs"]
                ax[0].axvline(e, color=c, ls=":", lw=1)
                ax[0].text(e + 3 if m == "cnn_small" else e - 3, 12 if m == "cnn_small" else 22,
                           f"round 01\n{name}\nstopped\n(epoch {e})", fontsize=8, color=c,
                           ha="left" if m == "cnn_small" else "right")
    for a, t, yl in zip(ax, ("Validation top-1 (new photos)", "Top-1 on training photos, no augmentation",
                             "Memorizing gap (training minus validation)", "Validation loss"),
                        ("top-1, %", "top-1, %", "points", "loss")):
        a.set_title(t, loc="left", fontsize=10.5)
        a.set_xlabel("epoch")
        a.set_ylabel(yl)
        a.set_xlim(0, 300)
        grid(a, "both")
    ax[0].legend(frameon=False, fontsize=8, loc="upper left")
    ax[3].set_ylim(2, 5.5)
    header(fig, "How training went: round 01 (dashed) vs round 02 final (solid)",
           "Round 01 used early stopping (patience 60); round 02 trains all 300 epochs. Validation top-1 smoothed over "
           "7 epochs; training-photo accuracy measured every 5 epochs on 600 photos.")
    savefig(fig, "x04_training_curves.png")


# ------------------------------------------------------------------------------------------------------------
# 5. calibration
# ------------------------------------------------------------------------------------------------------------
def fig_calibration(D):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
    for a, m in zip(ax, MODELS):
        name, c = MODELS[m]
        a.plot([0, 100], [0, 100], color="#999999", lw=1, ls=":")
        for rnd, ls, mk in ((1, "--", "o"), (2, "-", "s")):
            r = D[m][rnd]
            rel = r["rel"][r["rel"].fraction > 0.003]
            conf = (r["preds"].confidence.mean() * 100)
            a.plot(rel.confidence * 100, rel.accuracy * 100, ls=ls, marker=mk, color=c if rnd == 2 else "#a0a0a0",
                   lw=2, label=f"round 0{rnd}: ECE {r['ece']:.1f} %, says {conf:.0f} % on average, right "
                               f"{r['top1']:.0f} %")
            fact(f"{name} round 0{rnd}: mean confidence {conf:.1f}, accuracy {r['top1']:.2f}, ECE {r['ece']:.2f}")
        a.set_xlim(0, 100)
        a.set_ylim(0, 100)
        a.set_xlabel("confidence the model gives, %")
        a.set_ylabel("how often it is right, %")
        a.set_title(name, loc="left", color=c)
        a.text(70, 32, "below the line =\nover-confident", color=MUTED, fontsize=9)
        a.text(8, 80, "above the line =\nunder-confident", color=MUTED, fontsize=9)
        a.legend(frameon=True, fontsize=8.5, loc="lower right", framealpha=0.95, edgecolor="#dddddd")
        grid(a, "both")
    header(fig, "Are the confidence numbers honest? Round 01 vs round 02",
           "Dotted diagonal = perfectly honest. Round 02's stronger regularization (weight decay 0.2 + label "
           "smoothing 0.2) makes both models under-confident.", top=0.87)
    savefig(fig, "x05_calibration.png")


# ------------------------------------------------------------------------------------------------------------
# 6. species
# ------------------------------------------------------------------------------------------------------------
def fig_species(D):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6.4))
    for a, m in zip(ax, MODELS):
        name, c = MODELS[m]
        f1, f2 = D[m][1]["pc"].f1.values, D[m][2]["pc"].f1.values
        names = D[m][1]["pc"].class_name.values
        d = f2 - f1
        up, dn, eq = int((d > 1e-9).sum()), int((d < -1e-9).sum()), int((abs(d) <= 1e-9).sum())
        a.scatter(f1, f2, s=22, c=np.where(d > 0, c, "#b0b0b0"), edgecolor="white", lw=0.5)
        a.plot([0, 100], [0, 100], color="#999999", lw=1, ls=":")
        spots = [(4, 94), (4, 87), (4, 80), (62, 14), (62, 7)]           # text positions in empty corners
        for (tx, ty), i in zip(spots, list(np.argsort(d)[::-1][:3]) + list(np.argsort(d)[:2])):
            a.annotate(f"{names[i]} ({d[i]:+.0f})", (f1[i], f2[i]), xytext=(tx, ty), textcoords="data", fontsize=8,
                       color=INK, arrowprops=dict(arrowstyle="-", color="#999999", lw=0.7))
        a.set_xlim(0, 100)
        a.set_ylim(0, 100)
        a.set_xlabel("round 01 F1 per species, %")
        a.set_ylabel("round 02 F1 per species, %")
        a.set_title(f"{name}: {up} species better, {dn} worse, {eq} same  (median {np.median(d):+.1f})",
                    loc="left", color=c, fontsize=10.5)
        grid(a, "both")
        fact(f"{name} species: up {up}, down {dn}, same {eq}, median {np.median(d):+.1f}; best "
             f"{[(names[i], round(d[i])) for i in np.argsort(d)[-3:]]}, worst {[(names[i], round(d[i])) for i in np.argsort(d)[:3]]}")
    header(fig, "Which species improved? Each dot is one of the 200 species",
           "Above the diagonal (coloured) = better in round 02. 30 test photos per species, so a single species can "
           "move by 10+ points by chance; the count of species that improved is the robust part.", top=0.88)
    savefig(fig, "x06_species.png")


# ------------------------------------------------------------------------------------------------------------
# 7. photos fixed and broken by round 02
# ------------------------------------------------------------------------------------------------------------
def fig_fixed(D):
    test = pd.read_csv(ROOT / "DataPreprocessing_Results" / "test.csv").set_index("image_id")
    names = D["cnn_small"][1]["pc"].set_index("class_id").class_name.to_dict()
    ok = {(m, r): (D[m][r]["preds"].pred == D[m][r]["preds"].label).values for m in MODELS for r in (1, 2)}
    ids = D["cnn_small"][1]["preds"].image_id.values
    lab = D["cnn_small"][1]["preds"].label.values
    fixed = np.where(~ok["cnn_small", 1] & ~ok["cnn_large", 1] & ok["cnn_small", 2] & ok["cnn_large", 2])[0]
    broke = np.where(ok["cnn_small", 1] & ok["cnn_large", 1] & ~ok["cnn_small", 2] & ~ok["cnn_large", 2])[0]
    fact(f"photos fixed for both models {len(fixed)}, broken for both {len(broke)}")
    pick = list(fixed[np.linspace(0, len(fixed) - 1, 8).astype(int)]) + \
        list(broke[np.linspace(0, len(broke) - 1, 4).astype(int)])
    from PIL import Image
    fig, axes = plt.subplots(2, 6, figsize=(19, 9.6))
    for n, (a, i) in enumerate(zip(axes.ravel(), pick)):
        fp = str(test.loc[ids[i], "filepath"]).replace("\\", "/").split("CUB_200_2011/", 1)[1]
        im = Image.open(CUB / fp).convert("RGB")
        w, h = im.size
        k = min(w, h)
        im = im.crop(((w - k) // 2, (h - k) // 2, (w + k) // 2, (h + k) // 2)).resize((300, 300))
        a.imshow(im)
        a.axis("off")
        good = n < 8
        a.set_title(("FIXED  " if good else "BROKEN  ") + names[lab[i]][:24], fontsize=9, fontweight="bold",
                    color="#1a7f37" if good else "#c0392b")
        p1 = D["cnn_large"][1]["preds"].pred.values[i]
        p2 = D["cnn_large"][2]["preds"].pred.values[i]
        a.text(0, -0.05, f"round 01 said: {names[p1][:26]}", transform=a.transAxes, fontsize=8, va="top",
               color="#c0392b" if good else "#1a7f37")
        a.text(0, -0.15, f"round 02 says: {names[p2][:26]}", transform=a.transAxes, fontsize=8, va="top",
               color="#1a7f37" if good else "#c0392b")
    fig.suptitle(f"Test photos that BOTH CNNs got wrong in round 01 and right in round 02 ({len(fixed)} photos), "
                 f"and the reverse ({len(broke)} photos)", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.94, "8 fixed + 4 broken photos, spread evenly through each list (not hand-picked). "
             "The predictions shown are CNN large's.", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.subplots_adjust(hspace=0.38)
    savefig(fig, "x07_photos_fixed_and_broken.png")


# ------------------------------------------------------------------------------------------------------------
# 8. GIF: the race, round 01 vs round 02
# ------------------------------------------------------------------------------------------------------------
def gif_race(D):
    series = []
    for m in MODELS:
        name, c = MODELS[m]
        for rnd, ls in ((1, "--"), (2, "-")):
            h = D[m][rnd]["hist"].set_index("epoch")
            series.append((f"{name}, round 0{rnd}", c, ls, h.val_acc1.rolling(9, center=True, min_periods=1).mean(),
                           h.clean_acc1.dropna()))
    frames = list(range(4, 301, 4)) + [300] * 12
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for a, t, yl in zip(ax, ("Accuracy on NEW photos (validation)", "Accuracy on the TRAINING photos (memorizing)"),
                        (58, 105)):
        a.set_xlim(0, 300)
        a.set_ylim(0, yl)
        a.set_title(t, loc="left", fontsize=11)
        a.set_xlabel("epoch")
        a.set_ylabel("top-1, %")
        grid(a, "both")
    arts = []
    for lab, c, ls, v, cl in series:
        lw = 2.4 if ls == "-" else 1.5
        l1, = ax[0].plot([], [], color=c, lw=lw, ls=ls, label=lab)
        l2, = ax[1].plot([], [], color=c, lw=lw, ls=ls)
        arts.append((v, cl, l1, l2))
    ax[0].legend(frameon=False, fontsize=8.5, loc="lower right")
    ttl = fig.suptitle("", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    def update(e):
        for v, cl, l1, l2 in arts:
            vv, cc = v.loc[:e], cl.loc[:e]
            l1.set_data(vv.index, vv.values)
            l2.set_data(cc.index, cc.values)
        ttl.set_text(f"Round 01 (dashed) vs round 02 final (solid), epoch {e}/300")
        return []

    FuncAnimation(fig, update, frames=frames, blit=False).save(ANI / "x_race_round01_vs_round02.gif",
                                                               writer=PillowWriter(fps=12), dpi=80)
    plt.close(fig)
    print("wrote", ANI / "x_race_round01_vs_round02.gif")


def main():
    D = {m: {1: load(m, 1), 2: load(m, 2)} for m in MODELS}
    for m in MODELS:
        a, b = D[m][2], D[m][1]
        d, n10, n01, p = mcnemar(a, b)
        fact(f"{MODELS[m][0]} round02 vs round01: {d:+.2f}, {n10}/{n01}, p={p:.2g}; r1 {b['top1']:.2f} r2 {a['top1']:.2f}"
             f"; top5 {b['top5']:.2f}->{a['top5']:.2f}; f1 {b['f1']:.2f}->{a['f1']:.2f}; gap {b['gap']:.1f}->{a['gap']:.1f}"
             f"; ece {b['ece']:.2f}->{a['ece']:.2f}; val {b['val']:.2f}->{a['val']:.2f}; epochs {b['epochs']}->{a['epochs']}"
             f"; best ep {b['best_epoch']}->{a['best_epoch']}; min {b['min']:.1f}->{a['min']:.1f}")
        for rnd in (1, 2):
            h = D[m][rnd]["hist"]
            fact(f"{MODELS[m][0]} round0{rnd}: min val loss {h.val_loss.min():.2f} @ {int(h.val_loss.idxmin()) + 1}, "
                 f"last clean {h.clean_acc1.dropna().iloc[-1]:.1f}")
    for rnd in (1, 2):
        d, n10, n01, p = mcnemar(D["cnn_large"][rnd], D["cnn_small"][rnd])
        fact(f"round 0{rnd} large vs small: {d:+.2f}, {n10}/{n01}, p={p:.2g}")
    fig_headline(D)
    fig_photos()
    fig_steps(D)
    fig_curves(D)
    fig_calibration(D)
    fig_species(D)
    fig_fixed(D)
    gif_race(D)
    (HERE / "docs" / "facts.txt").write_text("\n".join(FACTS) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
