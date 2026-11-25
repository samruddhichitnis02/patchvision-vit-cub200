"""Figures, GIF and facts for Vision_Transformer/README.md: plain ViT round 01 (strong augmentation + Mixup/CutMix)
vs round 02 (auto augmentation, no Mixup/CutMix, regularization search, final run r8).

Reads only finished results: 01_strongaug_mixup_cutmix/results/base/seed42/ and
02_autoaug_nomix_regularization/results/vit_base/<run>/seed42/. Run from anywhere:
    python Vision_Transformer/docs/make_comparison.py
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

HERE = Path(__file__).resolve().parents[1]                       # Vision_Transformer/
ROOT = HERE.parent
R01 = HERE / "01_strongaug_mixup_cutmix" / "results" / "base" / "seed42"
R02 = HERE / "02_autoaug_nomix_regularization" / "results" / "vit_base"
FIG, ANI = HERE / "docs" / "figures", HERE / "docs" / "animations"
for d in (FIG, ANI):
    d.mkdir(parents=True, exist_ok=True)
CUB = ROOT / "CUB_200_2011"

INK, MUTED, GRID = "#222222", "#666666", "#e6e6e6"
C1, C2 = "#9e9e9e", "#c0392b"                                    # round 01 grey, round 02 red (as in the round READMEs)
GOOD, BAD = "#1a7f37", "#c0392b"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100,
                     "axes.edgecolor": "#999999", "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.titleweight": "bold", "axes.titlesize": 11.5})
FINAL = "r8_combined"
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


def load(rnd, run=FINAL):
    d = R01 if rnd == 1 else R02 / run / "seed42"
    m = json.loads((d / "test_metrics.json").read_text())
    r = m["row"]
    preds = pd.read_csv(d / "test_predictions.csv").sort_values("image_id").reset_index(drop=True)
    return {"round": rnd, "run": run, "best_epoch": r["best_epoch"], "epochs": r["epochs_run"],
            "val": r["val_acc1_at_best"], "gap": r["overfit_gap_clean_minus_val"], "top1": r["test_acc1"],
            "top5": r["test_acc5"], "f1": r["test_macro_f1"], "ece": r["test_ece"], "min": r["train_time_s"] / 60,
            "hist": pd.read_csv(d / "history.csv"), "preds": preds,
            "pc": pd.read_csv(d / "test_per_class.csv"), "rel": pd.read_csv(d / "test_reliability.csv")}


def mcnemar(a, b):
    """a, b = loaded runs. Returns (a minus b in points, only a right, only b right, exact two-sided p)."""
    ca = (a["preds"].pred == a["preds"].label).values
    cb = (b["preds"].pred == b["preds"].label).values
    n10, n01 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    n, k = n10 + n01, min(n10, n01)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return (ca.mean() - cb.mean()) * 100, n10, n01, p


def ptxt(p):
    return f"p = {p:.2g}" if p >= 1e-4 else "p < 0.0001"


# ------------------------------------------------------------------------------------------------------------
# 1. headline: round 01 vs round 02 final, five measures
# ------------------------------------------------------------------------------------------------------------
def fig_headline(D):
    meas = [("top1", "Test top-1, %", True), ("top5", "Test top-5, %", True), ("f1", "Test macro-F1, %", True),
            ("gap", "Memorizing gap, points", False), ("ece", "Calibration error (ECE), %", False)]
    fig, ax = plt.subplots(1, 5, figsize=(18, 4.4))
    for a, (k, t, up) in zip(ax, meas):
        for j, rnd in enumerate((1, 2)):
            v = D[rnd][k]
            a.bar(j, v, 0.65, color=C2 if rnd == 2 else C1, edgecolor="white")
            a.text(j, v * 1.02 + 0.2, f"{v:.1f}" if k in ("gap", "ece") else f"{v:.2f}", ha="center", fontsize=10,
                   fontweight="bold" if rnd == 2 else "normal", color=INK)
        a.set_xticks([0, 1])
        a.set_xticklabels(["round 01", "round 02\n(r8)"], fontsize=9)
        a.set_xlim(-0.6, 1.6)
        a.set_title(t + ("\n(higher is better)" if up else "\n(lower is better)"), fontsize=10, loc="left")
        a.set_ylim(0, max(D[1][k], D[2][k]) * 1.18)
        grid(a)
    header(fig, "Plain ViT: round 01 vs round 02, same 5,794 test photos",
           "Grey = round 01 (strong augmentation + Mixup/CutMix, early stopping, no EMA).  Red = round 02 final r8 "
           "(auto augmentation, no Mixup/CutMix, drop-path 0.4 + label smoothing 0.2 + weight decay 0.2, EMA, 300 "
           "epochs).", top=0.86)
    savefig(fig, "x01_round01_vs_round02.png")


# ------------------------------------------------------------------------------------------------------------
# 2. what one training photo looks like in each round (the project's real transforms)
# ------------------------------------------------------------------------------------------------------------
def fig_photos():
    import torch
    from PIL import Image
    from torchvision import tv_tensors
    from torchvision.transforms import v2 as T
    sys.path.insert(0, str(HERE))
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
    bmask = tv_tensors.Mask(torch.zeros(1, *bird.shape[-2:]))     # the bird-box mask is not used here
    r2 = [auto(bird, bmask)[0] for _ in range(6)]
    fig, ax = plt.subplots(2, 7, figsize=(19, 7.6))
    for r in (0, 1):
        ax[r, 0].imshow(show(T.Resize((224, 224))(bird)))
        ax[r, 0].set_title("original photo", fontsize=9.5)
    for i, t in enumerate(r1):
        ax[0, i + 1].imshow(show(t))
        ax[0, i + 1].set_title(f"strong view {i + 1}", fontsize=9.5)
    for j, (x, info) in enumerate(mixed):
        lam = info["lam"]
        ax[0, 5 + j].imshow(show(x))
        ax[0, 5 + j].set_title(f"+ {'Mixup' if info['type'] == 'mixup' else 'CutMix'} with a Blue Jay\n"
                               f"label {lam * 100:.0f} % bunting / {100 - lam * 100:.0f} % jay", fontsize=9,
                               color=C2)
    for i, t in enumerate(r2):
        ax[1, i + 1].imshow(show(t))
        ax[1, i + 1].set_title(f"auto view {i + 1}", fontsize=9.5)
    for a in ax.ravel():
        a.axis("off")
    fig.subplots_adjust(left=0.035, right=0.995, top=0.80, bottom=0.01, hspace=0.32, wspace=0.05)
    fig.text(0.012, 0.62, "ROUND 01", rotation=90, fontsize=13, fontweight="bold", color=INK, va="center")
    fig.text(0.012, 0.20, "ROUND 02", rotation=90, fontsize=13, fontweight="bold", color=INK, va="center")
    fig.suptitle("What the ViT is shown: the same training photo in each round", x=0.01, y=0.99, ha="left",
                 fontsize=14, fontweight="bold")
    fig.text(0.01, 0.925, "Round 01: crop + flip + rotate/shift/zoom + colour change + erase, ALL stacked on every "
             "photo, then every batch is mixed (Mixup or CutMix).  Round 02: crop + flip + ONE random TrivialAugment "
             "operation + erase, never mixed.", fontsize=9, color=MUTED, ha="left")
    savefig(fig, "x02_training_photos.png")


# ------------------------------------------------------------------------------------------------------------
# 3. where the gain came from: round 01 -> r0 (new recipe) -> r2 (drop-path 0.4) -> r8 (+ ls 0.2 + wd 0.2)
# ------------------------------------------------------------------------------------------------------------
STEPS = [(1, None, "round 01\nstrong aug +\nMixup/CutMix"), (2, "r0_reference", "r0\nauto aug,\nno mixing, EMA"),
         (2, "r2_droppath0.4", "r2\n+ drop-path\n0.4"), (2, "r8_combined", "r8\n+ label smoothing 0.2\n+ weight decay 0.2")]


def fig_steps(D):
    runs = [D[1] if rnd == 1 else load(2, run) for rnd, run, _ in STEPS]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.2))
    for a, (k, t) in zip(ax, (("top1", "Test top-1, %"), ("gap", "Memorizing gap, points"))):
        vals = [r[k] for r in runs]
        a.bar(range(4), vals, 0.6, color=[C1] + [C2] * 3, edgecolor="white")
        for i, v in enumerate(vals):
            a.text(i, v + 0.3, f"{v:.2f}" if k == "top1" else f"{v:.1f}", ha="center", fontsize=9.5,
                   fontweight="bold")
            if i:
                d = v - vals[i - 1]
                txt = f"{d:+.2f}"
                if k == "top1":
                    txt += "\n" + ptxt(mcnemar(runs[i], runs[i - 1])[3])
                a.annotate(txt, xy=(i - 0.5, max(vals[i - 1], v) + (2.0 if k == "top1" else 1.4)), ha="center",
                           fontsize=8.5, color=GOOD if (d > 0) == (k == "top1") else BAD)
        a.set_xticks(range(4))
        a.set_xticklabels([s[2] for s in STEPS], fontsize=8.5)
        a.set_ylim(min(vals) - (6 if k == "top1" else 4), max(vals) + (5 if k == "top1" else 3.5))
        a.set_title(t, loc="left")
        grid(a)
    for i in range(1, 4):
        _, n10, n01, p = mcnemar(runs[i], runs[i - 1])
        fact(f"step {STEPS[i - 1][1] or 'round01'} -> {STEPS[i][1]}: {runs[i]['top1'] - runs[i - 1]['top1']:+.2f}, "
             f"{n10}/{n01}, p={p:.2g}; gap {runs[i - 1]['gap']:.1f}->{runs[i]['gap']:.1f}")
    header(fig, "Where the round 02 change came from, step by step",
           "Each bar adds one thing to the bar on its left. Number between bars = change (green better, red worse); "
           "p = exact paired test on the same test photos. The y-axes do not start at zero.", top=0.88)
    savefig(fig, "x03_where_the_gain_came_from.png")


# ------------------------------------------------------------------------------------------------------------
# 4. training curves: round 01 (dashed) vs round 02 final (solid)
# ------------------------------------------------------------------------------------------------------------
def fig_curves(D):
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.8))
    for rnd, ls, lw, c in ((1, "--", 1.8, C1), (2, "-", 2.2, C2)):
        h = D[rnd]["hist"].set_index("epoch")
        lab = f"round 0{rnd}" + (" (r8)" if rnd == 2 else "")
        ax[0].plot(h.index, h.val_acc1.rolling(7, center=True, min_periods=1).mean(), color=c, ls=ls, lw=lw, label=lab)
        cl = h.clean_acc1.dropna()
        ax[1].plot(cl.index, cl.values, color=c, ls=ls, lw=lw, label=lab)
        gg = (cl - h.val_acc1.loc[cl.index]).dropna()
        ax[2].plot(gg.index, gg.values, color=c, ls=ls, lw=lw, label=lab)
        ax[3].plot(h.index, h.val_loss, color=c, ls=ls, lw=lw, label=lab)
        be = D[rnd]["best_epoch"]
        ax[0].plot(be, h.val_acc1.loc[be], "o", color=c, ms=7, mec="white", zorder=5)
    e = D[1]["epochs"]
    ax[0].axvline(e, color=C1, ls=":", lw=1)
    ax[0].text(e - 4, 8, f"round 01\nstopped early\n(epoch {e})", fontsize=8, color=MUTED, ha="right")
    for a, t, yl in zip(ax, ("Validation top-1 (new photos); dot = chosen epoch", "Top-1 on training photos, no augmentation",
                             "Memorizing gap (training minus validation)", "Validation loss"),
                        ("top-1, %", "top-1, %", "points", "loss")):
        a.set_title(t, loc="left", fontsize=10.5)
        a.set_xlabel("epoch")
        a.set_ylabel(yl)
        a.set_xlim(0, 300)
        grid(a, "both")
    ax[0].legend(frameon=False, fontsize=9, loc="upper left")
    ax[3].set_ylim(2, 5.5)
    header(fig, "How training went: round 01 (dashed) vs round 02 final r8 (solid)",
           "Round 01 used early stopping (patience 60); round 02 trains all 300 epochs. Validation top-1 smoothed over "
           "7 epochs; training-photo accuracy measured every 5 epochs on 600 photos.")
    savefig(fig, "x04_training_curves.png")
    for rnd in (1, 2):
        h = D[rnd]["hist"]
        cl = h.set_index("epoch").clean_acc1.dropna()
        fact(f"round0{rnd}: min val loss {h.val_loss.min():.3f} @ {int(h.loc[h.val_loss.idxmin(), 'epoch'])}, "
             f"last val loss {h.val_loss.iloc[-1]:.3f}, clean>=99 first @ {int(cl[cl >= 99].index.min())}, "
             f"last clean {cl.iloc[-1]:.1f}, last gap {cl.iloc[-1] - h.val_acc1.iloc[-1]:.1f}")


# ------------------------------------------------------------------------------------------------------------
# 5. calibration
# ------------------------------------------------------------------------------------------------------------
def fig_calibration(D):
    fig, a = plt.subplots(figsize=(7.5, 6.2))
    a.plot([0, 100], [0, 100], color="#999999", lw=1, ls=":")
    for rnd, ls, mk, c in ((1, "--", "o", C1), (2, "-", "s", C2)):
        r = D[rnd]
        rel = r["rel"][r["rel"].fraction > 0.003]
        conf = r["preds"].confidence.mean() * 100
        a.plot(rel.confidence * 100, rel.accuracy * 100, ls=ls, marker=mk, color=c, lw=2,
               label=f"round 0{rnd}: ECE {r['ece']:.1f} %, says {conf:.0f} % on average, right {r['top1']:.0f} %")
        fact(f"round 0{rnd}: mean confidence {conf:.1f}, accuracy {r['top1']:.2f}, ECE {r['ece']:.2f}")
    r0 = load(2, "r0_reference")
    rel = r0["rel"][r0["rel"].fraction > 0.003]
    a.plot(rel.confidence * 100, rel.accuracy * 100, ls=":", marker=".", color="#555555", lw=1.4,
           label=f"round 02 r0 (no mixing, old regularizers): ECE {r0['ece']:.1f} %")
    fact(f"r0: mean confidence {r0['preds'].confidence.mean() * 100:.1f}, ECE {r0['ece']:.2f}")
    a.set_xlim(0, 100)
    a.set_ylim(0, 100)
    a.set_xlabel("confidence the model gives, %")
    a.set_ylabel("how often it is right, %")
    a.text(70, 40, "below the line =\nover-confident", color=MUTED, fontsize=9)
    a.text(8, 80, "above the line =\nunder-confident", color=MUTED, fontsize=9)
    a.legend(frameon=True, fontsize=8, loc="lower right", framealpha=0.95, edgecolor="#dddddd")
    grid(a, "both")
    header(fig, "Are the confidence numbers honest?",
           "Dotted diagonal = perfectly honest. Removing Mixup/CutMix made the ViT over-confident (r0); "
           "label smoothing 0.2 in r8 brings it back.", top=0.89)
    savefig(fig, "x05_calibration.png")


# ------------------------------------------------------------------------------------------------------------
# 6. species
# ------------------------------------------------------------------------------------------------------------
def fig_species(D):
    fig, a = plt.subplots(figsize=(8, 7))
    f1, f2 = D[1]["pc"].f1.values, D[2]["pc"].f1.values
    names = D[1]["pc"].class_name.values
    d = f2 - f1
    up, dn, eq = int((d > 1e-9).sum()), int((d < -1e-9).sum()), int((abs(d) <= 1e-9).sum())
    a.scatter(f1, f2, s=24, c=np.where(d > 0, C2, "#b0b0b0"), edgecolor="white", lw=0.5)
    a.plot([0, 100], [0, 100], color="#999999", lw=1, ls=":")
    spots = [(3, 95), (3, 89), (3, 83), (58, 14), (58, 8), (58, 2)]
    order = list(np.argsort(d)[::-1][:3]) + list(np.argsort(d)[:3])
    for (tx, ty), i in zip(spots, order):
        a.annotate(f"{names[i]} ({d[i]:+.0f})", (f1[i], f2[i]), xytext=(tx, ty), textcoords="data", fontsize=8,
                   color=INK, arrowprops=dict(arrowstyle="-", color="#999999", lw=0.7))
    a.set_xlim(0, 100)
    a.set_ylim(0, 100)
    a.set_xlabel("round 01 F1 per species, %")
    a.set_ylabel("round 02 (r8) F1 per species, %")
    a.set_title(f"{up} species better, {dn} worse, {eq} same  (median {np.median(d):+.1f})", loc="left",
                fontsize=10.5)
    grid(a, "both")
    fact(f"species: up {up}, down {dn}, same {eq}, median {np.median(d):+.2f}; median F1 {np.median(f1):.1f}->"
         f"{np.median(f2):.1f}; best {[(names[i], round(d[i])) for i in order[:3]]}, "
         f"worst {[(names[i], round(d[i])) for i in order[3:]]}")
    header(fig, "Which species improved? Each dot is one of the 200 species",
           "Above the diagonal (red) = better in round 02. 30 test photos per species, so one species\ncan move by 10+ "
           "points by chance; the count of species that improved is the robust part.", top=0.86)
    savefig(fig, "x06_species.png")


# ------------------------------------------------------------------------------------------------------------
# 7. photos fixed and broken by round 02
# ------------------------------------------------------------------------------------------------------------
def fig_fixed(D):
    from PIL import Image
    test = pd.read_csv(ROOT / "DataPreprocessing_Results" / "test.csv").set_index("image_id")
    names = D[1]["pc"].set_index("class_id").class_name.to_dict()
    ok1 = (D[1]["preds"].pred == D[1]["preds"].label).values
    ok2 = (D[2]["preds"].pred == D[2]["preds"].label).values
    ids, lab = D[1]["preds"].image_id.values, D[1]["preds"].label.values
    fixed, broke = np.where(~ok1 & ok2)[0], np.where(ok1 & ~ok2)[0]
    fact(f"photos fixed {len(fixed)}, broken {len(broke)}, both right {int((ok1 & ok2).sum())}, "
         f"both wrong {int((~ok1 & ~ok2).sum())}")
    pick = list(fixed[np.linspace(0, len(fixed) - 1, 8).astype(int)]) + \
        list(broke[np.linspace(0, len(broke) - 1, 4).astype(int)])
    fig, axes = plt.subplots(2, 6, figsize=(19, 9.6))
    for n, (a, i) in enumerate(zip(axes.ravel(), pick)):
        fp = str(test.loc[ids[i], "filepath"]).replace("\\", "/").split("CUB_200_2011/", 1)[1]
        im = Image.open(CUB / fp).convert("RGB")
        w, h = im.size
        k = min(w, h)
        a.imshow(im.crop(((w - k) // 2, (h - k) // 2, (w + k) // 2, (h + k) // 2)).resize((300, 300)))
        a.axis("off")
        good = n < 8
        a.set_title(("FIXED  " if good else "BROKEN  ") + names[lab[i]][:24], fontsize=9, fontweight="bold",
                    color=GOOD if good else BAD)
        p1, p2 = D[1]["preds"].pred.values[i], D[2]["preds"].pred.values[i]
        a.text(0, -0.05, f"round 01 said: {names[p1][:26]}", transform=a.transAxes, fontsize=8, va="top",
               color=BAD if good else GOOD)
        a.text(0, -0.15, f"round 02 says: {names[p2][:26]}", transform=a.transAxes, fontsize=8, va="top",
               color=GOOD if good else BAD)
    fig.suptitle(f"Test photos the ViT got wrong in round 01 and right in round 02 ({len(fixed)} photos), "
                 f"and the reverse ({len(broke)} photos)", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.94, "8 fixed + 4 broken photos, spread evenly through each list (not hand-picked).",
             fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.subplots_adjust(hspace=0.38)
    savefig(fig, "x07_photos_fixed_and_broken.png")


# ------------------------------------------------------------------------------------------------------------
# 8. GIF: the race, round 01 vs round 02
# ------------------------------------------------------------------------------------------------------------
def gif_race(D):
    series = []
    for rnd, ls, c in ((1, "--", C1), (2, "-", C2)):
        h = D[rnd]["hist"].set_index("epoch")
        series.append((f"round 0{rnd}" + (" (r8)" if rnd == 2 else ""), c, ls,
                       h.val_acc1.rolling(9, center=True, min_periods=1).mean(), h.clean_acc1.dropna()))
    frames = list(range(4, 301, 4)) + [300] * 12
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for a, t, yl in zip(ax, ("Accuracy on NEW photos (validation)", "Accuracy on the TRAINING photos (memorizing)"),
                        (50, 105)):
        a.set_xlim(0, 300)
        a.set_ylim(0, yl)
        a.set_title(t, loc="left", fontsize=11)
        a.set_xlabel("epoch")
        a.set_ylabel("top-1, %")
        grid(a, "both")
    arts = []
    for lab, c, ls, v, cl in series:
        lw = 2.4 if ls == "-" else 1.8
        l1, = ax[0].plot([], [], color=c, lw=lw, ls=ls, label=lab)
        l2, = ax[1].plot([], [], color=c, lw=lw, ls=ls)
        arts.append((v, cl, l1, l2))
    ax[0].legend(frameon=False, fontsize=9, loc="lower right")
    ttl = fig.suptitle("", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    def update(e):
        for v, cl, l1, l2 in arts:
            vv, cc = v.loc[:e], cl.loc[:e]
            l1.set_data(vv.index, vv.values)
            l2.set_data(cc.index, cc.values)
        ttl.set_text(f"Plain ViT: round 01 (dashed) vs round 02 final (solid), epoch {e}/300")
        return []

    FuncAnimation(fig, update, frames=frames, blit=False).save(ANI / "x_race_round01_vs_round02.gif",
                                                               writer=PillowWriter(fps=12), dpi=80)
    plt.close(fig)
    print("wrote", ANI / "x_race_round01_vs_round02.gif")


def main():
    D = {1: load(1), 2: load(2)}
    a, b = D[2], D[1]
    d, n10, n01, p = mcnemar(a, b)
    fact(f"round02 vs round01: {d:+.2f}, {n10}/{n01}, p={p:.2g}; r1 {b['top1']:.2f} r2 {a['top1']:.2f}"
         f"; top5 {b['top5']:.2f}->{a['top5']:.2f}; f1 {b['f1']:.2f}->{a['f1']:.2f}; gap {b['gap']:.1f}->{a['gap']:.1f}"
         f"; ece {b['ece']:.2f}->{a['ece']:.2f}; val {b['val']:.2f}->{a['val']:.2f}; epochs {b['epochs']}->{a['epochs']}"
         f"; best ep {b['best_epoch']}->{a['best_epoch']}; min {b['min']:.1f}->{a['min']:.1f}")
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
