"""Figures, GIF and facts for ViT_PatchVisionX/README.md: PatchVision-X round 01 (strong augmentation + Mixup/CutMix;
`full` run and the E4 winner) vs round 02 (auto augmentation, no Mixup/CutMix, regularization search, final run r8).
The four training-time extras are on in every run of both rounds.

Reads only finished results: 01_strongaug_mixup_cutmix/results/{full,ovf5_dp0.3_autoaug}/seed42/ and
02_autoaug_nomix_regularization/results/patchvision_x/<run>/seed42/. Run from anywhere:
    python ViT_PatchVisionX/docs/make_comparison.py
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

HERE = Path(__file__).resolve().parents[1]                       # ViT_PatchVisionX/
ROOT = HERE.parent
R01 = HERE / "01_strongaug_mixup_cutmix" / "results"
R02 = HERE / "02_autoaug_nomix_regularization" / "results" / "patchvision_x"
FIG, ANI = HERE / "docs" / "figures", HERE / "docs" / "animations"
for d in (FIG, ANI):
    d.mkdir(parents=True, exist_ok=True)
CUB = ROOT / "CUB_200_2011"

INK, MUTED, GRID = "#222222", "#666666", "#e6e6e6"
C1, C2, CE4 = "#9e9e9e", "#c0392b", "#6a51a3"                    # round 01 `full` grey, round 02 red, E4 purple
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


def load(key):
    """key: 'full', 'e4' (round 01) or a round-02 run name."""
    d = {"full": R01 / "full" / "seed42", "e4": R01 / "ovf5_dp0.3_autoaug" / "seed42"}.get(key, R02 / key / "seed42")
    r = json.loads((d / "test_metrics.json").read_text())["row"]
    preds = pd.read_csv(d / "test_predictions.csv").sort_values("image_id").reset_index(drop=True)
    return {"key": key, "best_epoch": r["best_epoch"], "epochs": r["epochs_run"], "val": r["val_acc1_at_best"],
            "gap": r["overfit_gap_clean_minus_val"], "top1": r["test_acc1"], "top5": r["test_acc5"],
            "f1": r["test_macro_f1"], "ece": r["test_ece"], "attn": r["test_attn_in_box"],
            "min": r["train_time_s"] / 60, "spe": r["sec_per_epoch"], "hist": pd.read_csv(d / "history.csv"),
            "preds": preds, "pc": pd.read_csv(d / "test_per_class.csv"), "rel": pd.read_csv(d / "test_reliability.csv")}


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
# 1. headline: round 01 `full`, round 01 E4, round 02 r8, six measures
# ------------------------------------------------------------------------------------------------------------
def fig_headline(D):
    meas = [("top1", "Test top-1, %", True), ("top5", "Test top-5, %", True), ("f1", "Test macro-F1, %", True),
            ("gap", "Memorizing gap, points", False), ("ece", "Calibration error (ECE), %", False),
            ("attn", "Attention inside bird box", True)]
    cols = [("full", C1, "round 01\n`full`"), ("e4", CE4, "round 01\nE4"), (FINAL, C2, "round 02\nr8")]
    fig, ax = plt.subplots(1, 6, figsize=(20, 4.5))
    for a, (k, t, up) in zip(ax, meas):
        for j, (key, c, _) in enumerate(cols):
            v = D[key][k]
            a.bar(j, v, 0.65, color=c, edgecolor="white")
            lab = f"{v:.2f}" if k == "attn" else (f"{v:.1f}" if k in ("gap", "ece") else f"{v:.2f}")
            a.text(j, v * 1.02 + (0.005 if k == "attn" else 0.2), lab, ha="center", fontsize=9.5,
                   fontweight="bold" if key == FINAL else "normal", color=INK)
        a.set_xticks(range(3))
        a.set_xticklabels([c[2] for c in cols], fontsize=8.5)
        a.set_xlim(-0.6, 2.6)
        a.set_title(t + ("\n(higher is better)" if up else "\n(lower is better)"), fontsize=10, loc="left")
        a.set_ylim(0, max(D[c[0]][k] for c in cols) * 1.18)
        grid(a)
    header(fig, "PatchVision-X: round 01 vs round 02, same 5,794 test photos",
           "Grey = round 01 `full` (strong aug + Mixup/CutMix, early stopping).  Purple = round 01 E4 (+ drop-path 0.3 + "
           "auto aug, Mixup/CutMix kept, 300 epochs).  Red = round 02 final r8 (auto aug, NO Mixup/CutMix, drop-path 0.4 "
           "+ label smoothing 0.2 + weight decay 0.2, 300 epochs).  Extras on in all three.", top=0.86)
    savefig(fig, "x01_round01_vs_round02.png")


# ------------------------------------------------------------------------------------------------------------
# 2. what one training photo looks like in each round (the project's real data pipeline, box-guided crop included)
# ------------------------------------------------------------------------------------------------------------
def fig_photos():
    import torch
    from torchvision.transforms import v2 as T
    sys.path.insert(0, str(ROOT / "Vision_Transformer"))
    from patchvision.data import CUBDataset, MixupCutmix, build_transforms, load_bboxes
    mean = [0.4859701097011566, 0.49921685457229614, 0.43137678503990173]
    std = [0.22647330164909363, 0.2219720035791397, 0.2609000504016876]
    csv = ROOT / "DataPreprocessing_Results" / "train.csv"
    bb = load_bboxes(CUB / "bounding_boxes.txt")
    train = pd.read_csv(csv)
    ib = int(np.where(train.class_name.values == "Painted Bunting")[0][0])
    ij = int(np.where(train.class_name.values == "Blue Jay")[0][0])

    def ds(tier):
        return CUBDataset(csv, build_transforms(tier, 224, mean, std, True), bb, CUB, train=True, box_crop_prob=0.3)

    def show(a, t, m, title, col=INK):
        s, mu = torch.tensor(std)[:, None, None], torch.tensor(mean)[:, None, None]
        a.imshow((t.float() * s + mu).clamp(0, 1).permute(1, 2, 0).numpy())
        a.contour(m[0].numpy(), levels=[0.5], colors="#2ecc71", linewidths=1.4)
        a.set_title(title, fontsize=9, color=col)

    plain = CUBDataset(csv, build_transforms("strong", 224, mean, std, False), bb, CUB)
    strong, auto = ds("strong"), ds("auto")
    torch.manual_seed(4)
    random.seed(4)
    np.random.seed(4)
    r1 = [strong[ib] for _ in range(4)]
    xb, mb, _, _ = strong[ib]
    xj, mj, _, _ = strong[ij]
    mixed = []
    for pc in (0.0, 1.0):                                    # one Mixup and one CutMix example, as in training
        np.random.seed(11 if pc == 0 else 5)
        x, _, m, info = MixupCutmix(200, 0.2, 1.0, pc, 0.1)(torch.stack([xb, xj]), torch.tensor([0, 1]),
                                                             torch.stack([mb, mj]))
        mixed.append((x[0], m[0], info))
    r2 = [auto[ib] for _ in range(6)]
    fig, ax = plt.subplots(2, 7, figsize=(19, 7.6))
    x0, m0, _, _ = plain[ib]
    for r in (0, 1):
        show(ax[r, 0], x0, m0, "original photo")
    for i, (x, m, _, _) in enumerate(r1):
        show(ax[0, i + 1], x, m, f"strong view {i + 1}")
    for j, (x, m, info) in enumerate(mixed):
        lam = info["lam"]
        show(ax[0, 5 + j], x, m, f"+ {'Mixup' if info['type'] == 'mixup' else 'CutMix'} with a Blue Jay\n"
             f"label {lam * 100:.0f} % bunting / {100 - lam * 100:.0f} % jay", col=C2)
    for i, (x, m, _, _) in enumerate(r2):
        show(ax[1, i + 1], x, m, f"auto view {i + 1}")
    for a in ax.ravel():
        a.axis("off")
    fig.subplots_adjust(left=0.035, right=0.995, top=0.80, bottom=0.01, hspace=0.32, wspace=0.05)
    fig.text(0.012, 0.62, "ROUND 01", rotation=90, fontsize=13, fontweight="bold", color=INK, va="center")
    fig.text(0.012, 0.20, "ROUND 02", rotation=90, fontsize=13, fontweight="bold", color=INK, va="center")
    fig.suptitle("What PatchVision-X is shown: the same training photo in each round", x=0.01, y=0.99, ha="left",
                 fontsize=14, fontweight="bold")
    fig.text(0.01, 0.925, "Both rounds: 30 % of photos are first cropped to the bird box (green outline = the box, used "
             "only by the bird-box loss).  Round 01: crop + flip + rotate/shift/zoom + colour + erase, ALL stacked, then "
             "every batch mixed.  Round 02: crop + flip + ONE TrivialAugment operation + erase, never mixed.",
             fontsize=9, color=MUTED, ha="left")
    savefig(fig, "x02_training_photos.png")


# ------------------------------------------------------------------------------------------------------------
# 3. where the change came from: full -> r0 -> r3/r2 -> r8, with E4 as a reference line
# ------------------------------------------------------------------------------------------------------------
STEPS = [("full", "round 01 `full`\nstrong aug +\nMixup/CutMix"), ("r0_reference", "r0\nauto aug,\nno mixing, 300 ep"),
         ("r2_droppath0.4", "r2\n+ drop-path\n0.4"), ("r8_combined", "r8\n+ label smoothing 0.2\n+ weight decay 0.2")]


def fig_steps(D):
    runs = [D[k] if k in D else load(k) for k, _ in STEPS]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.4))
    for a, (k, t) in zip(ax, (("top1", "Test top-1, %"), ("gap", "Memorizing gap, points"))):
        vals = [r[k] for r in runs]
        a.bar(range(4), vals, 0.6, color=[C1] + [C2] * 3, edgecolor="white")
        e4 = D["e4"][k]
        a.axhline(e4, color=CE4, ls="--", lw=1.4, label=f"round 01 E4: {e4:.2f}" if k == "top1" else
                  f"round 01 E4: {e4:.1f}")
        a.legend(frameon=False, fontsize=8.5, loc="upper left")
        for i, v in enumerate(vals):
            a.text(i, v + 0.3, f"{v:.2f}" if k == "top1" else f"{v:.1f}", ha="center", fontsize=9.5,
                   fontweight="bold")
            if i:
                d = mcnemar(runs[i], runs[i - 1])[0] if k == "top1" else v - vals[i - 1]   # paired, same photos
                txt = f"{d:+.2f}"
                if k == "top1":
                    txt += "\n" + ptxt(mcnemar(runs[i], runs[i - 1])[3])
                a.annotate(txt, xy=(i - 0.5, max(vals[i - 1], v) + (1.0 if k == "top1" else 1.4)), ha="center",
                           fontsize=8.5, color=GOOD if (d > 0) == (k == "top1") else BAD)
        a.set_xticks(range(4))
        a.set_xticklabels([s[1] for s in STEPS], fontsize=8.5)
        lo, hi = min(vals + [e4]), max(vals + [e4])
        a.set_ylim(lo - (6 if k == "top1" else 4), hi + (5 if k == "top1" else 3.5))
        a.set_title(t, loc="left")
        grid(a)
    for i in range(1, 4):
        dd, n10, n01, p = mcnemar(runs[i], runs[i - 1])
        fact(f"step {STEPS[i - 1][0]} -> {STEPS[i][0]}: {dd:+.2f}, "
             f"{n10}/{n01}, p={p:.2g}; gap {runs[i - 1]['gap']:.1f}->{runs[i]['gap']:.1f}")
    header(fig, "Where the round 02 change came from, step by step",
           "Each bar adds one thing to the bar on its left. Number between bars = change (green better, red worse); "
           "p = exact paired test on the same test photos. Dashed purple = round 01 E4. The y-axes do not start at zero.",
           top=0.88)
    savefig(fig, "x03_where_the_change_came_from.png")


# ------------------------------------------------------------------------------------------------------------
# 4. training curves: round 01 `full` (dashed), E4 (dotted), round 02 r8 (solid)
# ------------------------------------------------------------------------------------------------------------
SER = [("full", "round 01 `full`", C1, "--", 1.8), ("e4", "round 01 E4", CE4, ":", 1.8),
       (FINAL, "round 02 (r8)", C2, "-", 2.3)]


def fig_curves(D):
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.8))
    for key, lab, c, ls, lw in SER:
        h = D[key]["hist"].set_index("epoch")
        ax[0].plot(h.index, h.val_acc1.rolling(7, center=True, min_periods=1).mean(), color=c, ls=ls, lw=lw, label=lab)
        cl = h.clean_acc1.dropna()
        ax[1].plot(cl.index, cl.values, color=c, ls=ls, lw=lw, label=lab)
        gg = (cl - h.val_acc1.loc[cl.index]).dropna()
        ax[2].plot(gg.index, gg.values, color=c, ls=ls, lw=lw, label=lab)
        ax[3].plot(h.index, h.val_attn_in_box.rolling(5, center=True, min_periods=1).mean(), color=c, ls=ls, lw=lw,
                   label=lab)
        be = D[key]["best_epoch"]
        ax[0].plot(be, h.val_acc1.loc[be], "o", color=c, ms=7, mec="white", zorder=5)
    e = D["full"]["epochs"]
    ax[0].axvline(e, color=C1, ls=":", lw=1)
    ax[0].text(e - 4, 8, f"`full`\nstopped early\n(epoch {e})", fontsize=8, color=MUTED, ha="right")
    for a, t, yl in zip(ax, ("Validation top-1 (new photos); dot = chosen epoch",
                             "Top-1 on training photos, no augmentation", "Memorizing gap (training minus validation)",
                             "Attention inside the bird box (validation)"),
                        ("top-1, %", "top-1, %", "points", "share of attention")):
        a.set_title(t, loc="left", fontsize=10.5)
        a.set_xlabel("epoch")
        a.set_ylabel(yl)
        a.set_xlim(0, 300)
        grid(a, "both")
    ax[0].legend(frameon=False, fontsize=9, loc="upper left")
    ax[3].set_ylim(0.3, 0.75)
    header(fig, "How training went: round 01 `full` (dashed), round 01 E4 (dotted), round 02 final r8 (solid)",
           "`full` used early stopping (patience 60); E4 and round 02 train all 300 epochs. Validation top-1 smoothed over "
           "7 epochs; training-photo accuracy measured every 5 epochs on 600 photos.")
    savefig(fig, "x04_training_curves.png")
    for key, *_ in SER:
        h = D[key]["hist"]
        cl = h.set_index("epoch").clean_acc1.dropna()
        fact(f"{key}: min val loss {h.val_loss.min():.3f} @ {int(h.loc[h.val_loss.idxmin(), 'epoch'])}, "
             f"last val loss {h.val_loss.iloc[-1]:.3f}, clean>=99 first @ {int(cl[cl >= 99].index.min())}, "
             f"last clean {cl.iloc[-1]:.1f}, last gap {cl.iloc[-1] - h.val_acc1.iloc[-1]:.1f}, "
             f"peak attn {h.val_attn_in_box.max():.3f} @ {int(h.loc[h.val_attn_in_box.idxmax(), 'epoch'])}, "
             f"last attn {h.val_attn_in_box.iloc[-1]:.3f}")


# ------------------------------------------------------------------------------------------------------------
# 5. calibration
# ------------------------------------------------------------------------------------------------------------
def fig_calibration(D):
    fig, a = plt.subplots(figsize=(7.5, 6.2))
    a.plot([0, 100], [0, 100], color="#999999", lw=1, ls=":")
    r0 = load("r0_reference")
    for r, lab, ls, mk, c in ((D["full"], "round 01 `full`", "--", "o", C1), (D["e4"], "round 01 E4", ":", "^", CE4),
                              (r0, "round 02 r0", ":", ".", "#555555"), (D[FINAL], "round 02 r8", "-", "s", C2)):
        rel = r["rel"][r["rel"].fraction > 0.003]
        conf = r["preds"].confidence.mean() * 100
        a.plot(rel.confidence * 100, rel.accuracy * 100, ls=ls, marker=mk, color=c, lw=2 if c == C2 else 1.6,
               label=f"{lab}: ECE {r['ece']:.1f} %, says {conf:.0f} %, right {r['top1']:.0f} %")
        fact(f"{lab}: mean confidence {conf:.1f}, accuracy {r['top1']:.2f}, ECE {r['ece']:.2f}")
    a.set_xlim(0, 100)
    a.set_ylim(0, 100)
    a.set_xlabel("confidence the model gives, %")
    a.set_ylabel("how often it is right, %")
    a.text(70, 40, "below the line =\nover-confident", color=MUTED, fontsize=9)
    a.text(8, 80, "above the line =\nunder-confident", color=MUTED, fontsize=9)
    a.legend(frameon=True, fontsize=7.8, loc="lower right", framealpha=0.95, edgecolor="#dddddd")
    grid(a, "both")
    header(fig, "Are the confidence numbers honest?",
           "Dotted diagonal = perfectly honest. Every PatchVision-X run says about 42 % on average and is\n"
           "UNDER-confident: the more accurate the run, the further above the diagonal.", top=0.87)
    savefig(fig, "x05_calibration.png")


# ------------------------------------------------------------------------------------------------------------
# 6. species
# ------------------------------------------------------------------------------------------------------------
def fig_species(D):
    fig, a = plt.subplots(figsize=(8, 7))
    f1, f2 = D["full"]["pc"].f1.values, D[FINAL]["pc"].f1.values
    names = D["full"]["pc"].class_name.values
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
    a.set_xlabel("round 01 `full` F1 per species, %")
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
    names = D["full"]["pc"].set_index("class_id").class_name.to_dict()
    p1, p2 = D["full"]["preds"], D[FINAL]["preds"]
    ok1, ok2 = (p1.pred == p1.label).values, (p2.pred == p2.label).values
    ids, lab = p1.image_id.values, p1.label.values
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
        a.text(0, -0.05, f"round 01 said: {names[p1.pred.values[i]][:26]}", transform=a.transAxes, fontsize=8,
               va="top", color=BAD if good else GOOD)
        a.text(0, -0.15, f"round 02 says: {names[p2.pred.values[i]][:26]}", transform=a.transAxes, fontsize=8,
               va="top", color=GOOD if good else BAD)
    fig.suptitle(f"Test photos PatchVision-X got wrong in round 01 `full` and right in round 02 ({len(fixed)} photos), "
                 f"and the reverse ({len(broke)} photos)", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.94, "8 fixed + 4 broken photos, spread evenly through each list (not hand-picked).",
             fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.subplots_adjust(hspace=0.38)
    savefig(fig, "x07_photos_fixed_and_broken.png")


# ------------------------------------------------------------------------------------------------------------
# 8. GIF: the race
# ------------------------------------------------------------------------------------------------------------
def gif_race(D):
    series = []
    for key, lab, c, ls, lw in SER:
        h = D[key]["hist"].set_index("epoch")
        series.append((lab, c, ls, lw, h.val_acc1.rolling(9, center=True, min_periods=1).mean(), h.clean_acc1.dropna()))
    frames = list(range(4, 301, 4)) + [300] * 12
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for a, t, yl in zip(ax, ("Accuracy on NEW photos (validation)", "Accuracy on the TRAINING photos (memorizing)"),
                        (60, 105)):
        a.set_xlim(0, 300)
        a.set_ylim(0, yl)
        a.set_title(t, loc="left", fontsize=11)
        a.set_xlabel("epoch")
        a.set_ylabel("top-1, %")
        grid(a, "both")
    arts = []
    for lab, c, ls, lw, v, cl in series:
        l1, = ax[0].plot([], [], color=c, lw=lw + 0.3, ls=ls, label=lab)
        l2, = ax[1].plot([], [], color=c, lw=lw + 0.3, ls=ls)
        arts.append((v, cl, l1, l2))
    ax[0].legend(frameon=False, fontsize=9, loc="lower right")
    ttl = fig.suptitle("", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    def update(e):
        for v, cl, l1, l2 in arts:
            vv, cc = v.loc[:e], cl.loc[:e]
            l1.set_data(vv.index, vv.values)
            l2.set_data(cc.index, cc.values)
        ttl.set_text(f"PatchVision-X: round 01 `full` (dashed), E4 (dotted), round 02 final (solid), epoch {e}/300")
        return []

    FuncAnimation(fig, update, frames=frames, blit=False).save(ANI / "x_race_round01_vs_round02.gif",
                                                               writer=PillowWriter(fps=12), dpi=80)
    plt.close(fig)
    print("wrote", ANI / "x_race_round01_vs_round02.gif")


def main():
    D = {"full": load("full"), "e4": load("e4"), FINAL: load(FINAL)}
    for other in ("full", "e4"):
        a, b = D[FINAL], D[other]
        d, n10, n01, p = mcnemar(a, b)
        fact(f"r8 vs {other}: {d:+.2f}, {n10}/{n01}, p={p:.2g}; {b['top1']:.2f}->{a['top1']:.2f}; top5 {b['top5']:.2f}->"
             f"{a['top5']:.2f}; f1 {b['f1']:.2f}->{a['f1']:.2f}; gap {b['gap']:.1f}->{a['gap']:.1f}; ece {b['ece']:.2f}->"
             f"{a['ece']:.2f}; val {b['val']:.2f}->{a['val']:.2f}; attn {b['attn']:.3f}->{a['attn']:.3f}; epochs "
             f"{b['epochs']}->{a['epochs']}; best ep {b['best_epoch']}->{a['best_epoch']}; min {b['min']:.1f}->"
             f"{a['min']:.1f}; s/ep {b['spe']:.1f}->{a['spe']:.1f}")
    d, n10, n01, p = mcnemar(D["e4"], D["full"])
    fact(f"e4 vs full: {d:+.2f}, {n10}/{n01}, p={p:.2g}")
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
