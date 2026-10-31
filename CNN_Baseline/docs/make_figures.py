"""
Builds every figure and animation used by CNN_Baseline/README.md from the FINISHED CNN runs
(CNN_Baseline/results/cnn_small and cnn_large). Nothing is drawn by hand except the architecture sketch.
The ViT numbers used as a dashed reference line are read from ViT_PatchVisionX/results/full.

    python docs/make_figures.py              # everything     (run from inside CNN_Baseline/)
    python docs/make_figures.py bars ema     # only the named parts (see PARTS at the bottom)

Outputs: docs/figures/*.png   docs/animations/*.gif   docs/facts.txt (the numbers quoted in the README)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parents[1]            # CNN_Baseline/
ROOT = HERE.parent
FIG, ANI = HERE / "docs" / "figures", HERE / "docs" / "animations"
FIG.mkdir(parents=True, exist_ok=True)
ANI.mkdir(parents=True, exist_ok=True)
CUB = ROOT / "CUB_200_2011"
SEED = 42
sys.path.insert(0, str(HERE))

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100,
                     "axes.grid": True, "grid.alpha": 0.25})
C_SMALL, C_LARGE, C_VIT = "#ff7f0e", "#2ca02c", "#1f77b4"
FACTS = []


def fact(s):
    FACTS.append(s)
    print("  fact:", s)


def load(name, base=HERE / "results"):
    rd = base / name / f"seed{SEED}"
    return {"name": name, "dir": rd, "row": json.loads((rd / "test_metrics.json").read_text())["row"],
            "hist": pd.read_csv(rd / "history.csv"), "pc": pd.read_csv(rd / "test_per_class.csv"),
            "preds": pd.read_csv(rd / "test_predictions.csv"), "rel": pd.read_csv(rd / "test_reliability.csv"),
            "conf": np.load(rd / "test_confusion_matrix.npy")}


def runs():
    return load("cnn_small"), load("cnn_large")


def vit_full():
    return json.loads((ROOT / "ViT_PatchVisionX" / "results" / "full" / f"seed{SEED}" / "test_metrics.json").read_text())["row"]


def save(fig, name):
    fig.savefig(FIG / name, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  figure", name)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def gif(anim, name, fps):
    anim.save(ANI / name, writer=PillowWriter(fps=fps), dpi=80)
    print("  animation", name, f"{(ANI / name).stat().st_size / 1e6:.1f} MB")


# ----------------------------------------------------------------------------------------------
def part_bars():
    s, l = runs()
    v = vit_full()
    n = len(s["preds"])
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    names, cols = ["CNN small\n4.2 M params", "CNN large\n11.3 M params"], [C_SMALL, C_LARGE]
    a = ax[0]
    vals = [s["row"]["test_acc1"], l["row"]["test_acc1"]]
    err = np.array([[x - wilson(round(x / 100 * n), n)[0], wilson(round(x / 100 * n), n)[1] - x] for x in vals]).T
    bs = a.bar(names, vals, color=cols, edgecolor="black", yerr=err, capsize=7, width=0.55)
    for b, x in zip(bs, vals):
        a.text(b.get_x() + b.get_width() / 2, x / 2, f"{x:.1f}%", ha="center", color="white", fontsize=18, fontweight="bold")
    a.axhline(v["test_acc1"], color=C_VIT, ls="--", lw=2)
    a.text(1.42, v["test_acc1"] + 0.5, f"PatchVision-X (ViT) {v['test_acc1']:.1f}%", color=C_VIT, ha="right", fontweight="bold")
    a.set_ylim(0, 58)
    a.set_ylabel("test top-1 accuracy (%)")
    a.set_title(f"Top-1 on {n:,} unseen test photos\n(black bars: 95% sampling interval, about +/-1.3 points)")
    a = ax[1]
    w = 0.26
    metrics = [("test_acc1", "top-1"), ("test_acc5", "top-5"), ("test_macro_f1", "macro-F1")]
    x = np.arange(2)
    for i, (k, nm) in enumerate(metrics):
        vals = [s["row"][k], l["row"][k]]
        bs = a.bar(x + (i - 1) * w, vals, w, color=cols, edgecolor="black", hatch=[None, "//", "xx"][i], label=nm)
        for b, y in zip(bs, vals):
            a.text(b.get_x() + b.get_width() / 2, y + 0.8, f"{y:.1f}", ha="center", fontsize=9)
    a.set_xticks(x)
    a.set_xticklabels(names)
    a.set_ylim(0, 85)
    a.set_ylabel("%")
    a.set_title("Top-1 (plain), top-5 (//), macro-F1 (xx)")
    fig.suptitle("How well do the two from-scratch CNNs do?   (dashed line = the Vision Transformer, for reference)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "c01_results_bars.png")
    fact(f"small top1 {s['row']['test_acc1']:.2f} top5 {s['row']['test_acc5']:.2f} f1 {s['row']['test_macro_f1']:.2f} ece {s['row']['test_ece']:.2f} gap {s['row']['overfit_gap_clean_minus_val']:.1f}")
    fact(f"large top1 {l['row']['test_acc1']:.2f} top5 {l['row']['test_acc5']:.2f} f1 {l['row']['test_macro_f1']:.2f} ece {l['row']['test_ece']:.2f} gap {l['row']['overfit_gap_clean_minus_val']:.1f}")
    for r in (s, l):
        fact(f"{r['name']}: best epoch {r['row']['best_epoch']} of {r['row']['epochs_run']} run, val top-1 at best {r['row']['val_acc1_at_best']:.1f}, "
             f"train {r['row']['train_time_s'] / 60:.1f} min, {r['row']['sec_per_epoch']:.1f} s/epoch, params {r['row']['params_M']:.3f} M, "
             f"GFLOPs {r['row']['GFLOPs']:.2f}, latency bs1 {r['row']['latency_ms_bs1']:.2f} ms")


# ----------------------------------------------------------------------------------------------
def part_arch():
    from cnn.config import get_experiment
    from cnn.model import build_model
    fig, ax = plt.subplots(2, 1, figsize=(15, 8.6))
    for a, (name, col) in zip(ax, [("cnn_small", C_SMALL), ("cnn_large", C_LARGE)]):
        e = get_experiment(name)
        w = e.model.widths
        total = sum(p.numel() for p in build_model(e.model).parameters())
        a.set_xlim(0, 100); a.set_ylim(-1.5, 22); a.axis("off"); a.grid(False)
        a.set_title(f"{name}:  {total / 1e6:.2f} M parameters   (box height = number of channels, box width = image size)", fontsize=12, fontweight="bold", color=col, loc="left")
        stages = [("Photo", 224, 3), ("Stem\n7x7 conv /2\n+ max-pool", 56, w[0]), ("Stage 1\n2 blocks", 56, w[0]), ("Stage 2\n2 blocks", 28, w[1]),
                  ("Stage 3\n2 blocks", 14, w[2]), ("Stage 4\n2 blocks", 7, w[3]), ("Global avg\npool", 1, w[3]), ("Linear\n200 species", 1, 200)]
        x = 2.0
        prev = None
        for lab, size, ch in stages:
            bw = 3.0 + 7.5 * math.sqrt(size / 224) * (1 if size > 1 else 0.35)
            bh = 3.0 + 12.0 * math.sqrt(min(ch, 512) / 512)
            cy = 10.5
            a.add_patch(Rectangle((x, cy - bh / 2), bw, bh, fc=col, alpha=0.35 if lab not in ("Photo",) else 0.15, ec="black", lw=1.4))
            a.text(x + bw / 2, cy + bh / 2 + 0.9, lab, ha="center", va="bottom", fontsize=8.6)
            dim = f"{size}x{size}x{ch}" if size > 1 else (f"{ch}" if lab.startswith("Global") else "200")
            a.text(x + bw / 2, cy - bh / 2 - 0.9, dim, ha="center", va="top", fontsize=8.6, fontweight="bold")
            if prev is not None:
                a.annotate("", xy=(x - 0.2, cy), xytext=(prev + 0.2, cy), arrowprops=dict(arrowstyle="-|>", lw=1.3))
            prev = x + bw
            x += bw + 3.3
        a.text(2, -0.6, "Each block = 3x3 conv + BatchNorm + ReLU + 3x3 conv + BatchNorm, plus a shortcut that adds the block's input to its output",
               fontsize=9, style="italic")
    fig.tight_layout()
    save(fig, "c02_architecture.png")


def part_params():
    from cnn.config import get_experiment
    from cnn.model import build_model
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    labels = ["stem", "stage 1", "stage 2", "stage 3", "stage 4", "head"]
    allv = {}
    for name, col, off in [("cnn_small", C_SMALL, -0.2), ("cnn_large", C_LARGE, 0.2)]:
        m = build_model(get_experiment(name).model)
        vals = [sum(p.numel() for p in m.stem.parameters())] + [sum(p.numel() for p in m.stages[i].parameters()) for i in range(4)] + \
               [sum(p.numel() for p in m.head.parameters())]
        allv[name] = vals
        ax[0].bar(np.arange(6) + off, np.array(vals) / 1e6, 0.4, color=col, edgecolor="black", label=f"{name} ({sum(vals) / 1e6:.2f} M)")
    ax[0].set_xticks(range(6)); ax[0].set_xticklabels(labels)
    ax[0].set_ylabel("parameters (millions)")
    ax[0].set_title("Where the parameters are")
    ax[0].legend()
    for name, col in [("cnn_small", C_SMALL), ("cnn_large", C_LARGE)]:
        v = np.array(allv[name]) / sum(allv[name]) * 100
        ax[1].plot(labels, v, "o-", color=col, lw=2.2, label=name)
    ax[1].set_ylabel("share of all parameters (%)")
    ax[1].set_title("Same shape, different scale: most weights sit in the last two stages")
    ax[1].legend()
    fig.tight_layout()
    save(fig, "c03_params_by_stage.png")
    s = allv["cnn_small"]
    fact(f"cnn_small params by part (M): " + ", ".join(f"{n} {x / 1e6:.2f}" for n, x in zip(labels, s)))
    fact(f"cnn_small share in stages 3+4: {(s[3] + s[4]) / sum(s) * 100:.0f}%")


# ----------------------------------------------------------------------------------------------
def part_curves():
    s, l = runs()
    fig, ax = plt.subplots(2, 3, figsize=(19, 9))
    for r, col in ((s, C_SMALL), (l, C_LARGE)):
        h = r["hist"]
        e = h["epoch"]
        ax[0, 0].plot(e, h["train_loss"], color=col, lw=1.6, label=f"{r['name']} train", alpha=0.9)
        ax[0, 0].plot(e, h["val_loss"], color=col, lw=1.6, ls="--", label=f"{r['name']} validation")
        ax[0, 1].plot(e, h["val_acc1"].rolling(5, min_periods=1).mean(), color=col, lw=2, label=r["name"])
        be = int(r["row"]["best_epoch"])
        ax[0, 1].scatter([be], [h.loc[h["epoch"] == be, "val_acc1"].iloc[0]], color=col, s=70, edgecolor="black", zorder=4)
        c = h.dropna(subset=["clean_acc1"])
        ax[0, 2].plot(c["epoch"], c["clean_acc1"], color=col, lw=2, label=f"{r['name']}: training photos")
        ax[0, 2].plot(e, h["val_acc1"].rolling(5, min_periods=1).mean(), color=col, lw=2, ls="--", label=f"{r['name']}: new photos")
        ax[1, 0].plot(c["epoch"], c["clean_acc1"] - c["val_acc1"], color=col, lw=2, label=r["name"])
        ax[1, 1].plot(e, h["train_lr_first"], color=col, lw=2, label=r["name"])
        ax[1, 2].plot(e, h["val_macro_f1"].rolling(5, min_periods=1).mean(), color=col, lw=2, label=f"{r['name']} macro-F1")
        ax[1, 2].plot(e, h["val_acc5"].rolling(5, min_periods=1).mean(), color=col, lw=2, ls=":", label=f"{r['name']} top-5")
    titles = ["Loss (solid = training with augmentation, dashed = validation)", "Validation top-1, 5-epoch average (dot = epoch picked)",
              "Training photos (no augmentation) vs new photos", "Memorizing gap = training minus validation (points)",
              "Learning rate: 10 warm-up epochs, then cosine decay", "Validation macro-F1 and top-5"]
    for a, t in zip(ax.ravel(), titles):
        a.set_title(t, fontsize=10)
        a.set_xlabel("epoch")
        a.legend(fontsize=7.5)
    ax[0, 1].set_ylabel("%"); ax[0, 2].set_ylabel("top-1 (%)"); ax[1, 0].set_ylabel("points"); ax[1, 2].set_ylabel("%")
    fig.suptitle("Training curves of the two CNNs (seed 42)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "c04_training_curves.png")


def part_ema():
    s, l = runs()
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    for a, r, col in ((ax[0], s, C_SMALL), (ax[1], l, C_LARGE)):
        h = r["hist"]
        a.plot(h["epoch"], h["valraw_acc1"].rolling(3, min_periods=1).mean(), color="#999999", lw=1.6, label="latest weights (raw)")
        a.plot(h["epoch"], h["val_acc1"].rolling(3, min_periods=1).mean(), color=col, lw=2.2, label="EMA weights (what we evaluate)")
        sub = h[h["epoch"] >= 50]
        d = (sub["val_acc1"] - sub["valraw_acc1"]).mean()
        a.set_title(f"{r['name']}: EMA is {d:+.1f} points vs raw on average (epochs 50 to {int(h['epoch'].max())})")
        a.set_xlabel("epoch"); a.set_ylabel("validation top-1 (%)"); a.legend()
        fact(f"{r['name']}: EMA minus raw validation top-1, mean over epochs>=50: {d:+.2f} points")
    fig.suptitle("EMA weights: a slowly moving average of the weights is steadier than the latest weights", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "c05_ema_effect.png")


# ----------------------------------------------------------------------------------------------
def part_species():
    s, l = runs()
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.2))
    pc = s["pc"].sort_values("f1")
    hard, easy = pc.head(12), pc.tail(12)
    ax[0].barh([f"{c}" for c in hard["class_name"]][::-1], hard["f1"][::-1], color="#d62728", edgecolor="black")
    ax[0].set_title("12 hardest species for CNN small (F1, %)")
    ax[1].barh([f"{c}" for c in easy["class_name"]], easy["f1"], color="#2ca02c", edgecolor="black")
    ax[1].set_title("12 easiest species for CNN small (F1, %)")
    for a in ax:
        a.set_xlabel("F1 score (%)")
    fig.tight_layout()
    save(fig, "c06_species_best_worst.png")
    fact("small hardest 5: " + ", ".join(f"{r.class_name} ({r.f1:.0f})" for r in pc.head(5).itertuples()))
    fact("small easiest 5: " + ", ".join(f"{r.class_name} ({r.f1:.0f})" for r in pc.tail(5)[::-1].itertuples()))
    pcl = l["pc"].sort_values("f1")
    fact("large hardest 5: " + ", ".join(f"{r.class_name} ({r.f1:.0f})" for r in pcl.head(5).itertuples()))
    fact("large easiest 5: " + ", ".join(f"{r.class_name} ({r.f1:.0f})" for r in pcl.tail(5)[::-1].itertuples()))
    a_, b_ = s["pc"].set_index("class_name")["f1"], l["pc"].set_index("class_name")["f1"]
    d = (b_ - a_)
    fig, a = plt.subplots(figsize=(6.8, 6.2))
    a.scatter(a_, b_, s=14, alpha=0.7, color=C_LARGE)
    a.plot([0, 100], [0, 100], "k--", lw=1)
    for name, r in pd.concat([d.nlargest(4), d.nsmallest(4)]).items():
        a.annotate(str(name)[:20], (a_[name], b_[name]), fontsize=6.5)
    a.set_xlabel("CNN small: F1 per species"); a.set_ylabel("CNN large: F1 per species")
    a.set_title(f"CNN large better on {int((d > 0).sum())} species, CNN small on {int((d < 0).sum())} (of {len(d)})")
    fig.tight_layout()
    save(fig, "c07_small_vs_large_species.png")
    fact(f"large better on {int((d > 0).sum())} species, small better on {int((d < 0).sum())}, tied {int((d == 0).sum())}")


def part_confusions():
    s, l = runs()
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.2))
    for a, r, col in ((ax[0], s, C_SMALL), (ax[1], l, C_LARGE)):
        conf = r["conf"].copy()
        np.fill_diagonal(conf, 0)
        names = r["pc"]["class_name"].tolist()
        top = np.dstack(np.unravel_index(np.argsort(-conf.ravel())[:10], conf.shape))[0]
        labs = [f"{names[i]}  ->  {names[j]}" for i, j in top]
        vals = [int(conf[i, j]) for i, j in top]
        a.barh(labs[::-1], vals[::-1], color=col, edgecolor="black")
        a.set_title(f"{r['name']}: 10 most frequent mix-ups (true -> predicted)", fontsize=10)
        a.set_xlabel("number of test photos")
        a.tick_params(axis="y", labelsize=8)
        if r is s:
            fact("small top mix-ups: " + "; ".join(f"{names[i]} -> {names[j]} ({int(conf[i, j])}x)" for i, j in top[:5]))
        else:
            fact("large top mix-ups: " + "; ".join(f"{names[i]} -> {names[j]} ({int(conf[i, j])}x)" for i, j in top[:5]))
    fig.tight_layout()
    save(fig, "c08_confusions.png")


def part_calibration():
    s, l = runs()
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    a = ax[0]
    a.plot([0, 1], [0, 1], "k--", lw=1.2, label="perfect calibration")
    for r, col in ((s, C_SMALL), (l, C_LARGE)):
        a.plot(r["rel"]["confidence"], r["rel"]["accuracy"], "o-", color=col, lw=2, label=f"{r['name']} (ECE {r['row']['test_ece']:.1f}%)")
    a.set_xlabel("what the model says (confidence)"); a.set_ylabel("how often it is right (accuracy)")
    a.set_title("Reliability: below the dashed line = over-confident")
    a.legend()
    a = ax[1]
    for r, col, off in ((s, C_SMALL, -0.2), (l, C_LARGE, 0.2)):
        a.bar(np.arange(len(r["rel"])) + off, r["rel"]["fraction"] * 100, 0.4, color=col, edgecolor="black", label=r["name"])
    a.set_xlabel("confidence bin (low to high)"); a.set_ylabel("share of test photos (%)")
    a.set_title("How many photos fall in each confidence bin")
    a.legend()
    fig.tight_layout()
    save(fig, "c09_calibration.png")


# ----------------------------------------------------------------------------------------------
def _setup_samples():
    s, l = runs()
    test = pd.read_csv(ROOT / "DataPreprocessing_Results" / "test.csv").set_index("image_id")
    ids = s["preds"]["image_id"].values
    lab = s["preds"]["label"].values
    pr = {"CNN small": s["preds"].set_index("image_id").loc[ids, "pred"].values,
          "CNN large": l["preds"].set_index("image_id").loc[ids, "pred"].values}
    names = s["pc"].set_index("class_id")["class_name"].to_dict()
    return ids, lab, pr, names, test


def _img(test, iid, size=300):
    fp = str(test.loc[iid, "filepath"]).replace("\\", "/")
    im = Image.open(CUB / fp.split("CUB_200_2011/", 1)[1]).convert("RGB")
    im.thumbnail((size, size))
    return im


def part_samples():
    ids, lab, pr, names, test = _setup_samples()
    idx = np.linspace(0, len(ids) - 1, 12).astype(int)
    fig, axes = plt.subplots(2, 6, figsize=(19, 8.4))
    for a, i in zip(axes.ravel(), idx):
        a.axis("off")
        a.imshow(_img(test, ids[i]))
        a.set_title("truth: " + names[lab[i]][:24], fontsize=8.5, fontweight="bold")
        for j, k in enumerate(pr):
            ok = pr[k][i] == lab[i]
            a.text(0, -0.05 - 0.10 * j, ("OK  " if ok else "X   ") + k + ": " + names[pr[k][i]][:20], transform=a.transAxes, fontsize=8,
                   color="#1a7f37" if ok else "#c0392b", va="top", fontweight="bold")
    fig.suptitle("12 test photos spread evenly through the test set (not hand-picked): green = correct, red = wrong", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "c10_samples.png")
    ok_s, ok_l = pr["CNN small"][idx] == lab[idx], pr["CNN large"][idx] == lab[idx]
    fact(f"of these 12 photos: small right on {int(ok_s.sum())}, large right on {int(ok_l.sum())}")
    both_wrong = np.where((pr["CNN small"] != lab) & (pr["CNN large"] != lab))[0]
    both_right = np.where((pr["CNN small"] == lab) & (pr["CNN large"] == lab))[0]
    fact(f"both CNNs right on {len(both_right)} photos, both wrong on {len(both_wrong)}, only small right {int(((pr['CNN small'] == lab) & (pr['CNN large'] != lab)).sum())}, only large right {int(((pr['CNN small'] != lab) & (pr['CNN large'] == lab)).sum())} (of {len(ids)})")


# ----------------------------------------------------------------------------------------------
def anim_race():
    s, l = runs()
    maxe = max(int(s["hist"]["epoch"].max()), int(l["hist"]["epoch"].max()))
    frames = list(range(2, maxe + 6, 4))
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for a, t, yl in zip(ax, ("Accuracy on NEW photos (validation)", "Accuracy on the TRAINING photos (memorizing)"), (55, 105)):
        a.set_xlim(0, maxe + 5); a.set_ylim(0, yl); a.set_title(t); a.set_xlabel("epoch"); a.set_ylabel("top-1 (%)")
    ser, lines = [], []
    for r, col in ((s, C_SMALL), (l, C_LARGE)):
        h = r["hist"].set_index("epoch")
        ser.append((h["val_acc1"].rolling(5, min_periods=1).mean(), h["clean_acc1"].dropna()))
        l1, = ax[0].plot([], [], color=col, lw=2.6, label=f"{r['name']} ({r['row']['params_M']:.1f} M)")
        d1 = ax[0].scatter([], [], color=col, s=55, zorder=4)
        l2, = ax[1].plot([], [], color=col, lw=2.6)
        lines.append((l1, d1, l2))
    ax[0].legend(loc="lower right")
    ttl = fig.suptitle("")

    def draw(e):
        for (v, c), (l1, d1, l2) in zip(ser, lines):
            vv = v[v.index <= e]
            l1.set_data(vv.index, vv.values)
            d1.set_offsets([[vv.index[-1], vv.values[-1]]] if len(vv) else np.empty((0, 2)))
            cc = c[c.index <= e]
            l2.set_data(cc.index, cc.values)
        ttl.set_text(f"CNN training race, epoch {min(e, maxe)}: both learn the training photos, the bigger one is a little better on new photos")
        return []

    gif(FuncAnimation(fig, draw, frames=frames, blit=False), "cnn_training_race.gif", 10)
    plt.close(fig)


def anim_gap():
    s, _ = runs()
    h = s["hist"].set_index("epoch")
    c = h["clean_acc1"].dropna()
    v = h["val_acc1"].rolling(5, min_periods=1).mean()
    maxe = int(h.index.max())
    frames = list(range(2, maxe + 6, 3))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, maxe + 3); ax.set_ylim(0, 105); ax.set_xlabel("epoch"); ax.set_ylabel("top-1 (%)")
    l1, = ax.plot([], [], color="#1f9d55", lw=2.6, label="training photos (it memorizes them)")
    l2, = ax.plot([], [], color=C_SMALL, lw=2.6, label="NEW photos (what we care about)")
    ax.legend(loc="lower right")
    state = {"fill": None}
    txt = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")

    def draw(e):
        cc, vv = c[c.index <= e], v[v.index <= e]
        l1.set_data(cc.index, cc.values)
        l2.set_data(vv.index, vv.values)
        if state["fill"] is not None:
            state["fill"].remove()
        common = cc.index.intersection(vv.index)
        state["fill"] = ax.fill_between(common, vv.loc[common].values, cc.loc[common].values, color="#e377c2", alpha=0.35)
        if len(common):
            txt.set_text(f"epoch {e}: memorizing gap = {cc.loc[common].values[-1] - vv.loc[common].values[-1]:.0f} points")
        return []

    gif(FuncAnimation(fig, draw, frames=frames, blit=False), "cnn_memorizing_gap.gif", 10)
    plt.close(fig)


def anim_reveal():
    ids, lab, pr, names, test = _setup_samples()
    idx = np.linspace(0, len(ids) - 1, 12).astype(int)
    cols = {"CNN small": C_SMALL, "CNN large": C_LARGE}
    fig = plt.figure(figsize=(10, 4.2))
    ax_i = fig.add_axes([0.02, 0.05, 0.40, 0.82])
    ax_t = fig.add_axes([0.45, 0.05, 0.53, 0.82])

    def draw(f):
        i = idx[f]
        ax_i.clear(); ax_t.clear()
        ax_i.imshow(_img(test, ids[i], 360)); ax_i.axis("off")
        ax_t.axis("off"); ax_t.set_xlim(0, 1); ax_t.set_ylim(0, 1)
        ax_t.text(0, 0.95, f"Photo {f + 1} of {len(idx)}", fontsize=10, color="#555", va="top")
        ax_t.text(0, 0.82, "Truth:  " + names[lab[i]], fontsize=13, fontweight="bold", va="top")
        for j, k in enumerate(pr):
            ok = pr[k][i] == lab[i]
            ax_t.text(0, 0.60 - 0.18 * j, f"{'OK ' if ok else 'X  '}{k}: {names[pr[k][i]]}", fontsize=12, va="top",
                      color="#1a7f37" if ok else "#c0392b", fontweight="bold")
        done = idx[: f + 1]
        ax_t.text(0, 0.06, "Running score  " + "   ".join(f"{k}: {int((pr[k][done] == lab[done]).sum())}/{len(done)}" for k in pr), fontsize=10)
        return []

    gif(FuncAnimation(fig, draw, frames=len(idx), blit=False), "cnn_prediction_reveal.gif", 0.8)
    plt.close(fig)


PARTS = {"bars": part_bars, "arch": part_arch, "params": part_params, "curves": part_curves, "ema": part_ema, "species": part_species,
         "confusions": part_confusions, "calibration": part_calibration, "samples": part_samples, "race": anim_race,
         "gap": anim_gap, "reveal": anim_reveal}

if __name__ == "__main__":
    for k in (sys.argv[1:] or list(PARTS)):
        print(f"[{k}]")
        PARTS[k]()
    if not sys.argv[1:]:                                    # only a full run knows every fact
        (HERE / "docs" / "facts.txt").write_text("\n".join(FACTS), encoding="utf-8")
    print("done ->", FIG.parent)
