"""
Builds every figure, animation and table used by the top-level README.md from the FINISHED run files of both studies
(ViT_PatchVisionX/results and CNN_Baseline/results).  Nothing is drawn by hand except the two architecture sketches.

    python assets/make_assets.py              # everything
    python assets/make_assets.py bars gap     # only the named parts (see PARTS at the bottom)

Outputs: assets/figures/*.png   assets/animations/*.gif   assets/tables/*.md
Run it again after new runs finish; missing runs are skipped.
"""
from __future__ import annotations

import json
import math
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIG, ANI, TAB = ROOT / "assets" / "figures", ROOT / "assets" / "animations", ROOT / "assets" / "tables"
for d in (FIG, ANI, TAB):
    d.mkdir(parents=True, exist_ok=True)
VIT, CNN = ROOT / "ViT_PatchVisionX" / "results", ROOT / "CNN_Baseline" / "results"
CUB = ROOT / "CUB_200_2011"
SEED = 42

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100,
                     "axes.grid": True, "grid.alpha": 0.25})

# label, run folder name, results root, colour
MAIN = [("ViT base", "base", VIT, "#8c8c8c"), ("PatchVision-X", "full", VIT, "#1f77b4"),
        ("CNN small", "cnn_small", CNN, "#ff7f0e"), ("CNN large", "cnn_large", CNN, "#2ca02c")]
WINNER = "PatchVision-X"
GOLD = "#d4a017"


# ----------------------------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------------------------
def load_run(label, name, base, color=None):
    rd = base / name / f"seed{SEED}"
    if not (rd / "test_metrics.json").exists():
        return None
    return {"label": label, "name": name, "dir": rd, "color": color,
            "row": json.loads((rd / "test_metrics.json").read_text())["row"],
            "hist": pd.read_csv(rd / "history.csv"), "preds": pd.read_csv(rd / "test_predictions.csv"),
            "pc": pd.read_csv(rd / "test_per_class.csv")}


def load_main():
    runs = [load_run(*m) for m in MAIN]
    return [r for r in runs if r is not None]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def savefig(fig, name):
    fig.savefig(FIG / name, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  figure", name)


def mcnemar_exact(a_correct: np.ndarray, b_correct: np.ndarray):
    """Paired exact (binomial) test on the photos where exactly one of the two models is right."""
    b = int((a_correct & ~b_correct).sum())      # A right, B wrong
    c = int((~a_correct & b_correct).sum())      # A wrong, B right
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return b, c, min(1.0, 2 * p)


# ----------------------------------------------------------------------------------------------
# 1. headline bars
# ----------------------------------------------------------------------------------------------
def part_bars():
    M = load_main()
    n_test = int(M[0]["preds"].shape[0])
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.4), gridspec_kw={"width_ratios": [1.15, 1]})
    # left: top-1 with 95% interval
    a = ax[0]
    x = np.arange(len(M))
    top1 = np.array([r["row"]["test_acc1"] for r in M])
    ci = [wilson(round(v / 100 * n_test), n_test) for v in top1]
    err = np.array([[v - lo, hi - v] for v, (lo, hi) in zip(top1, ci)]).T
    bars = a.bar(x, top1, color=[r["color"] for r in M], yerr=err, capsize=6, edgecolor="black", linewidth=0.8)
    for b, r, v in zip(bars, M, top1):
        if r["label"] == WINNER:
            b.set_edgecolor(GOLD)
            b.set_linewidth(3.5)
            a.text(b.get_x() + b.get_width() / 2, v + 4.2, "WINNER", ha="center", fontsize=11, color=GOLD, fontweight="bold")
        a.text(b.get_x() + b.get_width() / 2, v / 2, f"{v:.1f}%", ha="center", color="white", fontsize=14, fontweight="bold")
    a.set_xticks(x)
    a.set_xticklabels([f"{r['label']}\n{r['row']['params_M']:.1f} M params" for r in M])
    a.set_ylabel("test top-1 accuracy (%)")
    a.set_ylim(0, max(top1) + 10)
    a.set_title(f"Top-1 accuracy on {n_test:,} unseen test photos\n(black bars = 95% sampling interval, about +/-1.3 points)")
    # right: top-5 and macro-F1
    a = ax[1]
    w = 0.38
    for i, (key, name, hatch) in enumerate([("test_acc5", "top-5", None), ("test_macro_f1", "macro-F1", "//")]):
        vals = [r["row"][key] for r in M]
        bs = a.bar(x + (i - 0.5) * w, vals, w, color=[r["color"] for r in M], edgecolor="black", linewidth=0.6, hatch=hatch)
        for b, v in zip(bs, vals):
            a.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}", ha="center", fontsize=8)
    a.set_xticks(x)
    a.set_xticklabels([r["label"] for r in M])
    a.set_ylabel("%")
    a.set_ylim(0, 85)
    a.set_title("Top-5 (solid) and macro-F1 (hatched)")
    fig.suptitle("Accuracy: PatchVision-X is best on every accuracy metric (one seed)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "01_headline_accuracy.png")


# ----------------------------------------------------------------------------------------------
# 2. efficiency
# ----------------------------------------------------------------------------------------------
def part_efficiency():
    M = load_main()
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.8))
    names = [r["label"] for r in M]
    cols = [r["color"] for r in M]
    # accuracy vs parameters (bubble = GFLOPs)
    a = ax[0]
    for r in M:
        s = r["row"]
        a.scatter(s["params_M"], s["test_acc1"], s=s["GFLOPs"] * 380, color=r["color"], edgecolor="black", alpha=0.85, zorder=3)
        a.annotate(f"{r['label']}\n{s['GFLOPs']:.2f} GFLOPs", (s["params_M"], s["test_acc1"]), textcoords="offset points",
                   xytext=(10, -26 if r["label"] == "ViT base" else 10), fontsize=8)
    a.set_xlabel("parameters (millions)")
    a.set_ylabel("test top-1 (%)")
    a.set_xlim(3.5, 13.5)
    a.set_ylim(37, 50)
    a.set_title("Accuracy vs size\n(bubble area = compute per photo)")
    for k, (key, ttl, fmt, log) in enumerate([("latency_ms_bs1", "Inference time, 1 photo (ms)\nlower is better", "{:.1f}", True),
                                              ("train_min", "Training time (minutes)\nlower is better", "{:.0f}", False),
                                              ("sec_per_epoch", "Seconds per training epoch\nlower is better", "{:.1f}", False)], 1):
        a = ax[k]
        vals = [(r["row"]["train_time_s"] / 60 if key == "train_min" else r["row"][key]) for r in M]
        bs = a.bar(names, vals, color=cols, edgecolor="black")
        for b, v in zip(bs, vals):
            a.text(b.get_x() + b.get_width() / 2, v * (1.12 if log else 1.0) + (0 if log else max(vals) * 0.02), fmt.format(v), ha="center", fontsize=9)
        a.set_title(ttl)
        if log:
            a.set_yscale("log")
        a.tick_params(axis="x", rotation=18)
    fig.suptitle("Cost.  Parameters and GFLOPs are hardware-independent.  The three time charts are NOT like-for-like (see note)",
                 fontsize=13, fontweight="bold")
    gpu_tag = ["RTX 4090" if "ViT_PatchVisionX" in str(r["dir"]) else "RTX 5090" for r in M]
    for a in ax[1:]:
        a.set_xticks(range(len(M)))
        a.set_xticklabels([f"{r['label']}\n[{g}]" for r, g in zip(M, gpu_tag)], rotation=0, fontsize=8)
        top = max(p.get_height() for p in a.patches)
        a.set_ylim(a.get_ylim()[0], top * (2.2 if a.get_yscale() == "log" else 1.15))
    fig.text(0.5, -0.05, "ViT runs were trained and timed on an RTX 4090 (PyTorch 2.5.1), CNN runs on an RTX 5090 (PyTorch 2.11).  "
             "A 5090 is faster than a 4090, so the time gaps overstate the architecture difference.  A same-GPU benchmark is pending.",
             ha="center", fontsize=9, color="#a00000")
    fig.tight_layout()
    savefig(fig, "02_efficiency.png")


# ----------------------------------------------------------------------------------------------
# 3. learning curves
# ----------------------------------------------------------------------------------------------
def part_curves():
    M = load_main()
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    for r in M:
        h = r["hist"]
        ax[0].plot(h["epoch"], h["val_acc1"].rolling(5, min_periods=1).mean(), color=r["color"], lw=2, label=r["label"])
        be = int(r["row"]["best_epoch"])
        ax[0].scatter([be], [h.loc[h["epoch"] == be, "val_acc1"].iloc[0]], color=r["color"], s=60, zorder=4, edgecolor="black")
        c = h.dropna(subset=["clean_acc1"])
        ax[1].plot(c["epoch"], c["clean_acc1"], color=r["color"], lw=2, label=r["label"])
        ax[2].plot(c["epoch"], c["clean_acc1"] - c["val_acc1"], color=r["color"], lw=2, label=r["label"])
    ax[0].set_title("New photos (validation), 5-epoch average\ndots = epoch chosen as best")
    ax[1].set_title("Training photos, no augmentation\nall models reach about 100%: they memorize")
    ax[2].set_title("Memorizing gap = training minus validation\n(points, lower is better)")
    ax[0].set_ylabel("top-1 (%)")
    ax[1].set_ylabel("top-1 (%)")
    ax[2].set_ylabel("points")
    for a in ax:
        a.set_xlabel("epoch")
    ax[0].legend()
    fig.suptitle("Learning curves: everyone memorizes the 5,394 training photos, PatchVision-X generalizes best", fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "03_learning_curves.png")


# ----------------------------------------------------------------------------------------------
# 4. gap / calibration / attention
# ----------------------------------------------------------------------------------------------
def part_gap():
    M = load_main()
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for a, (key, ttl, fmt) in zip(ax[:2], [("overfit_gap_clean_minus_val", "Memorizing gap (points)\nlower is better", "{:.1f}"),
                                           ("test_ece", "Calibration error ECE (%)\nlower is better", "{:.1f}")]):
        vals = [r["row"][key] for r in M]
        bs = a.bar([r["label"] for r in M], vals, color=[r["color"] for r in M], edgecolor="black")
        for b, v in zip(bs, vals):
            a.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.015, fmt.format(v), ha="center")
        a.set_title(ttl)
        a.tick_params(axis="x", rotation=18)
    a = ax[2]
    vit = [r for r in M if r["label"].startswith(("ViT", "Patch"))]
    vals = [r["row"]["test_attn_in_box"] for r in vit]
    bs = a.bar([r["label"] for r in vit], vals, color=[r["color"] for r in vit], edgecolor="black")
    for b, v in zip(bs, vals):
        a.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center")
    a.axhline(0.34, color="red", ls="--", lw=1.4)
    a.text(1.45, 0.355, "chance (0.34)", color="red", ha="right", fontsize=9)
    a.set_ylim(0, 0.8)
    a.set_title("Share of attention inside the bird's box\n(ViTs only: a CNN has no attention map)")
    fig.tight_layout()
    savefig(fig, "04_gap_calibration_attention.png")


# ----------------------------------------------------------------------------------------------
# 5. scorecard
# ----------------------------------------------------------------------------------------------
def part_scorecard():
    M = load_main()
    cols = [("Top-1", lambda r: r["row"]["test_acc1"], True, "{:.1f}%"), ("Top-5", lambda r: r["row"]["test_acc5"], True, "{:.1f}%"),
            ("Macro-F1", lambda r: r["row"]["test_macro_f1"], True, "{:.1f}%"),
            ("Memorizing\ngap", lambda r: r["row"]["overfit_gap_clean_minus_val"], False, "{:.1f}"),
            ("Calibration\nECE", lambda r: r["row"]["test_ece"], False, "{:.1f}%"),
            ("Parameters", lambda r: r["row"]["params_M"], False, "{:.1f}M"), ("Compute\nGFLOPs", lambda r: r["row"]["GFLOPs"], False, "{:.2f}"),
            ("Inference\nms/photo\n(different GPUs)", lambda r: r["row"]["latency_ms_bs1"], False, "{:.1f}"),
            ("Training\nminutes\n(different GPUs)", lambda r: r["row"]["train_time_s"] / 60, False, "{:.0f}")]
    n, m = len(M), len(cols)
    RANKED = 7                      # the two time columns are shown but NOT ranked: ViT and CNN runs used different GPUs
    rank = np.zeros((n, m))
    for j, (_, f, hi, _) in enumerate(cols[:RANKED]):
        v = np.array([f(r) for r in M])
        order = np.argsort(-v if hi else v)
        for k, i in enumerate(order):
            rank[i, j] = k + 1
    fig, ax = plt.subplots(figsize=(15, 4.6))
    ax.grid(False)
    shown = rank.copy()
    shown[:, RANKED:] = (n + 1) / 2                 # neutral colour for the unranked columns
    ax.imshow(shown, cmap="RdYlGn_r", vmin=1, vmax=n, aspect="auto")
    for i, r in enumerate(M):
        for j, (_, f, _, fmt) in enumerate(cols):
            best = j < RANKED and rank[i, j] == 1
            ax.text(j, i, ("* " if best else "") + fmt.format(f(r)), ha="center", va="center",
                    fontweight="bold" if best else "normal", fontsize=11, color="#555555" if j >= RANKED else "black")
    ax.set_xticks(range(m))
    ax.set_xticklabels([c[0] for c in cols])
    ax.xaxis.tick_top()
    wins = (rank[:, :RANKED] == 1).sum(1)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{r['label']}   ({int(w)} of {RANKED} best)" for r, w in zip(M, wins)], fontsize=11)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axvline(4.5, color="white", lw=4)
    ax.axvline(6.5, color="white", lw=4)
    ax.set_title("Scorecard (green = best, * = best).  Quality | size and compute | time (grey: ViT on RTX 4090, CNN on RTX 5090, not ranked)",
                 pad=52, fontsize=11)
    fig.tight_layout()
    savefig(fig, "05_scorecard.png")


# ----------------------------------------------------------------------------------------------
# 6. per species
# ----------------------------------------------------------------------------------------------
def part_species():
    M = {r["label"]: r for r in load_main()}
    fig, ax = plt.subplots(1, 3, figsize=(19, 6))
    pairs = [("CNN small", "PatchVision-X"), ("CNN large", "PatchVision-X"), ("CNN small", "CNN large")]
    for a, (la, lb) in zip(ax, pairs):
        df = pd.DataFrame({"a": M[la]["pc"].set_index("class_name")["f1"], "b": M[lb]["pc"].set_index("class_name")["f1"]})
        df["d"] = df["b"] - df["a"]
        a.scatter(df["a"], df["b"], s=14, alpha=0.7, color=M[lb]["color"])
        a.plot([0, 100], [0, 100], "k--", lw=1)
        for name, r in pd.concat([df.nlargest(4, "d"), df.nsmallest(4, "d")]).iterrows():
            a.annotate(str(name)[:20], (r["a"], r["b"]), fontsize=6.5)
        a.set_xlabel(f"{la}: F1 per species")
        a.set_ylabel(f"{lb}: F1 per species")
        a.set_title(f"{lb} better on {int((df['d'] > 0).sum())} species,\n{la} on {int((df['d'] < 0).sum())}"
                    f" (of {len(df)})")
    fig.suptitle("Species by species: each dot is one bird species (above the dashed line = the y-axis model is better)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "06_per_species.png")
    # best / worst species and most confused pairs for the winner and the matched CNN
    out = []
    for lab in ("PatchVision-X", "CNN small"):
        pc = M[lab]["pc"].sort_values("f1")
        out.append(f"**{lab}: hardest 5 species (F1)**: " + ", ".join(f"{r.class_name} ({r.f1:.0f})" for r in pc.head(5).itertuples()))
        out.append(f"**{lab}: easiest 5 species (F1)**: " + ", ".join(f"{r.class_name} ({r.f1:.0f})" for r in pc.tail(5)[::-1].itertuples()))
        conf = np.load(M[lab]["dir"] / "test_confusion_matrix.npy").copy()
        np.fill_diagonal(conf, 0)
        names = M[lab]["pc"]["class_name"].tolist()
        top = np.dstack(np.unravel_index(np.argsort(-conf.ravel())[:5], conf.shape))[0]
        out.append(f"**{lab}: most frequent mix-ups**: " + "; ".join(f"{names[i]} -> {names[j]} ({int(conf[i, j])}x)" for i, j in top))
        out.append("")
    (TAB / "species_notes.md").write_text("\n\n".join(out), encoding="utf-8")


# ----------------------------------------------------------------------------------------------
# 7. significance (paired)
# ----------------------------------------------------------------------------------------------
def part_significance():
    M = {r["label"]: r for r in load_main()}
    base = M["PatchVision-X"]["preds"][["image_id", "label"]]
    cor = {k: v["preds"].set_index("image_id").loc[base["image_id"]].pipe(lambda d: (d["pred"] == d["label"]).values) for k, v in M.items()}
    rows = []
    for a, b in [("PatchVision-X", "CNN small"), ("PatchVision-X", "CNN large"), ("CNN large", "CNN small"), ("PatchVision-X", "ViT base")]:
        x, y, p = mcnemar_exact(cor[a], cor[b])
        diff = 100 * (cor[a].mean() - cor[b].mean())
        rows.append(f"| {a} vs {b} | {diff:+.2f} | {x} | {y} | {p:.4f} | {'yes' if p < 0.05 else 'no'} |")
    both = (cor["PatchVision-X"] & cor["CNN small"] & cor["CNN large"]).sum()
    none = (~cor["PatchVision-X"] & ~cor["CNN small"] & ~cor["CNN large"]).sum()
    md = ["| Comparison (A vs B) | A minus B, top-1 points | photos only A gets right | photos only B gets right | p-value (exact paired test) | significant at 5%? |",
          "|---|---|---|---|---|---|"] + rows
    md += ["", f"All three of PatchVision-X, CNN small and CNN large are right on {both} photos and all three are wrong on {none} photos "
               f"(of {len(base)})."]
    (TAB / "significance.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


# ----------------------------------------------------------------------------------------------
# 8. architectures
# ----------------------------------------------------------------------------------------------
def _box(ax, x, y, w, h, text, fc, ec="black", ls="-", fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, ls=ls, lw=1.4))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")


def _arrow(ax, x, y0, y1):
    ax.annotate("", xy=(x, y1), xytext=(x, y0), arrowprops=dict(arrowstyle="-|>", lw=1.4))


def part_architectures():
    fig, ax = plt.subplots(1, 2, figsize=(16, 9.6), gridspec_kw={"width_ratios": [1.15, 1]})
    # PatchVision-X
    a = ax[0]
    a.set_xlim(0, 10); a.set_ylim(0, 14); a.axis("off"); a.grid(False)
    a.set_title("PatchVision-X  (4.27 M parameters, 1.55 GFLOPs)", fontsize=13, fontweight="bold", color="#1f77b4")
    X = 3.6
    steps = [(13.0, "Photo 224 x 224 x 3", "#f2f2f2", 0.8),
             (11.6, "Conv tokenizer: 2 conv layers\n-> 196 tokens (14 x 14), 256 numbers each", "#dae8fc", 1.0),
             (9.6, "STAGE 1:  4 blocks on 196 tokens\neach block = LFE (depthwise conv, locality)\n+ multi-scale GRAPH attention\n(8 heads: 3x3, 5x5, 9x9, global)\n+ MLP", "#ffe6cc", 2.0),
             (7.35, "Graph pooling: merge similar tokens  196 -> 49", "#dae8fc", 0.9),
             (5.6, "STAGE 2:  4 blocks on 49 tokens (same block design)", "#ffe6cc", 1.0),
             (4.1, "Average the 49 tokens -> 1 vector", "#e1d5e7", 0.8),
             (2.9, "Linear layer -> 200 species scores", "#d5e8d4", 0.8)]
    prev = None
    for y, t, c, h in steps:
        _box(a, X, y, 6.3, h, t, c, fs=8.5)
        if prev is not None:
            _arrow(a, X, prev, y + h / 2)
        prev = y - h / 2
    _box(a, 8.55, 9.6, 2.7, 3.6, "TRAINING ONLY\n(not used at test time)\n\n- bird-box attention loss\n- flip-consistency loss\n- EMA weights\n- box-guided crop", "#fff2cc", ls="--", fs=8)
    a.text(X, 1.6, "global receptive field from layer 1, but only 5,394 photos to learn from", ha="center", fontsize=9, style="italic")
    # CNN
    a = ax[1]
    a.set_xlim(0, 10); a.set_ylim(0, 14); a.axis("off"); a.grid(False)
    a.set_title("CNN small  (4.22 M parameters, 1.40 GFLOPs)", fontsize=13, fontweight="bold", color="#ff7f0e")
    X = 5.0
    steps = [(13.0, "Photo 224 x 224 x 3", "#f2f2f2", 0.8),
             (11.6, "Stem: 7x7 conv /2 + max-pool  ->  56 x 56 x 39", "#dae8fc", 0.9),
             (10.0, "STAGE 1: 2 residual blocks, 39 channels, 56 x 56", "#ffe6cc", 0.9),
             (8.5, "STAGE 2: 2 residual blocks, 78 channels, 28 x 28", "#ffe6cc", 0.9),
             (7.0, "STAGE 3: 2 residual blocks, 156 channels, 14 x 14", "#ffe6cc", 0.9),
             (5.5, "STAGE 4: 2 residual blocks, 312 channels, 7 x 7", "#ffe6cc", 0.9),
             (4.1, "Global average pooling -> 1 vector", "#e1d5e7", 0.8),
             (2.9, "Linear layer -> 200 species scores", "#d5e8d4", 0.8)]
    prev = None
    for y, t, c, h in steps:
        _box(a, X, y, 8.0, h, t, c, fs=9.5)
        if prev is not None:
            _arrow(a, X, prev, y + h / 2)
        prev = y - h / 2
    a.text(X, 1.6, "built-in assumption: neighbouring pixels belong together (local, shared filters)", ha="center", fontsize=9, style="italic")
    a.text(X, 0.7, "same data, augmentation, mixup/CutMix, label smoothing, AdamW, EMA, early stopping as PatchVision-X", ha="center", fontsize=8)
    fig.tight_layout()
    savefig(fig, "07_architectures.png")


# ----------------------------------------------------------------------------------------------
# 9. sample predictions
# ----------------------------------------------------------------------------------------------
def _sample_setup():
    M = {r["label"]: r for r in load_main()}
    test = pd.read_csv(ROOT / "DataPreprocessing_Results" / "test.csv").set_index("image_id")
    ids = M["PatchVision-X"]["preds"]["image_id"].values
    pred = {k: M[k]["preds"].set_index("image_id").loc[ids, "pred"].values for k in M}
    label = M["PatchVision-X"]["preds"].set_index("image_id").loc[ids, "label"].values
    names = M["PatchVision-X"]["pc"].set_index("class_id")["class_name"].to_dict()
    return M, test, ids, pred, label, names


def _img(test, iid, size=300):
    fp = str(test.loc[iid, "filepath"]).replace("\\", "/")
    im = Image.open(CUB / fp.split("CUB_200_2011/", 1)[1]).convert("RGB")
    im.thumbnail((size, size))
    return im


def _grid(idx, title, fname, M, test, ids, pred, label, names, show=("CNN small", "CNN large", "PatchVision-X"), ncols=6):
    n = len(idx)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.9 * nrows))
    axes = np.atleast_2d(axes)
    for a in axes.ravel():
        a.axis("off")
    for a, i in zip(axes.ravel(), idx):
        a.imshow(_img(test, ids[i]))
        a.set_title("truth: " + names[label[i]][:24], fontsize=8.5, fontweight="bold")
        txt = []
        for k in show:
            ok = pred[k][i] == label[i]
            txt.append((("OK  " if ok else "X   ") + k + ": " + names[pred[k][i]][:20], "#1a7f37" if ok else "#c0392b"))
        for j, (t, c) in enumerate(txt):
            a.text(0.0, -0.06 - 0.09 * j, t, transform=a.transAxes, fontsize=8, color=c, va="top", fontweight="bold")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    savefig(fig, fname)


def part_samples():
    M, test, ids, pred, label, names = _sample_setup()
    n = len(ids)
    even = np.linspace(0, n - 1, 12).astype(int)
    _grid(even, "12 test photos spread evenly through the test set (not hand-picked): green = correct, red = wrong",
          "08_samples_even.png", M, test, ids, pred, label, names)
    ok = {k: pred[k] == label for k in pred}
    rng = np.random.RandomState(0)
    only_vit = np.where(ok["PatchVision-X"] & ~ok["CNN small"] & ~ok["CNN large"])[0]
    only_cnn = np.where(~ok["PatchVision-X"] & ok["CNN small"] & ok["CNN large"])[0]
    pick_v = np.sort(rng.choice(only_vit, 6, replace=False))
    pick_c = np.sort(rng.choice(only_cnn, 6, replace=False))
    _grid(pick_v, f"PatchVision-X right, BOTH CNNs wrong ({len(only_vit)} such photos; 6 drawn at random)",
          "09_samples_vit_wins.png", M, test, ids, pred, label, names)
    _grid(pick_c, f"Both CNNs right, PatchVision-X wrong ({len(only_cnn)} such photos; 6 drawn at random)",
          "10_samples_cnn_wins.png", M, test, ids, pred, label, names)
    (TAB / "disagreements.md").write_text(
        f"PatchVision-X is right and both CNNs are wrong on **{len(only_vit)}** test photos; both CNNs are right and PatchVision-X is wrong on "
        f"**{len(only_cnn)}** (of {n}).", encoding="utf-8")


# ----------------------------------------------------------------------------------------------
# 10. E4 regularization study
# ----------------------------------------------------------------------------------------------
E4_RUNS = [("full (reference)", "full", "#7f7f7f"), ("drop-path 0.3", "ovf1_droppath0.3", "#1f77b4"),
           ("weight decay 0.1", "ovf2_wd0.1", "#ff7f0e"), ("auto-augment", "ovf3_autoaug", "#2ca02c"),
           ("dropout 0.1", "ovf4_dropout0.1", "#9467bd"), ("drop-path 0.3 + auto-aug", "ovf5_dp0.3_autoaug", "#d62728"),
           ("all four together", "ovf6_all_four", "#8c564b")]


def _e4_runs():
    out = []
    for lab, name, col in E4_RUNS:
        r = load_run(lab, name, VIT, col)
        if r is not None:
            out.append(r)
    return out


def e4_table(R):
    rows = []
    for r in R:
        h, s = r["hist"], r["row"]
        sm = h["val_acc1"].rolling(10, min_periods=10).mean()
        rows.append({"run": r["label"], "epochs run": int(s["epochs_run"]), "best epoch": int(s["best_epoch"]),
                     "lowest val loss": h["val_loss"].min(), "val top-1 smoothed (%)": sm.max(),
                     "memorizing gap (pts)": s["overfit_gap_clean_minus_val"], "test top-1 (%)": s["test_acc1"],
                     "test top-5 (%)": s["test_acc5"], "test macro-F1 (%)": s["test_macro_f1"], "test ECE (%)": s["test_ece"]})
    return pd.DataFrame(rows)


def part_e4():
    R = _e4_runs()
    if len(R) < 2:
        print("  E4: not enough finished runs")
        return
    t = e4_table(R)
    t.round(2).to_csv(TAB / "e4_regularization.csv", index=False)
    md = ["| Run | Epochs run | Best epoch | Lowest val loss | Val top-1, smoothed (%) | Gap (pts) | Test top-1 (%) | Test top-5 (%) | Test macro-F1 (%) | Test ECE (%) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    best_val = t["val top-1 smoothed (%)"].idxmax()
    for i, r in t.iterrows():
        b = "**" if i == best_val else ""
        md.append(f"| {b}{r['run']}{b} | {r['epochs run']} | {r['best epoch']} | {r['lowest val loss']:.2f} | {b}{r['val top-1 smoothed (%)']:.1f}{b} | "
                  f"{r['memorizing gap (pts)']:.1f} | {r['test top-1 (%)']:.2f} | {r['test top-5 (%)']:.2f} | {r['test macro-F1 (%)']:.2f} | {r['test ECE (%)']:.2f} |")
    (TAB / "e4_regularization.md").write_text("\n".join(md), encoding="utf-8")
    fig, ax = plt.subplots(1, 4, figsize=(21, 5))
    labels = [r["label"] for r in R]
    cols = [r["color"] for r in R]
    for a, (key, ttl, fmt) in zip(ax[:3], [("val top-1 smoothed (%)", "Validation top-1, best 10-epoch average (%)\nTHIS picks the winner", "{:.1f}"),
                                           ("test top-1 (%)", "Test top-1 (%)\nfor the record only", "{:.1f}"),
                                           ("memorizing gap (pts)", "Memorizing gap (points)\nlower is better", "{:.1f}")]):
        vals = t[key].values
        bs = a.barh(labels, vals, color=cols, edgecolor="black")
        for b, v in zip(bs, vals):
            a.text(v + max(vals) * 0.005, b.get_y() + b.get_height() / 2, fmt.format(v), va="center", fontsize=9)
        a.invert_yaxis()
        a.set_title(ttl)
        lo = min(vals)
        a.set_xlim(lo - (max(vals) - lo) * 1.5 - 1 if key != "memorizing gap (pts)" else min(vals) - 3, max(vals) + (max(vals) - lo) * 0.45 + 0.6)
    for r in R:
        h = r["hist"]
        ax[3].plot(h["epoch"], h["val_acc1"].rolling(10, min_periods=1).mean(), color=r["color"], lw=1.8, label=r["label"])
    ax[3].set_title("Validation top-1 while training\n(10-epoch average)")
    ax[3].set_xlabel("epoch")
    ax[3].legend(fontsize=7)
    fig.suptitle("E4: regularization study (seed 42).  Axes of the bar charts do not start at zero: read the numbers.", fontsize=12, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "11_e4_regularization.png")


# ----------------------------------------------------------------------------------------------
# animations
# ----------------------------------------------------------------------------------------------
def _save_gif(anim, name, fps):
    anim.save(ANI / name, writer=PillowWriter(fps=fps), dpi=80)
    print("  animation", name, f"{(ANI / name).stat().st_size / 1e6:.1f} MB")


def anim_race():
    M = load_main()
    maxe = max(int(r["hist"]["epoch"].max()) for r in M)
    frames = list(range(2, maxe + 6, 4))
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    series = []
    for r in M:
        h = r["hist"].set_index("epoch")
        v = h["val_acc1"].rolling(5, min_periods=1).mean()
        c = h["clean_acc1"].dropna()
        series.append((r, v, c))
    for a, t, yl in zip(ax, ("Accuracy on NEW photos (validation)", "Accuracy on the TRAINING photos (memorizing)"), (60, 105)):
        a.set_xlim(0, maxe + 5)
        a.set_ylim(0, yl)
        a.set_title(t)
        a.set_xlabel("epoch")
        a.set_ylabel("top-1 (%)")
    lines = []
    for r, v, c in series:
        l1, = ax[0].plot([], [], color=r["color"], lw=2.4, label=r["label"])
        d1 = ax[0].scatter([], [], color=r["color"], s=50, zorder=4)
        l2, = ax[1].plot([], [], color=r["color"], lw=2.4)
        lines.append((l1, d1, l2))
    ax[0].legend(loc="lower right", fontsize=8)
    ttl = fig.suptitle("")

    def draw(e):
        for (r, v, c), (l1, d1, l2) in zip(series, lines):
            vv = v[v.index <= e]
            l1.set_data(vv.index, vv.values)
            d1.set_offsets([[vv.index[-1], vv.values[-1]]] if len(vv) else np.empty((0, 2)))
            cc = c[c.index <= e]
            l2.set_data(cc.index, cc.values)
        ttl.set_text(f"Training race, epoch {min(e, maxe)}: every model learns the training photos, only some generalize")
        return []

    _save_gif(FuncAnimation(fig, draw, frames=frames, blit=False), "training_race.gif", 10)
    plt.close(fig)


def anim_predictions():
    M, test, ids, pred, label, names = _sample_setup()
    idx = np.linspace(0, len(ids) - 1, 12).astype(int)
    show = ["CNN small", "CNN large", "PatchVision-X"]
    colors = {k: M[k]["color"] for k in show}
    fig = plt.figure(figsize=(10, 4.4))
    ax_i = fig.add_axes([0.02, 0.05, 0.40, 0.82])
    ax_t = fig.add_axes([0.45, 0.05, 0.53, 0.82])
    tally = {k: 0 for k in show}

    def draw(f):
        i = idx[f]
        ax_i.clear(); ax_t.clear()
        ax_i.imshow(_img(test, ids[i], 360)); ax_i.axis("off")
        ax_t.axis("off"); ax_t.set_xlim(0, 1); ax_t.set_ylim(0, 1)
        ax_t.text(0, 0.95, f"Photo {f + 1} of {len(idx)}", fontsize=10, color="#555", va="top")
        ax_t.text(0, 0.84, "Truth:  " + names[label[i]], fontsize=13, fontweight="bold", va="top")
        for j, k in enumerate(show):
            ok = pred[k][i] == label[i]
            ax_t.text(0, 0.66 - 0.17 * j, f"{'OK ' if ok else 'X  '}{k}: {names[pred[k][i]]}", fontsize=11.5, va="top",
                      color="#1a7f37" if ok else "#c0392b", fontweight="bold")
        done = idx[: f + 1]
        sc = "   ".join(f"{k}: {int((pred[k][done] == label[done]).sum())}/{len(done)}" for k in show)
        ax_t.text(0, 0.06, "Running score  " + sc, fontsize=9.5, color="#333")
        return []

    _save_gif(FuncAnimation(fig, draw, frames=len(idx), blit=False), "prediction_reveal.gif", 0.8)
    plt.close(fig)


def anim_e4():
    R = _e4_runs()
    if len(R) < 3:
        return
    maxe = max(int(r["hist"]["epoch"].max()) for r in R)
    frames = list(range(5, maxe + 8, 5))
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_xlim(0, maxe + 5)
    allv = np.concatenate([r["hist"]["val_acc1"].rolling(10, min_periods=1).mean().values for r in R])
    ax.set_ylim(20, allv.max() + 3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation top-1 (%), 10-epoch average")
    lines = [ax.plot([], [], color=r["color"], lw=2.2, label=r["label"])[0] for r in R]
    ax.legend(loc="lower right", fontsize=8)
    ttl = ax.set_title("")

    def draw(e):
        for r, l in zip(R, lines):
            v = r["hist"].set_index("epoch")["val_acc1"].rolling(10, min_periods=1).mean()
            v = v[v.index <= e]
            l.set_data(v.index, v.values)
        ttl.set_text(f"E4 regularization race: epoch {min(e, maxe)}")
        return []

    _save_gif(FuncAnimation(fig, draw, frames=frames, blit=False), "e4_regularization_race.gif", 8)
    plt.close(fig)


PARTS = {"bars": part_bars, "efficiency": part_efficiency, "curves": part_curves, "gap": part_gap, "scorecard": part_scorecard,
         "species": part_species, "significance": part_significance, "arch": part_architectures, "samples": part_samples,
         "e4": part_e4, "race": anim_race, "reveal": anim_predictions, "e4race": anim_e4}

if __name__ == "__main__":
    which = sys.argv[1:] or list(PARTS)
    for k in which:
        print(f"[{k}]")
        PARTS[k]()
    print("done ->", FIG.parent)

