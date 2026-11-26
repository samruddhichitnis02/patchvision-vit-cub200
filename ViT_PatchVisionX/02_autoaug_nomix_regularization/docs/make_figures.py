"""Figures, animations and tables for ViT_PatchVisionX/02_autoaug_nomix_regularization/README.md (PatchVision-X only).

    python docs/make_figures.py            (from inside ViT_PatchVisionX/02_autoaug_nomix_regularization/)

Reads the finished runs in results/patchvision_x/<run>/seed42/, round 01's `full` run and the E4 winner
(../01_strongaug_mixup_cutmix/results/{full,ovf5_dp0.3_autoaug}/seed42/), and, for the cross-model figure, the round-02
runs of the plain ViT and the CNNs. Writes docs/figures/*.png, docs/animations/*.gif, docs/tables/*.md and
docs/facts.txt.

Colour = which KIND of change a run makes; darker shade = stronger setting. Every run is also labelled by name.
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

HERE = Path(__file__).resolve().parents[1]                       # ViT_PatchVisionX/02_autoaug_nomix_regularization/
REPO = HERE.parents[1]
MODEL = "patchvision_x"
RES = HERE / "results" / MODEL
R01 = HERE.parent / "01_strongaug_mixup_cutmix" / "results" / "full" / "seed42"
E4 = HERE.parent / "01_strongaug_mixup_cutmix" / "results" / "ovf5_dp0.3_autoaug" / "seed42"
OTHERS = {"Plain ViT": REPO / "Vision_Transformer" / "02_autoaug_nomix_regularization" / "results" / "vit_base",
          "CNN small": REPO / "CNN_Baseline" / "02_autoaug_nomix_regularization" / "results" / "cnn_small",
          "CNN large": REPO / "CNN_Baseline" / "02_autoaug_nomix_regularization" / "results" / "cnn_large"}
FIG, ANI, TAB = HERE / "docs" / "figures", HERE / "docs" / "animations", HERE / "docs" / "tables"
for d in (FIG, ANI, TAB):
    d.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID = "#222222", "#666666", "#e6e6e6"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100,
                     "axes.edgecolor": "#999999", "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.titleweight": "bold", "axes.titlesize": 11.5})

RUNS = {
    "r0_reference": ("r0 reference", "reference", "#7f7f7f"),
    "r1_droppath0.2": ("r1 drop-path 0.2", "drop-path", "#9ecae1"),
    "r2_droppath0.4": ("r2 drop-path 0.4", "drop-path", "#4292c6"),
    "r3_droppath0.5": ("r3 drop-path 0.5", "drop-path", "#08519c"),
    "r4_labelsmooth0.2": ("r4 label smoothing 0.2", "label smoothing", "#8064a2"),
    "r5_wd0.1": ("r5 weight decay 0.1", "weight decay", "#74c476"),
    "r6_wd0.2": ("r6 weight decay 0.2", "weight decay", "#238b45"),
    "r7_sam": ("r7 SAM", "SAM", "#f28e2b"),
    "r8_combined": ("r8 combined (winners)", "combined", "#c0392b"),
}
WHAT = {"r0_reference": "nothing (new base recipe)", "r1_droppath0.2": "drop-path 0.1 → 0.2",
        "r2_droppath0.4": "drop-path 0.1 → 0.4", "r3_droppath0.5": "drop-path 0.1 → 0.5",
        "r4_labelsmooth0.2": "label smoothing 0.1 → 0.2", "r5_wd0.1": "weight decay 0.05 → 0.1",
        "r6_wd0.2": "weight decay 0.05 → 0.2", "r7_sam": "SAM optimizer (rho 0.05)",
        "r8_combined": "drop-path 0.4 + label smoothing 0.2 + weight decay 0.2"}
NAME = "PatchVision-X (4.27 M)"
R01_COL, E4_COL, BEST_COL = "#bdbdbd", "#6a51a3", "#c0392b"
FACTS: list[str] = []


def fact(s):
    FACTS.append(s)
    print(s)


def savefig(fig, name):
    fig.savefig(FIG / name, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", FIG / name)


def header(fig, title, sub, top=0.9):
    fig.suptitle(title, x=0.01, y=0.995, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.925, sub, fontsize=9, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0, 1, top))


def grid(a, axis="y"):
    a.grid(axis=axis, color=GRID, lw=0.8)
    a.set_axisbelow(True)


def run_dir(run):
    return RES / run / "seed42"


def load():
    rows, hist = [], {}
    for run in RUNS:
        d = run_dir(run)
        if not (d / "test_metrics.json").exists():
            continue
        m = json.loads((d / "test_metrics.json").read_text())
        r, cfg = m["row"], m["config"]
        rows.append({"run": run, "label": RUNS[run][0], "best_epoch": r["best_epoch"],
                     "val_top1": r["val_acc1_at_best"], "gap": r["overfit_gap_clean_minus_val"],
                     "test_top1": r["test_acc1"], "test_top5": r["test_acc5"], "test_f1": r["test_macro_f1"],
                     "test_ece": r["test_ece"], "attn": r["test_attn_in_box"], "train_min": r["train_time_s"] / 60,
                     "sec_epoch": r["sec_per_epoch"], "latency": r.get("latency_ms_bs1"),
                     "drop_path": cfg["model"]["drop_path"], "label_smoothing": cfg["train"]["label_smoothing"],
                     "weight_decay": cfg["train"]["weight_decay"], "sam": cfg["train"].get("use_sam", False)})
        hist[run] = pd.read_csv(d / "history.csv")
    return pd.DataFrame(rows), hist


def final_run(df):
    """Best validation top-1. r8 wins a tie (planned recipe); otherwise a tie goes to the EARLIER run (smaller change)."""
    best = df[df.val_top1 >= df.val_top1.max() - 1e-9]
    if "r8_combined" in best.run.values:
        return "r8_combined"
    return best["run"].iloc[0]


def correct(path):
    p = pd.read_csv(path).set_index("image_id")
    return p.pred == p.label


def mcnemar(ca, cb):
    cb = cb.loc[ca.index]
    n10, n01 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    n, k = n10 + n01, min(n10, n01)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return (ca.mean() - cb.mean()) * 100, n10, n01, p


def row01(path):
    return json.loads((path / "test_metrics.json").read_text())["row"]


# ------------------------------------------------------------------------------------------------------------
# 1. every run against r0: validation (decides) and test (record)
# ------------------------------------------------------------------------------------------------------------
def fig_overview(df):
    ref = df.set_index("run").loc["r0_reference"]
    best = final_run(df)
    o = row01(R01)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    order = list(RUNS)
    y = np.arange(len(order))[::-1]
    for a, col, rcol, title in ((ax[0], "val_top1", "val_acc1_at_best", "Validation top-1 (600 photos): DECIDES the winners"),
                                (ax[1], "test_top1", "test_acc1", "Test top-1 (5,794 photos): for the record")):
        a.axvline(ref[col], color=INK, lw=1, ls="--", zorder=1)
        a.axvline(o[rcol], color="#999999", lw=1.2, ls=":", zorder=1)
        a.text(o[rcol], y[0] + 0.62, "round 01 `full`", fontsize=8, color=MUTED, ha="center")
        for yi, run in zip(y, order):
            if run in df.run.values:
                v = df.set_index("run").loc[run, col]
                a.barh(yi, v, color=RUNS[run][2], height=0.66, edgecolor="white", linewidth=2, zorder=2)
                dd = v - ref[col]
                a.text(v + 0.2, yi, f"{v:.2f}%" + ("" if run == "r0_reference" else f"  ({dd:+.1f})"), va="center",
                       fontsize=9, color=INK, fontweight="bold" if run == best else "normal")
            else:
                a.barh(yi, ref[col], color="white", height=0.66, edgecolor="#bbbbbb", linewidth=1.2, ls="--",
                       hatch="//", zorder=2)
                a.text(ref[col] + 0.2, yi, "pending", va="center", fontsize=9, color=MUTED, style="italic")
        a.set_xlim(38, max(df[col].max(), ref[col]) + 5)
        a.set_ylim(y[-1] - 0.6, y[0] + 1.1)
        a.set_title(title, loc="left")
        a.set_xlabel("top-1 accuracy, %")
        grid(a, "x")
    ax[0].set_yticks(y, [RUNS[r][0] for r in order])
    header(fig, f"Round 02, {NAME}: every run against the r0 reference",
           "auto augmentation, no Mixup/CutMix, 300 epochs each, extras on, seed 42. Brackets = change vs r0. "
           "Dashed = r0, dotted = round 01 `full`. Test noise: about +/-1.3 points")
    savefig(fig, "pvx_01_overview.png")


# ------------------------------------------------------------------------------------------------------------
# 2. gap vs accuracy
# ------------------------------------------------------------------------------------------------------------
def fig_gap(df):
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ref = df.set_index("run").loc["r0_reference"]
    lo_y, hi_y = df.val_top1.min() - 1.5, df.val_top1.max() + 1.5
    ax.axvline(ref.gap, color="#bbbbbb", lw=1, ls="--")
    ax.axhline(ref.val_top1, color="#bbbbbb", lw=1, ls="--")
    ax.text(ref.gap - 0.05, lo_y + 0.3, "less memorizing  <-", ha="right", fontsize=9, color=MUTED)
    ax.text(ref.gap + 0.05, lo_y + 0.3, "->  more memorizing", ha="left", fontsize=9, color=MUTED)
    offset = {"r1_droppath0.2": (8, 7, "left"), "r7_sam": (-10, 6, "right"), "r0_reference": (-10, -14, "right"),
              "r5_wd0.1": (6, -16, "left"), "r3_droppath0.5": (8, -14, "left"), "r4_labelsmooth0.2": (8, -12, "left")}
    for r in df.itertuples():
        ax.scatter(r.gap, r.val_top1, s=150, color=RUNS[r.run][2], edgecolor="white", linewidth=2, zorder=3)
        dx, dy, ha = offset.get(r.run, (8, 6, "left"))
        ax.annotate(r.label, (r.gap, r.val_top1), xytext=(dx, dy), textcoords="offset points", fontsize=9,
                    color=INK, ha=ha)
    ax.fill_between([df.gap.min() - 1, ref.gap], ref.val_top1, hi_y, color="#2ca02c", alpha=0.07, lw=0)
    ax.text(df.gap.min() - 0.9, hi_y - 0.2, "goal: up and to the left\n(more accurate AND less memorizing)",
            fontsize=9, color="#2a7a2a", va="top")
    ax.set_xlim(df.gap.min() - 1, df.gap.max() + 1.5)
    ax.set_ylim(lo_y, hi_y)
    ax.set_xlabel("memorizing gap at the chosen epoch: clean-training minus validation top-1 (points)")
    ax.set_ylabel("validation top-1, %")
    grid(ax, "both")
    header(fig, f"{NAME}: which changes reduce memorizing without hurting accuracy?",
           "each dot is one run; dashed lines = r0 reference", top=0.9)
    savefig(fig, "pvx_02_gap_vs_accuracy.png")


# ------------------------------------------------------------------------------------------------------------
# 3. dose-response
# ------------------------------------------------------------------------------------------------------------
def fig_dose(df):
    d = df.set_index("run")
    fams = [("drop-path (stochastic depth)", ["r0_reference", "r1_droppath0.2", "r2_droppath0.4", "r3_droppath0.5"],
             "drop_path", "#08519c"),
            ("weight decay", ["r0_reference", "r5_wd0.1", "r6_wd0.2"], "weight_decay", "#238b45"),
            ("label smoothing", ["r0_reference", "r4_labelsmooth0.2"], "label_smoothing", "#8064a2")]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
    for a, (name, runs, key, col) in zip(ax, fams):
        runs = [r for r in runs if r in d.index]
        xs = [d.loc[r, key] for r in runs]
        a.plot(xs, [d.loc[r, "val_top1"] for r in runs], "-o", color=col, lw=2, ms=8, label="validation",
               markeredgecolor="white", markeredgewidth=1.5)
        a.plot(xs, [d.loc[r, "test_top1"] for r in runs], "--s", color=col, lw=1.5, ms=7, alpha=0.6, label="test",
               markeredgecolor="white")
        for x, r in zip(xs, runs):
            a.annotate(f"{d.loc[r, 'test_top1']:.1f}", (x, d.loc[r, "test_top1"]), xytext=(0, -14),
                       textcoords="offset points", ha="center", fontsize=8.5, color=MUTED)
        a.set_xticks(xs, [f"{x:g}" + (" (r0)" if r == "r0_reference" else "") for x, r in zip(xs, runs)])
        a.set_title(name, loc="left")
        a.set_xlabel("setting value")
        grid(a, "both")
    ax[0].set_ylabel("top-1 accuracy, %")
    ax[0].set_ylim(df.test_top1.min() - 1.2, df.val_top1.max() + 0.6)
    ax[0].legend(frameon=False, fontsize=9, loc="upper left")
    header(fig, f"{NAME}: how strong should each setting be?",
           "solid = validation (decides), dashed = test (grey numbers). r0 = the reference value of each setting")
    savefig(fig, "pvx_03_dose_response.png")


# ------------------------------------------------------------------------------------------------------------
# 4. learning curves (every run) + EMA effect + attention on the bird
# ------------------------------------------------------------------------------------------------------------
def fig_curves(df, hist):
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for r in hist:
        h = hist[r]
        c, lab = RUNS[r][2], RUNS[r][0]
        ax[0, 0].plot(h.epoch, h.val_acc1.rolling(9, center=True, min_periods=1).mean(), color=c, lw=1.8, label=lab)
        ax[0, 1].plot(h.epoch, h.val_loss, color=c, lw=1.6)
        cl = h.dropna(subset=["clean_acc1"])
        ax[1, 0].plot(cl.epoch, cl.clean_acc1, color=c, lw=1.8)
        ax[1, 1].plot(cl.epoch, cl.clean_acc1 - cl.val_acc1, color=c, lw=1.8)
    ax[0, 0].set_title("Validation top-1, % (9-epoch smoothing)", loc="left")
    ax[0, 1].set_title("Validation loss", loc="left")
    ax[1, 0].set_title("Training photos, no augmentation: top-1, %", loc="left")
    ax[1, 1].set_title("Memorizing gap (clean train minus val), points", loc="left")
    for a in ax.flat:
        a.set_xlabel("epoch")
        grid(a, "both")
    ax[0, 0].legend(frameon=False, fontsize=8.5, loc="lower right", ncol=2)
    header(fig, f"{NAME}: how training went, every run",
           "300 epochs each; every run's own curves.png is in results/patchvision_x/<run>/seed42/", top=0.93)
    savefig(fig, "pvx_04_learning_curves.png")


def fig_ema(hist):
    h = hist["r0_reference"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(h.epoch, h.valraw_acc1.rolling(5, center=True, min_periods=1).mean(), color="#9e9e9e", lw=1.8,
            label="raw (latest) weights")
    ax.plot(h.epoch, h.val_acc1.rolling(5, center=True, min_periods=1).mean(), color="#08519c", lw=2,
            label="EMA (averaged) weights: what is evaluated")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation top-1, % (5-epoch smoothing)")
    ax.legend(frameon=False, loc="lower right")
    grid(ax, "both")
    gain = (h.val_acc1 - h.valraw_acc1)[h.epoch >= 50]
    header(fig, f"{NAME}, r0: EMA weights vs raw weights",
           f"average gain after epoch 50: {gain.mean():+.1f} points; largest {gain.max():+.1f}", top=0.88)
    savefig(fig, "pvx_05_ema_effect.png")
    fact(f"EMA gain r0 after ep50 mean {gain.mean():.2f} max {gain.max():.2f}")


def fig_attention(df, hist):
    """Share of attention inside the bird box: per run (test) and over training (validation)."""
    o = row01(R01)
    e4 = row01(E4)
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1, 1.25]})
    a = ax[0]
    labels = ["round 01 `full`", "round 01 E4"] + list(df.label)
    vals = [o["test_attn_in_box"], e4["test_attn_in_box"]] + list(df.attn)
    cols = [R01_COL, E4_COL] + [RUNS[r][2] for r in df.run]
    y = np.arange(len(vals))[::-1]
    a.barh(y, vals, color=cols, height=0.62, edgecolor="white", linewidth=2)
    for yi, v in zip(y, vals):
        a.text(v + 0.005, yi, f"{v:.2f}", va="center", fontsize=9, color=INK)
    a.axvline(0.34, color=INK, lw=1, ls="--")
    a.text(0.345, y[0] + 0.55, "chance (box covers 34 %)", fontsize=8, color=MUTED)
    a.set_yticks(y, labels)
    a.set_xlim(0.3, 0.75)
    a.set_xlabel("share of attention inside the bird box (test)")
    a.set_title("Where the model looks, per run", loc="left")
    grid(a, "x")
    a = ax[1]
    h01 = pd.read_csv(R01 / "history.csv")
    a.plot(h01.epoch, h01.val_attn_in_box.rolling(5, center=True, min_periods=1).mean(), color=R01_COL, lw=2,
           label="round 01 `full`")
    for r in ["r0_reference", "r3_droppath0.5", "r8_combined"]:
        if r in hist:
            h = hist[r]
            a.plot(h.epoch, h.val_attn_in_box.rolling(5, center=True, min_periods=1).mean(), color=RUNS[r][2], lw=2,
                   label=RUNS[r][0])
    a.axhline(0.34, color=INK, lw=1, ls="--")
    a.set_xlabel("epoch")
    a.set_ylabel("attention inside the box (validation, 5-epoch smoothing)")
    a.set_title("How it learns to look at the bird", loc="left")
    a.legend(frameon=False, fontsize=8.5, loc="lower right")
    grid(a, "both")
    header(fig, f"{NAME}: does it still look at the bird?",
           "the bird-box loss (weight 0.1) is on in every run; the box is never given to the model as input")
    savefig(fig, "pvx_06_attention_on_bird.png")
    fact(f"attn in box: round01 full {o['test_attn_in_box']:.3f}, E4 {e4['test_attn_in_box']:.3f}; " +
         ", ".join(f"{r.run} {r.attn:.3f}" for r in df.itertuples()))


# ------------------------------------------------------------------------------------------------------------
# 5. calibration
# ------------------------------------------------------------------------------------------------------------
def fig_calibration(df):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})
    a = ax[0]
    y = np.arange(len(df))[::-1]
    a.barh(y, df.test_ece, color=[RUNS[r][2] for r in df.run], height=0.62, edgecolor="white", linewidth=2)
    for yi, r in zip(y, df.itertuples()):
        a.text(r.test_ece + 0.2, yi, f"{r.test_ece:.1f}%", va="center", fontsize=9, color=INK)
    a.set_yticks(y, df.label)
    a.set_xlabel("expected calibration error on test, % (lower = more honest confidence)")
    a.set_title("Calibration error per run", loc="left")
    grid(a, "x")
    a = ax[1]
    a.plot([0, 1], [0, 1], color="#bbbbbb", lw=1, ls="--")
    for r in ["r0_reference", "r4_labelsmooth0.2", final_run(df), "r5_wd0.1"]:
        if r not in df.run.values:
            continue
        rel = pd.read_csv(run_dir(r) / "test_reliability.csv")
        rel = rel[rel.fraction > 0.002]
        a.plot(rel.confidence, rel.accuracy, "-o", color=RUNS[r][2], lw=2, ms=5, label=RUNS[r][0])
    a.set_xlabel("model's confidence")
    a.set_ylabel("actual accuracy")
    a.set_title("Reliability: is '70% sure' right 70% of the time?", loc="left")
    a.legend(frameon=False, fontsize=8.5, loc="lower right")
    grid(a, "both")
    header(fig, f"{NAME}: are the confidence numbers honest?",
           "ECE of every run (left); reliability curves of four runs (right): below the dashed line = over-confident, "
           "above = under-confident")
    savefig(fig, "pvx_07_calibration.png")
    for r in df.run:
        conf = pd.read_csv(run_dir(r) / "test_predictions.csv").confidence
        fact(f"{r} mean confidence {100 * conf.mean():.1f}")


# ------------------------------------------------------------------------------------------------------------
# 6. round 01 (full, E4) vs round 02 (r0, final)
# ------------------------------------------------------------------------------------------------------------
def fig_before_after(df):
    o, e4 = row01(R01), row01(E4)
    d = df.set_index("run")
    best = final_run(df)
    cols = [("Round 01\n`full`", o["test_acc1"], o["overfit_gap_clean_minus_val"], o["test_ece"], R01_COL),
            ("Round 01\nE4 (ovf5)", e4["test_acc1"], e4["overfit_gap_clean_minus_val"], e4["test_ece"], E4_COL),
            ("Round 02\nr0", d.loc["r0_reference", "test_top1"], d.loc["r0_reference", "gap"],
             d.loc["r0_reference", "test_ece"], RUNS["r0_reference"][2]),
            ("Round 02\n" + RUNS[best][0].replace(" ", "\n", 1), d.loc[best, "test_top1"], d.loc[best, "gap"],
             d.loc[best, "test_ece"], RUNS[best][2])]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
    x = np.arange(len(cols))
    for a, idx, title, fmt in ((ax[0], 1, "Test top-1, %", "{:.2f}"), (ax[1], 2, "Memorizing gap, points", "{:.1f}"),
                               (ax[2], 3, "Calibration error (ECE), %", "{:.1f}")):
        v = [c[idx] for c in cols]
        a.bar(x, v, color=[c[4] for c in cols], width=0.62, edgecolor="white", linewidth=2)
        for xi, vi in zip(x, v):
            a.text(xi, vi * 1.01, fmt.format(vi), ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold")
        a.set_xticks(x, [c[0] for c in cols], fontsize=8.5)
        a.set_ylim(0, max(v) * 1.18)
        a.set_title(title, loc="left")
        grid(a, "y")
    header(fig, f"{NAME}: round 01 vs round 02",
           "lower is better for the gap and for ECE. `full` = round 01 base recipe (early stopping); E4 = round 01's "
           "tuned winner (drop-path 0.3 + auto aug + Mixup/CutMix, 300 epochs)")
    savefig(fig, "pvx_08_round01_vs_round02.png")
    return o, e4


def fig_improvement(df):
    best = final_run(df)
    a, b = correct(run_dir(best) / "test_predictions.csv"), correct(R01 / "test_predictions.csv")
    dd, n10, n01, p = mcnemar(a, b)
    b = b.loc[a.index]
    fo = pd.read_csv(R01 / "test_per_class.csv").set_index("class_name").f1
    fn = pd.read_csv(run_dir(best) / "test_per_class.csv").set_index("class_name").f1.loc[fo.index]
    dlt = (fn - fo).sort_values()
    dlt[dlt.abs() < 1e-9] = 0.0                                  # identical F1 up to float rounding counts as unchanged
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw={"width_ratios": [1, 1.6]})
    a0 = ax[0]
    cats = [("both right", int((a & b).sum()), "#bdbdbd"), ("only round 02\nright", n10, RUNS[best][2]),
            ("only round 01\nright", n01, "#7f7f7f"), ("both wrong", int((~a & ~b).sum()), "#e0e0e0")]
    yy = np.arange(len(cats))[::-1]
    a0.barh(yy, [c[1] for c in cats], color=[c[2] for c in cats], height=0.6, edgecolor="white", linewidth=2)
    for yi, c in zip(yy, cats):
        a0.text(c[1] + 40, yi, f"{c[1]:,}", va="center", fontsize=10, color=INK)
    a0.set_yticks(yy, [c[0] for c in cats])
    a0.set_xlim(0, max(c[1] for c in cats) * 1.2)
    a0.set_xlabel("test photos (5,794)")
    a0.set_title(f"Photo by photo: {n10} vs {n01}, p = {p:.1g}", loc="left")
    grid(a0, "x")
    a1 = ax[1]
    x = np.arange(len(dlt))
    a1.bar(x, dlt.values, width=1.0, color=[RUNS[best][2] if v > 0 else "#7f7f7f" for v in dlt.values], lw=0)
    a1.axhline(0, color=INK, lw=0.8)
    a1.axhline(dlt.median(), color=RUNS[best][2], lw=1, ls="--")
    a1.text(2, dlt.median() + 1, f"median species {dlt.median():+.1f}", fontsize=9, color=INK)
    for name in list(dlt.index[-2:]):
        i = dlt.index.get_loc(name)
        a1.annotate(f"{name} ({dlt[name]:+.0f})", (i, dlt[name]), xytext=(-6, -2), textcoords="offset points",
                    ha="right", va="center", fontsize=8.5, color=INK)
    lo = dlt.index[:2]
    a1.text(4, dlt.min() + 0.5, "biggest losses: " + ", ".join(f"{n} ({dlt[n]:+.0f})" for n in lo), fontsize=8.5,
            color=INK, va="center")
    a1.set_ylim(dlt.min() - 3, dlt.max() + 3)
    a1.set_xticks([])
    a1.set_xlabel(f"200 species, sorted by change   ({int((dlt > 0).sum())} better, {int((dlt < 0).sum())} worse, "
                  f"{int((dlt == 0).sum())} unchanged)")
    a1.set_ylabel("F1 change, points")
    a1.set_title("Species by species: test F1, round 02 minus round 01 `full`", loc="left")
    grid(a1, "y")
    header(fig, f"{NAME}: did it improve? Round 02 {RUNS[best][0]} vs round 01 `full`",
           "the same 5,794 test photos; p = exact paired test on the photos where the two models disagree")
    savefig(fig, "pvx_09_did_it_improve.png")
    fact(f"best {best} vs round01 full: diff {dd:+.2f} n10 {n10} n01 {n01} p {p:.1e}; both right {int((a & b).sum())} "
         f"both wrong {int((~a & ~b).sum())}; species better {int((dlt > 0).sum())} worse {int((dlt < 0).sum())} "
         f"same {int((dlt == 0).sum())} median {dlt.median():+.2f}")
    fact("species gains: " + ", ".join(f"{n} {dlt[n]:+.1f}" for n in dlt.index[::-1][:3]) +
         " | losses: " + ", ".join(f"{n} {dlt[n]:+.1f}" for n in dlt.index[:3]))
    r0 = correct(run_dir("r0_reference") / "test_predictions.csv")
    for nm, other in [("r0 vs round01 full", (r0, correct(R01 / "test_predictions.csv"))),
                      (f"{best} vs E4", (a, correct(E4 / "test_predictions.csv"))),
                      ("r0 vs E4", (r0, correct(E4 / "test_predictions.csv")))]:
        x_ = mcnemar(*other)
        fact(f"{nm}: diff {x_[0]:+.2f} n10 {x_[1]} n01 {x_[2]} p {x_[3]:.1e}")
    for r in ["r2_droppath0.4", "r3_droppath0.5"]:
        x_ = mcnemar(a, correct(run_dir(r) / "test_predictions.csv"))
        fact(f"{best} vs {r}: diff {x_[0]:+.2f} n10 {x_[1]} n01 {x_[2]} p {x_[3]:.1e}")


# ------------------------------------------------------------------------------------------------------------
# 7. final run in detail: species, confusions
# ------------------------------------------------------------------------------------------------------------
def fig_species(df):
    best = final_run(df)
    pc = pd.read_csv(run_dir(best) / "test_per_class.csv").sort_values("f1")
    worst, top = pc.head(10), pc.tail(10)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, dd, col, t in [(axs[0], top, "#08519c", "10 easiest species (test F1)"),
                           (axs[1], worst, "#c0395b", "10 hardest species (test F1)")]:
        ax.barh(dd.class_name, dd.f1, color=col)
        for yv, v in enumerate(dd.f1):
            ax.text(v + 1, yv, f"{v:.1f}", va="center", fontsize=8.5)
        ax.set_xlim(0, 100)
        ax.set_title(t, loc="left")
        ax.tick_params(axis="y", labelsize=8.5)
        grid(ax, "x")
    header(fig, f"{NAME}, {RUNS[best][0]}: which species are easy and which are hard?",
           "F1 on the test photos of each species")
    savefig(fig, "pvx_10_species.png")
    fact(f"{best} species: median {pc.f1.median():.1f}; >80 {(pc.f1 > 80).sum()}; <20 {(pc.f1 < 20).sum()}; "
         f"=0 {(pc.f1 == 0).sum()}")
    fact("easiest: " + ", ".join(f"{r.class_name} {r.f1:.1f}" for r in top.iloc[::-1].itertuples()))
    fact("hardest: " + ", ".join(f"{r.class_name} {r.f1:.1f}" for r in worst.itertuples()))

    pred = pd.read_csv(run_dir(best) / "test_predictions.csv")
    names = dict(zip(pc.class_id, pc.class_name))
    sup = pc.set_index("class_id").support
    t = pred[pred.pred != pred.label].groupby(["label", "pred"]).size().sort_values(ascending=False).head(12)
    lab = [f"{names[a]} → {names[b]}" for a, b in t.index]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.barh(lab[::-1], list(t.values)[::-1], color="#08519c")
    for yv, ((a_, _), n) in enumerate(list(t.items())[::-1]):
        ax.text(n + 0.2, yv, f"{n} photos ({100 * n / sup[a_]:.0f} % of the species)", va="center", fontsize=8.5)
    ax.set_xlim(0, max(t.values) * 1.6)
    ax.set_xlabel("test photos")
    ax.tick_params(axis="y", labelsize=8.5)
    grid(ax, "x")
    header(fig, f"{NAME}, {RUNS[best][0]}: most common mistakes", "true species → predicted species", top=0.88)
    savefig(fig, "pvx_11_confusions.png")
    fact("confusions: " + "; ".join(f"{l} {n}" for l, n in zip(lab, t.values)))


# ------------------------------------------------------------------------------------------------------------
# 8. against the plain ViT and the CNNs: change vs each model's own r0 (validation)
# ------------------------------------------------------------------------------------------------------------
def fig_models(df):
    models = {"PatchVision-X": RES, **OTHERS}
    cols = {"PatchVision-X": "#c0392b", "Plain ViT": "#08519c", "CNN small": "#74c476", "CNN large": "#238b45"}
    runs = [r for r in RUNS if r != "r0_reference"]
    delta, final = {}, {}
    for m, root in models.items():
        v = {}
        for r in RUNS:
            f = root / r / "seed42" / "test_metrics.json"
            if f.exists():
                row = json.loads(f.read_text())["row"]
                v[r] = (row["val_acc1_at_best"], row["test_acc1"])
        if "r0_reference" not in v:
            continue
        delta[m] = [v[r][0] - v["r0_reference"][0] if r in v else np.nan for r in runs]
        final[m] = (v["r0_reference"][1], v.get("r8_combined", (np.nan, np.nan))[1])
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.9), gridspec_kw={"width_ratios": [1.9, 1]})
    a = ax[0]
    x = np.arange(len(runs))
    w = 0.8 / len(delta)
    for i, (m, dv) in enumerate(delta.items()):
        a.bar(x - 0.4 + w / 2 + i * w, dv, width=w, color=cols[m], edgecolor="white", linewidth=1, label=m)
    a.axhline(0, color=INK, lw=0.8)
    def short(r):
        w = RUNS[r][0].replace(" (winners)", "").split(" ")
        return w[0] + "\n" + " ".join(w[1:-1]) + ("\n" if len(w) > 2 else "") + (w[-1] if len(w) > 1 else "")
    a.set_xticks(x, [short(r) for r in runs], fontsize=8.5)
    a.set_ylabel("validation top-1 change vs own r0, points")
    a.set_title("Each change against each model's own r0 (validation)", loc="left")
    a.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper left")
    grid(a, "y")
    a = ax[1]
    names = list(final)
    xx = np.arange(len(names))
    a.bar(xx - 0.2, [final[m][0] for m in names], width=0.38, color="#d9d9d9", edgecolor="white", label="r0")
    a.bar(xx + 0.2, [final[m][1] for m in names], width=0.38, color=[cols[m] for m in names], edgecolor="white",
          label="r8 combined")
    for xi, m in zip(xx, names):
        a.text(xi + 0.2, final[m][1] + 0.4, f"{final[m][1]:.1f}", ha="center", fontsize=9, fontweight="bold")
        a.text(xi - 0.2, final[m][0] + 0.4, f"{final[m][0]:.1f}", ha="center", fontsize=8.5, color=MUTED)
    a.set_xticks(xx, names, fontsize=8.5)
    a.set_ylim(30, max(v[1] for v in final.values()) + 4)
    a.set_ylabel("test top-1, %")
    a.set_title("Test top-1: r0 (grey) and r8", loc="left")
    grid(a, "y")
    header(fig, "Round 02 across the four models: the same runs, the same rule",
           "PatchVision-X 4.27 M, plain ViT 4.31 M, CNN small 4.22 M, CNN large 11.28 M parameters; one seed (42)")
    savefig(fig, "pvx_12_against_other_models.png")
    for m in delta:
        fact(f"{m}: val change vs r0 " + ", ".join(f"{r} {v:+.2f}" for r, v in zip(runs, delta[m])) +
             f"; r0 test {final[m][0]:.2f} r8 test {final[m][1]:.2f}")


# ------------------------------------------------------------------------------------------------------------
# GIF: validation race
# ------------------------------------------------------------------------------------------------------------
def gif_race(hist):
    runs = [r for r in RUNS if r in hist]
    sm = {r: hist[r].set_index("epoch").val_acc1.rolling(9, center=True, min_periods=1).mean() for r in runs}
    frames = list(range(5, 301, 5)) + [300] * 12
    fig, ax = plt.subplots(figsize=(10.5, 5))
    lines = {r: ax.plot([], [], color=RUNS[r][2], lw=2.2)[0] for r in runs}
    tags = {r: ax.text(0, 0, "", fontsize=8.5, color=INK, va="center") for r in runs}
    keys = {r: ax.plot([], [], "s", color=RUNS[r][2], ms=7)[0] for r in runs}
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 58)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation top-1, % (9-epoch smoothing)")
    grid(ax, "both")
    fig.tight_layout()

    def update(e):
        ax.set_title(f"{NAME}, round 02 race: epoch {e}/300", loc="left", fontsize=11.5)
        ends = []
        for r in runs:
            s = sm[r].loc[:e]
            lines[r].set_data(s.index, s.values)
            ends.append((s.values[-1], r))
        ends.sort()
        last = -99
        for v, r in ends:
            yv = max(v, last + 1.8)
            last = yv
            keys[r].set_data([e + 6], [yv])
            tags[r].set_position((e + 11, yv))
            tags[r].set_text(f"{RUNS[r][0]}  {v:.1f}")
        return []

    FuncAnimation(fig, update, frames=frames, blit=False).save(ANI / "pvx_race.gif", writer=PillowWriter(fps=12), dpi=85)
    plt.close(fig)
    print("wrote", ANI / "pvx_race.gif")


# ------------------------------------------------------------------------------------------------------------
# tables + facts
# ------------------------------------------------------------------------------------------------------------
def tables(df):
    ref = df.set_index("run").loc["r0_reference"]
    best = final_run(df)
    lines = ["| Run | What changes | Best epoch | **Val top-1** | vs r0 (photos) | Gap (pts) | Test top-1 | Test top-5 | "
             "Test macro-F1 | Test ECE | Attention in box |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for run in RUNS:
        if run not in df.run.values:
            lines.append(f"| {run} | {WHAT[run]} | *pending* | *pending* | | | | | | | |")
            continue
        r = df.set_index("run").loc[run]
        b = "**" if run == best else ""
        dv = "" if run == "r0_reference" else f"{round((r.val_top1 - ref.val_top1) * 6):+d}"
        lines.append(f"| {b}{run}{b} | {WHAT[run]} | {r.best_epoch} | {b}{r.val_top1:.2f} %{b} | {dv} | {r.gap:.1f} | "
                     f"{r.test_top1:.2f} % | {r.test_top5:.2f} % | {r.test_f1:.2f} % | {r.test_ece:.2f} % | "
                     f"{r.attn:.2f} |")
    (TAB / "patchvision_x_round02.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    df.to_csv(TAB / "patchvision_x_round02.csv", index=False)
    r0 = correct(run_dir("r0_reference") / "test_predictions.csv")
    sig = ["| Run vs r0 | Change, test top-1 (pts) | Only run right | Only r0 right | p (exact paired test) | Significant at 5 %? |",
           "|---|---|---|---|---|---|"]
    for a in [r for r in RUNS if r != "r0_reference" and r in df.run.values]:
        dd, n10, n01, p = mcnemar(correct(run_dir(a) / "test_predictions.csv"), r0)
        sig.append(f"| {a} | {dd:+.2f} | {n10} | {n01} | {p:.4f} | {'yes' if p < 0.05 else 'no'} |")
    (TAB / "patchvision_x_significance.md").write_text("\n".join(sig) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\n".join(sig))
    for r in df.itertuples():
        fact(f"{r.run}: val {r.val_top1:.2f} gap {r.gap:.1f} test {r.test_top1:.2f} top5 {r.test_top5:.2f} "
             f"f1 {r.test_f1:.2f} ece {r.test_ece:.2f} attn {r.attn:.3f} best_ep {r.best_epoch} {r.train_min:.0f} min "
             f"{r.sec_epoch:.1f} s/ep latency {r.latency:.1f} ms")


def facts_curves(hist):
    for r, h in hist.items():
        cl = h.dropna(subset=["clean_acc1"])
        e99 = cl.loc[cl.clean_acc1 >= 99, "epoch"]
        vmin = h.loc[h.val_loss.idxmin()]
        fact(f"{r}: lowest val loss {vmin.val_loss:.3f} @ {int(vmin.epoch)}, final {h.val_loss.iloc[-1]:.3f}; "
             f"clean >=99 first @ {int(e99.iloc[0]) if len(e99) else None}; final clean {cl.clean_acc1.iloc[-1]:.1f}; "
             f"final gap {(cl.clean_acc1 - cl.val_acc1).iloc[-1]:.1f}; final gate mean {h.val_gate_mean.iloc[-1]:.3f} "
             f"sparsity {h.val_gate_sparsity.iloc[-1]:.4f}")


if __name__ == "__main__":
    df, hist = load()
    fig_overview(df)
    fig_gap(df)
    fig_dose(df)
    fig_curves(df, hist)
    fig_ema(hist)
    fig_attention(df, hist)
    fig_calibration(df)
    o, e4 = fig_before_after(df)
    for nm, x in (("round01 full", o), ("round01 E4", e4)):
        fact(f"{nm}: test {x['test_acc1']:.2f} top5 {x['test_acc5']:.2f} f1 {x['test_macro_f1']:.2f} "
             f"gap {x['overfit_gap_clean_minus_val']:.1f} ece {x['test_ece']:.2f} val {x['val_acc1_at_best']:.2f} "
             f"best_ep {x['best_epoch']} sec/ep {x['sec_per_epoch']:.1f}")
    fig_improvement(df)
    fig_species(df)
    fig_models(df)
    gif_race(hist)
    tables(df)
    facts_curves(hist)
    fact(f"final run (validation rule): {final_run(df)}")
    (HERE / "docs" / "facts.txt").write_text("\n".join(FACTS) + "\n", encoding="utf-8")
