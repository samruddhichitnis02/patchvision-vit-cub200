"""Figures, animations and tables for CNN_Baseline/02_autoaug_nomix_regularization/README.md.

Reads only the finished runs in results/<model>/<run>/seed42/ (and round 01's cnn_small for the before/after
comparison), so it can be re-run at any time:

    python docs/make_figures.py            (from inside CNN_Baseline/02_autoaug_nomix_regularization/)

Runs that do not exist yet (r8_combined, every cnn_large run) are drawn as "pending" placeholders and picked up
automatically once they finish. Writes docs/figures/*.png, docs/animations/*.gif, docs/tables/*.md and facts.txt.

Colour = which KIND of change a run makes (reference, drop-path, label smoothing, weight decay, SAM, combined);
darker shade = stronger setting. Every run is also labelled by name, so colour is never the only cue.
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

HERE = Path(__file__).resolve().parents[1]                       # CNN_Baseline/02_autoaug_nomix_regularization/
RES = HERE / "results"
R01 = HERE.parent / "01_strongaug_mixup_cutmix" / "results"
FIG, ANI, TAB = HERE / "docs" / "figures", HERE / "docs" / "animations", HERE / "docs" / "tables"
for d in (FIG, ANI, TAB):
    d.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID = "#222222", "#666666", "#e6e6e6"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100,
                     "axes.edgecolor": "#999999", "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.titleweight": "bold", "axes.titlesize": 11.5})

# run -> (short label, family, colour). Families: grey reference, blue drop-path, purple label smoothing,
# green weight decay, orange SAM, red combined. Shades get darker with a stronger setting.
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
MODELS = {"cnn_small": "CNN small (4.22 M)", "cnn_large": "CNN large (11.28 M)"}
ONE_PHOTO_TEST = 100 / 5794


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


def run_dir(model, run):
    return RES / model / run / "seed42"


def load(model):
    """Finished runs of one model: summary rows + per-epoch histories."""
    rows, hist = [], {}
    for run in RUNS:
        d = run_dir(model, run)
        if not (d / "test_metrics.json").exists():
            continue
        m = json.loads((d / "test_metrics.json").read_text())
        r = m["row"]
        cfg = m["config"]
        rows.append({"run": run, "label": RUNS[run][0], "family": RUNS[run][1], "best_epoch": r["best_epoch"],
                     "val_top1": r["val_acc1_at_best"], "gap": r["overfit_gap_clean_minus_val"],
                     "test_top1": r["test_acc1"], "test_top5": r["test_acc5"], "test_f1": r["test_macro_f1"],
                     "test_ece": r["test_ece"], "train_min": r["train_time_s"] / 60,
                     "drop_path": cfg["model"]["drop_path"], "label_smoothing": cfg["train"]["label_smoothing"],
                     "weight_decay": cfg["train"]["weight_decay"], "sam": cfg["train"].get("use_sam", False)})
        hist[run] = pd.read_csv(d / "history.csv")
    return pd.DataFrame(rows), hist


def mcnemar(model, a, b):
    """Exact paired test on the same test photos: (A minus B in points, only-A-right, only-B-right, p)."""
    pa = pd.read_csv(run_dir(model, a) / "test_predictions.csv").set_index("image_id")
    pb = pd.read_csv(run_dir(model, b) / "test_predictions.csv").set_index("image_id")
    ca, cb = pa.pred == pa.label, pb.loc[pa.index].pred == pb.loc[pa.index].label
    n10, n01 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    n, k = n10 + n01, min(n10, n01)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return (ca.mean() - cb.mean()) * 100, n10, n01, p


# ------------------------------------------------------------------------------------------------------------
# 1. the round at a glance: validation (decides) and test (record) for every run, r8 placeholder
# ------------------------------------------------------------------------------------------------------------
def fig_overview(model, df):
    ref = df.set_index("run").loc["r0_reference"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    order = list(RUNS)
    y = np.arange(len(order))[::-1]
    for a, col, title in ((ax[0], "val_top1", "Validation top-1 (600 photos): DECIDES the winners"),
                          (ax[1], "test_top1", "Test top-1 (5,794 photos): for the record")):
        a.axvline(ref[col], color=INK, lw=1, ls="--", zorder=1)
        for yi, run in zip(y, order):
            if run in df.run.values:
                v = df.set_index("run").loc[run, col]
                a.barh(yi, v, color=RUNS[run][2], height=0.66, edgecolor="white", linewidth=2, zorder=2)
                d = v - ref[col]
                a.text(v + 0.3, yi, f"{v:.2f}%" + ("" if run == "r0_reference" else f"  ({d:+.1f})"), va="center",
                       fontsize=9, color=INK, fontweight="bold" if run == df.loc[df.val_top1.idxmax(), "run"] else
                       "normal")
            else:
                a.barh(yi, ref[col], color="white", height=0.66, edgecolor="#bbbbbb", linewidth=1.2, ls="--",
                       hatch="//", zorder=2)
                a.text(ref[col] + 0.3, yi, "pending", va="center", fontsize=9, color=MUTED, style="italic")
        a.set_xlim(35, max(df[col].max(), ref[col]) + 6)
        a.set_title(title, loc="left")
        a.set_xlabel("top-1 accuracy, %")
        grid(a, "x")
    ax[0].set_yticks(y, [RUNS[r][0] for r in order])
    ax[0].text(ref["val_top1"], len(order) - 0.35, " r0 reference", fontsize=8.5, color=MUTED, va="bottom")
    header(fig, f"Round 02, {MODELS[model]}: every run against the r0 reference",
           "auto augmentation, no Mixup/CutMix, 300 epochs each, seed 42. Numbers in brackets = change vs r0. "
           f"Test noise: 1 photo = {ONE_PHOTO_TEST:.3f} points, 95% interval about +/-1.3 points")
    savefig(fig, f"{model}_01_overview.png")


# ------------------------------------------------------------------------------------------------------------
# 2. does it reduce overfitting? gap vs accuracy
# ------------------------------------------------------------------------------------------------------------
def fig_gap(model, df):
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ref = df.set_index("run").loc["r0_reference"]
    ax.axvline(ref.gap, color="#bbbbbb", lw=1, ls="--")
    ax.axhline(ref.val_top1, color="#bbbbbb", lw=1, ls="--")
    ax.text(ref.gap - 0.05, ax.get_ylim()[0] if False else 38.6, "less overfitting  <-", ha="right", fontsize=9,
            color=MUTED)
    ax.text(ref.gap + 0.05, 38.6, "->  more overfitting", ha="left", fontsize=9, color=MUTED)
    offset = {"cnn_small": {"r1_droppath0.2": (8, -15, "left"), "r4_labelsmooth0.2": (8, 8, "left"),
                            "r5_wd0.1": (-8, 8, "right")},
              "cnn_large": {"r1_droppath0.2": (8, 2, "left"), "r5_wd0.1": (-10, -17, "right"),
                            "r4_labelsmooth0.2": (8, 8, "left")}}[model]   # hand-placed so close points never overlap
    for r in df.itertuples():
        ax.scatter(r.gap, r.val_top1, s=150, color=RUNS[r.run][2], edgecolor="white", linewidth=2, zorder=3)
        dx, dy, ha = offset.get(r.run, (8, 6, "left"))
        ax.annotate(r.label, (r.gap, r.val_top1), xytext=(dx, dy), textcoords="offset points", fontsize=9,
                    color=INK, ha=ha)
    ax.fill_between([df.gap.min() - 1, ref.gap], ref.val_top1, 52, color="#2ca02c", alpha=0.07, lw=0)
    ax.text(df.gap.min() - 0.9, ref.val_top1 + 0.25, "goal: up and to the left\n(more accurate AND less memorizing)",
            fontsize=9, color="#2a7a2a", va="bottom")
    ax.set_xlim(df.gap.min() - 1, df.gap.max() + 1.2)
    ax.set_ylim(38, 52)
    ax.set_xlabel("overfitting gap at the chosen epoch: clean-training minus validation top-1 (points)")
    ax.set_ylabel("validation top-1, %")
    grid(ax, "both")
    header(fig, f"{MODELS[model]}: which changes reduce memorizing without hurting accuracy?",
           "A smaller gap only counts as progress if accuracy goes UP (drop-path 0.5 shrinks the gap by learning less)",
           top=0.9)
    savefig(fig, f"{model}_02_gap_vs_accuracy.png")


# ------------------------------------------------------------------------------------------------------------
# 3. dose-response: how strong should each setting be?
# ------------------------------------------------------------------------------------------------------------
def fig_dose(model, df):
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
    ax[0].legend(frameon=False, fontsize=9, loc="lower left")
    ax[1].text(0.97, 0.04, "still rising at 0.2,\nthe largest value tried", transform=ax[1].transAxes, ha="right",
               fontsize=9, color="#238b45")
    ax[0].text(0.97, 0.9, "best at 0.2,\ntoo strong above", transform=ax[0].transAxes, ha="right", fontsize=9,
               color="#08519c", va="top")
    header(fig, f"{MODELS[model]}: how strong should each setting be?",
           "solid = validation (decides), dashed = test (grey numbers). r0 = the reference value of each setting")
    savefig(fig, f"{model}_03_dose_response.png")


# ------------------------------------------------------------------------------------------------------------
# 4. learning curves: validation accuracy and the memorizing gap over 300 epochs
# ------------------------------------------------------------------------------------------------------------
def fig_curves(model, df, hist):
    show = [r for r in ["r0_reference", "r1_droppath0.2", "r3_droppath0.5", "r4_labelsmooth0.2", "r6_wd0.2",
                        "r7_sam", "r8_combined"] if r in hist]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
    for r in show:
        h = hist[r]
        c, lab = RUNS[r][2], RUNS[r][0]
        ax[0].plot(h.epoch, h.val_acc1.rolling(9, center=True, min_periods=1).mean(), color=c, lw=2, label=lab)
        cl = h.dropna(subset=["clean_acc1"])
        ax[1].plot(cl.epoch, cl.clean_acc1, color=c, lw=2)
        ax[2].plot(cl.epoch, cl.clean_acc1 - cl.val_acc1, color=c, lw=2)
    ax[0].set_title("Validation top-1 (9-epoch smoothing)", loc="left")
    ax[1].set_title("Training photos, no augmentation", loc="left")
    ax[2].set_title("Memorizing gap (clean train minus val)", loc="left")
    ax[0].set_ylabel("%")
    ax[2].set_ylabel("points")
    for a in ax:
        a.set_xlabel("epoch")
        grid(a, "both")
    ax[0].legend(frameon=False, fontsize=8.5, loc="lower right")
    header(fig, f"{MODELS[model]}: how training went", "a selection of runs for readability; every run's own "
           "curves.png is in results/<model>/<run>/seed42/")
    savefig(fig, f"{model}_04_learning_curves.png")


# ------------------------------------------------------------------------------------------------------------
# 5. calibration: ECE per run + reliability of reference / label smoothing / weight decay
# ------------------------------------------------------------------------------------------------------------
def fig_calibration(model, df):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})
    a = ax[0]
    y = np.arange(len(df))[::-1]
    a.barh(y, df.test_ece, color=[RUNS[r][2] for r in df.run], height=0.62, edgecolor="white", linewidth=2)
    for yi, r in zip(y, df.itertuples()):
        a.text(r.test_ece + 0.25, yi, f"{r.test_ece:.1f}%", va="center", fontsize=9, color=INK)
    a.set_yticks(y, df.label)
    a.set_xlabel("expected calibration error on test, % (lower = more honest confidence)")
    a.set_title("Calibration error per run", loc="left")
    grid(a, "x")
    a = ax[1]
    a.plot([0, 1], [0, 1], color="#bbbbbb", lw=1, ls="--")
    for r in ["r0_reference", "r4_labelsmooth0.2", "r6_wd0.2"]:
        if r not in df.run.values:
            continue
        rel = pd.read_csv(run_dir(model, r) / "test_reliability.csv")
        rel = rel[rel.fraction > 0.002]
        a.plot(rel.confidence, rel.accuracy, "-o", color=RUNS[r][2], lw=2, ms=5, label=RUNS[r][0])
    a.text(0.62, 0.2, "below the line = over-confident", fontsize=8.5, color=MUTED)
    a.text(0.05, 0.82, "above the line =\nunder-confident", fontsize=8.5, color=MUTED)
    a.set_xlabel("model's confidence")
    a.set_ylabel("actual accuracy")
    a.set_title("Reliability: is '70% sure' right 70% of the time?", loc="left")
    a.legend(frameon=False, fontsize=8.5, loc="lower right")
    grid(a, "both")
    header(fig, f"{MODELS[model]}: are the confidence numbers honest?",
           "label smoothing 0.2 makes the model UNDER-confident (right more often than it says); "
           "accuracy is unaffected and this can be fixed after training (temperature scaling)")
    savefig(fig, f"{model}_05_calibration.png")


# ------------------------------------------------------------------------------------------------------------
# 6. round 01 vs round 02: old recipe, new reference, best single change
# ------------------------------------------------------------------------------------------------------------
def fig_before_after(model, df):
    old = R01 / model / "seed42" / "test_metrics.json"
    if not old.exists():
        return None
    o = json.loads(old.read_text())["row"]
    d = df.set_index("run")
    best = df.loc[df.val_top1.idxmax(), "run"]
    cols = [("Round 01\nstrong aug\n+ Mixup/CutMix", o["test_acc1"], o["test_acc5"], o["test_macro_f1"],
             o["overfit_gap_clean_minus_val"], o["test_ece"], "#cccccc"),
            ("Round 02 r0\nauto aug,\nno mixing", d.loc["r0_reference", "test_top1"], d.loc["r0_reference", "test_top5"],
             d.loc["r0_reference", "test_f1"], d.loc["r0_reference", "gap"], d.loc["r0_reference", "test_ece"],
             RUNS["r0_reference"][2]),
            ("Round 02 best\n" + RUNS[best][0].replace(" ", "\n", 1), d.loc[best, "test_top1"], d.loc[best, "test_top5"], d.loc[best, "test_f1"],
             d.loc[best, "gap"], d.loc[best, "test_ece"], RUNS[best][2])]
    if best == "r8_combined" and "r6_wd0.2" in d.index:          # the simpler alternative, shown beside the winner
        cols.insert(2, ("Round 02 r6\nweight decay\n0.2 alone", d.loc["r6_wd0.2", "test_top1"],
                        d.loc["r6_wd0.2", "test_top5"], d.loc["r6_wd0.2", "test_f1"], d.loc["r6_wd0.2", "gap"],
                        d.loc["r6_wd0.2", "test_ece"], RUNS["r6_wd0.2"][2]))
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.3))
    x = np.arange(len(cols))
    for a, idx, title, fmt in ((ax[0], 1, "Test top-1, %", "{:.2f}"), (ax[1], 4, "Overfitting gap, points", "{:.1f}"),
                               (ax[2], 5, "Calibration error (ECE), %", "{:.1f}")):
        v = [c[idx] for c in cols]
        a.bar(x, v, color=[c[6] for c in cols], width=0.62, edgecolor="white", linewidth=2)
        for xi, vi in zip(x, v):
            a.text(xi, vi * 1.01, fmt.format(vi), ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold")
        a.set_xticks(x, [c[0] for c in cols], fontsize=8.5)
        a.set_ylim(0, max(v) * 1.18)
        a.set_title(title, loc="left")
        grid(a, "y")
    header(fig, f"{MODELS[model]}: round 01 recipe vs round 02",
           "lower is better for the gap and for ECE. Round 01 used early stopping; round 02 trains every run for "
           "300 epochs")
    savefig(fig, f"{model}_06_round01_vs_round02.png")
    return o


# ------------------------------------------------------------------------------------------------------------
# 7. did it improve? final round-02 model vs the original round-01 model, photo by photo and species by species
# ------------------------------------------------------------------------------------------------------------
def fig_improvement(model, df):
    old = R01 / model / "seed42"
    if not (old / "test_predictions.csv").exists():
        return None
    best = df.loc[df.val_top1.idxmax(), "run"]
    new = run_dir(model, best)
    po = pd.read_csv(old / "test_predictions.csv").set_index("image_id")
    pn = pd.read_csv(new / "test_predictions.csv").set_index("image_id").loc[po.index]
    a, b = pn.pred == pn.label, po.pred == po.label
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    n, k = n10 + n01, min(n10, n01)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    fo = pd.read_csv(old / "test_per_class.csv").set_index("class_name").f1
    fn = pd.read_csv(new / "test_per_class.csv").set_index("class_name").f1.loc[fo.index]
    dlt = (fn - fo).sort_values()
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw={"width_ratios": [1, 1.6]})
    a0 = ax[0]
    cats = [("both right", int((a & b).sum()), "#bdbdbd"), ("only the NEW\nmodel right", n10, RUNS[best][2]),
            ("only the OLD\nmodel right", n01, "#7f7f7f"), ("both wrong", int((~a & ~b).sum()), "#e0e0e0")]
    yy = np.arange(len(cats))[::-1]
    a0.barh(yy, [c[1] for c in cats], color=[c[2] for c in cats], height=0.6, edgecolor="white", linewidth=2)
    for yi, c in zip(yy, cats):
        a0.text(c[1] + 40, yi, f"{c[1]:,}", va="center", fontsize=10, color=INK)
    a0.set_yticks(yy, [c[0] for c in cats])
    a0.set_xlim(0, max(c[1] for c in cats) * 1.2)
    a0.set_xlabel("test photos (5,794)")
    a0.set_title(f"Photo by photo: {n10} vs {n01}, p = {p:.0e}", loc="left")
    grid(a0, "x")
    a1 = ax[1]
    x = np.arange(len(dlt))
    a1.bar(x, dlt.values, width=1.0, color=[RUNS[best][2] if v > 0 else "#7f7f7f" for v in dlt.values], lw=0)
    a1.axhline(0, color=INK, lw=0.8)
    a1.axhline(dlt.median(), color=RUNS[best][2], lw=1, ls="--")
    a1.text(2, dlt.median() + 1, f"median species {dlt.median():+.1f}", fontsize=9, color=INK)
    for name in list(dlt.index[-2:]):                                     # two biggest gains, at the right end
        i = dlt.index.get_loc(name)
        a1.annotate(f"{name} ({dlt[name]:+.0f})", (i, dlt[name]), xytext=(-6, -2), textcoords="offset points",
                    ha="right", va="center", fontsize=8.5, color=INK)
    lo = dlt.index[:2]                                                    # two biggest losses, one label
    a1.text(4, dlt.min() + 0.5, "biggest losses: " + ", ".join(f"{n} ({dlt[n]:+.0f})" for n in lo), fontsize=8.5,
            color=INK, va="center")
    a1.set_ylim(dlt.min() - 3, dlt.max() + 3)
    a1.set_xticks([])
    a1.set_xlabel(f"200 species, sorted by change   ({int((dlt > 0).sum())} better, {int((dlt < 0).sum())} worse, "
                  f"{int((dlt == 0).sum())} unchanged)")
    a1.set_ylabel("F1 change, points")
    a1.set_title("Species by species: test F1, new minus old", loc="left")
    grid(a1, "y")
    header(fig, f"{MODELS[model]}: did it improve? Round 02 {RUNS[best][0]} vs the original round-01 model",
           "the same 5,794 test photos; p = exact paired test on the photos where the two models disagree")
    savefig(fig, f"{model}_07_did_it_improve.png")
    return {"best": best, "n10": n10, "n01": n01, "p": p, "better": int((dlt > 0).sum()),
            "worse": int((dlt < 0).sum()), "median": float(dlt.median())}


# ------------------------------------------------------------------------------------------------------------
# GIF: validation-accuracy race of every run
# ------------------------------------------------------------------------------------------------------------
def gif_race(model, hist):
    runs = [r for r in RUNS if r in hist]
    sm = {r: hist[r].set_index("epoch").val_acc1.rolling(9, center=True, min_periods=1).mean() for r in runs}
    epochs = np.arange(1, 301)
    frames = list(range(5, 301, 5)) + [300] * 12
    fig, ax = plt.subplots(figsize=(10.5, 5))
    lines = {r: ax.plot([], [], color=RUNS[r][2], lw=2.2)[0] for r in runs}
    tags = {r: ax.text(0, 0, "", fontsize=8.5, color=INK, va="center") for r in runs}
    keys = {r: ax.plot([], [], "s", color=RUNS[r][2], ms=7)[0] for r in runs}      # colour key next to each label
    ax.set_xlim(0, 390)
    ax.set_ylim(0, 56)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation top-1, % (9-epoch smoothing)")
    grid(ax, "both")
    ax.set_title(f"{MODELS[model]}, round 02 race: epoch 300/300", loc="left", fontsize=11.5)
    fig.tight_layout()

    def update(e):
        ax.set_title(f"{MODELS[model]}, round 02 race: epoch {e}/300", loc="left", fontsize=11.5)
        ends = []
        for r in runs:
            s = sm[r].loc[:e]
            lines[r].set_data(s.index, s.values)
            ends.append((s.values[-1], r))
        ends.sort()
        last = -99
        for v, r in ends:                           # nudge labels apart so they never overlap
            yv = max(v, last + 1.6)
            last = yv
            keys[r].set_data([e + 6], [yv])
            tags[r].set_position((e + 11, yv))
            tags[r].set_text(f"{RUNS[r][0]}  {v:.1f}")
        return []

    FuncAnimation(fig, update, frames=frames, blit=False).save(ANI / f"{model}_race.gif",
                                                               writer=PillowWriter(fps=12), dpi=85)
    plt.close(fig)
    print("wrote", ANI / f"{model}_race.gif")


# ------------------------------------------------------------------------------------------------------------
# tables + facts
# ------------------------------------------------------------------------------------------------------------
def tables(model, df, old):
    ref = df.set_index("run").loc["r0_reference"]
    lines = ["| Run | What changes | Best epoch | **Val top-1** | vs r0 | Gap (pts) | Test top-1 | Test top-5 | "
             "Test macro-F1 | Test ECE |", "|---|---|---|---|---|---|---|---|---|---|"]
    what = {"r0_reference": "nothing (new base recipe)", "r1_droppath0.2": "drop-path 0 -> 0.2",
            "r2_droppath0.4": "drop-path 0 -> 0.4", "r3_droppath0.5": "drop-path 0 -> 0.5",
            "r4_labelsmooth0.2": "label smoothing 0.1 -> 0.2", "r5_wd0.1": "weight decay 0.05 -> 0.1",
            "r6_wd0.2": "weight decay 0.05 -> 0.2", "r7_sam": "SAM optimizer (rho 0.05)",
            "r8_combined": "winners of r1-r7 combined"}
    best = df.loc[df.val_top1.idxmax(), "run"]
    for run in RUNS:
        if run not in df.run.values:
            lines.append(f"| {run} | {what[run]} | *pending* | *pending* | | | | | | |")
            continue
        r = df.set_index("run").loc[run]
        b = "**" if run == best else ""
        dv = "" if run == "r0_reference" else f"{r.val_top1 - ref.val_top1:+.2f}"
        lines.append(f"| {b}{run}{b} | {what[run]} | {r.best_epoch} | {b}{r.val_top1:.2f} %{b} | {dv} | {r.gap:.1f} | "
                     f"{r.test_top1:.2f} % | {r.test_top5:.2f} % | {r.test_f1:.2f} % | {r.test_ece:.2f} % |")
    (TAB / f"{model}_round02.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    df.to_csv(TAB / f"{model}_round02.csv", index=False)

    sig = ["| Comparison (A vs B) | A minus B, test top-1 (pts) | Photos only A right | Photos only B right | "
           "p-value (exact paired test) | Significant at 5 %? |", "|---|---|---|---|---|---|"]
    for a in [r for r in RUNS if r != "r0_reference" and r in df.run.values]:
        dd, n10, n01, p = mcnemar(model, a, "r0_reference")
        sig.append(f"| {a} vs r0_reference | {dd:+.2f} | {n10} | {n01} | {p:.4f} | {'yes' if p < 0.05 else 'no'} |")
    (TAB / f"{model}_significance.md").write_text("\n".join(sig) + "\n", encoding="utf-8")
    print("wrote tables for", model)
    print("\n".join(lines))
    print("\n".join(sig))


# ============================================================================================================
# CNN small vs CNN large (same layout of ideas as the round-01 README): each model's FINAL run = its best
# validation run (r8 once it exists and wins). Figures compare_*.png, animations compare_*.gif.
# ============================================================================================================
C_SMALL, C_LARGE = "#ff7f0e", "#2ca02c"                   # same model colours as the round-01 README
ROOT = HERE.parents[1]
CUB = ROOT / "CUB_200_2011"
FACTS = []


def fact(s):
    FACTS.append(s)
    print("  fact:", s)


def final_run(df):
    return df.loc[df.val_top1.idxmax(), "run"]


def load_final(model, run):
    d = run_dir(model, run)
    return {"model": model, "run": run, "name": MODELS[model].split(" (")[0],
            "row": json.loads((d / "test_metrics.json").read_text())["row"], "hist": pd.read_csv(d / "history.csv"),
            "pc": pd.read_csv(d / "test_per_class.csv"), "preds": pd.read_csv(d / "test_predictions.csv"),
            "rel": pd.read_csv(d / "test_reliability.csv"), "conf": np.load(d / "test_confusion_matrix.npy")}


def mcnemar_pair(pa, pb):
    pb = pb.set_index("image_id").loc[pa.image_id].reset_index()
    ca, cb = (pa.pred == pa.label).values, (pb.pred == pb.label).values
    n10, n01 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    n, k = n10 + n01, min(n10, n01)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return n10, n01, p, int((ca & cb).sum()), int((~ca & ~cb).sum())


def cmp_results(S, L, old):
    """Round 01 vs round 02 final, for both models: top-1, gap, ECE."""
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.6))
    groups = [("CNN small", old["cnn_small"], S["row"], C_SMALL), ("CNN large", old["cnn_large"], L["row"], C_LARGE)]
    for a, key, title, fmt in ((ax[0], "test_acc1", "Test top-1, %  (higher is better)", "{:.2f}"),
                               (ax[1], "overfit_gap_clean_minus_val", "Overfitting gap, points  (lower is better)",
                                "{:.1f}"),
                               (ax[2], "test_ece", "Calibration error (ECE), %  (lower is better)", "{:.1f}")):
        for gi, (nm, o, n, col) in enumerate(groups):
            for j, (v, alpha, hatch) in enumerate(((o[key], 0.35, "//"), (n[key], 1.0, None))):
                x = gi * 1.0 + (j - 0.5) * 0.38
                a.bar(x, v, 0.36, color=col, alpha=alpha, hatch=hatch, edgecolor="white", linewidth=2)
                a.text(x, v * 1.01, fmt.format(v), ha="center", va="bottom", fontsize=9.5, color=INK,
                       fontweight="bold" if j else "normal")
        a.set_xticks([-0.19, 0.19, 0.81, 1.19], ["round 01", "round 02\nfinal"] * 2, fontsize=8.5)
        for gi, (nm, *_ , col) in enumerate(groups):
            a.text(gi, -0.2, nm, transform=a.get_xaxis_transform(), ha="center", fontsize=10, color=col,
                   fontweight="bold")
        a.set_ylim(0, max(max(o[key], n[key]) for _, o, n, _ in groups) * 1.18)
        a.set_title(title, loc="left")
        grid(a, "y")
    header(fig, "Round 01 vs round 02, both CNNs",
           f"round 02 final = best validation run: CNN small {RUNS[S['run']][0]}, CNN large {RUNS[L['run']][0]}. "
           "Hatched = round 01 (strong aug + Mixup/CutMix)", top=0.88)
    savefig(fig, "compare_01_round01_vs_round02.png")


def cmp_effects(dfs):
    """Change vs r0 for every run, CNN small next to CNN large: does each change help both models the same way?"""
    runs = [r for r in RUNS if r != "r0_reference"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
    y = np.arange(len(runs))[::-1]
    for a, col, title in ((ax[0], "val_top1", "Validation top-1 vs r0 (600 photos): decides"),
                          (ax[1], "test_top1", "Test top-1 vs r0 (5,794 photos): for the record")):
        a.axvline(0, color=INK, lw=0.9)
        for off, model, c in ((0.19, "cnn_small", C_SMALL), (-0.19, "cnn_large", C_LARGE)):
            d = dfs[model].set_index("run")
            ref = d.loc["r0_reference", col]
            for yi, r in zip(y, runs):
                if r not in d.index:
                    a.text(0.1, yi + off, "pending", fontsize=8, color=MUTED, style="italic", va="center")
                    continue
                v = d.loc[r, col] - ref
                a.barh(yi + off, v, 0.36, color=c, edgecolor="white", linewidth=1.5,
                       label=MODELS[model] if yi == y[0] else None)
                a.text(v + (0.12 if v >= 0 else -0.12), yi + off, f"{v:+.1f}", va="center", fontsize=8,
                       ha="left" if v >= 0 else "right", color=INK)
        a.set_title(title, loc="left")
        a.set_xlabel("change vs that model's own r0, points")
        a.set_xlim(-5.2, 6)
        grid(a, "x")
    ax[0].set_yticks(y, [RUNS[r][0] for r in runs])
    ax[0].legend(frameon=False, fontsize=9, loc="upper left")
    header(fig, "Does each change help both CNNs the same way?",
           "each bar = one run minus the same model's r0 reference. Noise: a few validation photos = 0.5 points", top=0.89)
    savefig(fig, "compare_02_effect_of_each_change.png")


def cmp_curves(S, L, R0):
    fig, ax = plt.subplots(2, 3, figsize=(17, 8.6))
    for r, col, ls, lw in ((R0["cnn_small"], C_SMALL, ":", 1.4), (R0["cnn_large"], C_LARGE, ":", 1.4),
                           (S, C_SMALL, "-", 2.0), (L, C_LARGE, "-", 2.0)):
        h, lab = r["hist"], f"{r['name']} {RUNS[r['run']][0]}"
        e = h.epoch
        ax[0, 0].plot(e, h.val_loss, color=col, lw=lw, ls=ls, label=lab)
        ax[0, 1].plot(e, h.val_acc1.rolling(9, center=True, min_periods=1).mean(), color=col, lw=lw, ls=ls, label=lab)
        if ls == "-":
            be = int(r["row"]["best_epoch"])
            ax[0, 1].scatter([be], [h.loc[h.epoch == be, "val_acc1"].iloc[0]], color=col, s=60, edgecolor="white",
                             linewidth=1.5, zorder=4)
        c = h.dropna(subset=["clean_acc1"])
        ax[0, 2].plot(c.epoch, c.clean_acc1, color=col, lw=lw, ls=ls, label=lab)
        ax[1, 0].plot(c.epoch, c.clean_acc1 - c.val_acc1, color=col, lw=lw, ls=ls, label=lab)
        ax[1, 1].plot(e, h.train_loss, color=col, lw=lw, ls=ls, label=lab)
        ax[1, 2].plot(e, (h.val_acc1 - h.valraw_acc1).rolling(9, center=True, min_periods=1).mean(), color=col,
                      lw=lw, ls=ls, label=lab)
    titles = ["Validation loss", "Validation top-1, 9-epoch smoothing (dot = chosen epoch)",
              "Training photos, no augmentation (top-1)", "Memorizing gap: clean train minus validation (points)",
              "Training loss (augmented photos)", "EMA weights minus latest weights, validation top-1 (points)"]
    for a, t in zip(ax.ravel(), titles):
        a.set_title(t, loc="left", fontsize=10.5)
        a.set_xlabel("epoch")
        grid(a, "both")
    ax[1, 2].axhline(0, color=INK, lw=0.8)
    ax[0, 1].legend(frameon=False, fontsize=8.5, loc="lower right")
    header(fig, "How training went: CNN small vs CNN large",
           "solid = each model's final round-02 run, dotted = its r0 reference. 300 epochs, seed 42", top=0.92)
    savefig(fig, "compare_03_training_curves.png")
    for r in (S, L):
        h = r["hist"]
        sub = h[h.epoch >= 50]
        fact(f"{r['name']} {r['run']}: lowest val loss {h.val_loss.min():.2f} @ epoch {int(h.loc[h.val_loss.idxmin(), 'epoch'])}, "
             f"final {h.val_loss.iloc[-1]:.2f}; best epoch {r['row']['best_epoch']}; clean train at end "
             f"{h.clean_acc1.dropna().iloc[-1]:.1f}; EMA-raw mean ep>=50 {(sub.val_acc1 - sub.valraw_acc1).mean():+.2f}")


def cmp_calibration(S, L):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    a = ax[0]
    a.plot([0, 1], [0, 1], color="#bbbbbb", lw=1, ls="--")
    for r, col in ((S, C_SMALL), (L, C_LARGE)):
        rel = r["rel"][r["rel"].fraction > 0.002]
        a.plot(rel.confidence, rel.accuracy, "-o", color=col, lw=2, ms=5,
               label=f"{r['name']} {RUNS[r['run']][0]} (ECE {r['row']['test_ece']:.1f}%)")
    a.text(0.62, 0.15, "below the line = over-confident", fontsize=8.5, color=MUTED)
    a.text(0.03, 0.85, "above the line =\nunder-confident", fontsize=8.5, color=MUTED)
    a.set_xlabel("model's confidence")
    a.set_ylabel("actual accuracy")
    a.set_title("Reliability of the final models", loc="left")
    a.legend(frameon=False, fontsize=8.5, loc="lower right")
    grid(a, "both")
    a = ax[1]
    for r, col, off in ((S, C_SMALL, -0.2), (L, C_LARGE, 0.2)):
        a.bar(np.arange(len(r["rel"])) + off, r["rel"].fraction * 100, 0.4, color=col, edgecolor="white",
              label=r["name"])
    a.set_xlabel("confidence bin (low to high)")
    a.set_ylabel("share of test photos, %")
    a.set_title("How many photos fall in each confidence bin", loc="left")
    a.legend(frameon=False, fontsize=9)
    grid(a, "y")
    header(fig, "Are the confidence numbers honest? CNN small vs CNN large",
           "the final runs; label smoothing 0.2 (in CNN small's r8) makes a model under-confident")
    savefig(fig, "compare_04_calibration.png")


def cmp_species(S, L):
    a_, b_ = S["pc"].set_index("class_name").f1, L["pc"].set_index("class_name").f1
    d = (b_ - a_).sort_values()
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.25]})
    a = ax[0]
    a.scatter(a_, b_, s=18, color=[C_LARGE if v > 0 else C_SMALL for v in (b_ - a_)], alpha=0.8, edgecolor="white",
              linewidth=0.5)
    a.plot([0, 100], [0, 100], color="#999999", lw=1, ls="--")
    for name in list(d.index[:3]) + list(d.index[-3:]):
        a.annotate(str(name), (a_[name], b_[name]), xytext=(4, 2), textcoords="offset points", fontsize=7.5,
                   color=INK)
    a.text(3, 95, "above the line: CNN large better", fontsize=8.5, color=C_LARGE, va="top")
    a.text(97, 3, "below the line: CNN small better", fontsize=8.5, color=C_SMALL, ha="right")
    a.set_xlim(0, 100)
    a.set_ylim(0, 100)
    a.set_xlabel("CNN small: test F1 per species")
    a.set_ylabel("CNN large: test F1 per species")
    a.set_title(f"CNN large better on {int((d > 0).sum())} species, CNN small on {int((d < 0).sum())}", loc="left")
    grid(a, "both")
    a = ax[1]
    ps, pl = S["pc"].sort_values("f1"), L["pc"].sort_values("f1")
    rows = [("hardest", ps.head(6), pl.head(6)), ("easiest", ps.tail(6)[::-1], pl.tail(6)[::-1])]
    a.axis("off")
    yy = 0.97
    for title, s6, l6 in rows:
        a.text(0.0, yy, f"6 {title} species", fontsize=11, fontweight="bold", transform=a.transAxes, va="top")
        a.text(0.0, yy - 0.06, "CNN small", fontsize=9.5, color=C_SMALL, fontweight="bold", transform=a.transAxes,
               va="top")
        a.text(0.52, yy - 0.06, "CNN large", fontsize=9.5, color=C_LARGE, fontweight="bold", transform=a.transAxes,
               va="top")
        for i, (rs, rl) in enumerate(zip(s6.itertuples(), l6.itertuples())):
            a.text(0.0, yy - 0.12 - i * 0.05, f"{rs.class_name[:28]}  ({rs.f1:.0f})", fontsize=9, transform=a.transAxes,
                   va="top")
            a.text(0.52, yy - 0.12 - i * 0.05, f"{rl.class_name[:28]}  ({rl.f1:.0f})", fontsize=9,
                   transform=a.transAxes, va="top")
        yy -= 0.5
    header(fig, "Which species are easy and which are hard? CNN small vs CNN large (final runs)",
           "test F1 per species (30 test photos each, so one species' F1 is noisy)", top=0.9)
    savefig(fig, "compare_05_species.png")
    fact(f"large better on {int((d > 0).sum())} species, small better on {int((d < 0).sum())}, tied {int((d == 0).sum())}; "
         f"median large-small {d.median():+.1f}")
    fact("biggest large wins: " + ", ".join(f"{n} ({d[n]:+.0f})" for n in d.index[-3:][::-1]))
    fact("biggest small wins: " + ", ".join(f"{n} ({d[n]:+.0f})" for n in d.index[:3]))
    for r, p in ((S, ps), (L, pl)):
        fact(f"{r['name']} hardest 5: " + ", ".join(f"{x.class_name} ({x.f1:.0f})" for x in p.head(5).itertuples()))
        fact(f"{r['name']} easiest 5: " + ", ".join(f"{x.class_name} ({x.f1:.0f})" for x in p.tail(5)[::-1].itertuples()))


def cmp_confusions(S, L):
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.2))
    for a, r, col in ((ax[0], S, C_SMALL), (ax[1], L, C_LARGE)):
        conf = r["conf"].copy()
        np.fill_diagonal(conf, 0)
        names = r["pc"].class_name.tolist()
        top = np.dstack(np.unravel_index(np.argsort(-conf.ravel())[:10], conf.shape))[0]
        labs = [f"{names[i]}  ->  {names[j]}" for i, j in top]
        vals = [int(conf[i, j]) for i, j in top]
        a.barh(labs[::-1], vals[::-1], color=col, edgecolor="white", linewidth=1.5)
        for yi, v in enumerate(vals[::-1]):
            a.text(v + 0.2, yi, str(v), va="center", fontsize=8.5, color=INK)
        a.set_title(f"{r['name']} ({RUNS[r['run']][0]}): 10 most frequent mix-ups", loc="left", fontsize=10.5)
        a.set_xlabel("test photos (true -> predicted)")
        a.tick_params(axis="y", labelsize=8)
        grid(a, "x")
        fact(f"{r['name']} top mix-ups: " + "; ".join(f"{names[i]} -> {names[j]} ({int(conf[i, j])})" for i, j in top[:5]))
    header(fig, "What do the models confuse?", "30 test photos per species; almost every mix-up is between look-alike species "
           "(crows, terns, flycatchers, gulls, nightjars)",
           top=0.88)
    savefig(fig, "compare_06_confusions.png")


def _samples(S, L):
    test = pd.read_csv(ROOT / "DataPreprocessing_Results" / "test.csv").set_index("image_id")
    ids = S["preds"].image_id.values
    lab = S["preds"].label.values
    pr = {"CNN small": S["preds"].set_index("image_id").loc[ids, "pred"].values,
          "CNN large": L["preds"].set_index("image_id").loc[ids, "pred"].values}
    names = S["pc"].set_index("class_id").class_name.to_dict()
    return ids, lab, pr, names, test


def _img(test, iid, size=300):
    from PIL import Image
    fp = str(test.loc[iid, "filepath"]).replace("\\", "/")
    im = Image.open(CUB / fp.split("CUB_200_2011/", 1)[1]).convert("RGB")
    im.thumbnail((size, size))
    return im


def cmp_samples(S, L):
    ids, lab, pr, names, test = _samples(S, L)
    idx = np.linspace(0, len(ids) - 1, 12).astype(int)
    fig, axes = plt.subplots(2, 6, figsize=(19, 8.4))
    for a, i in zip(axes.ravel(), idx):
        a.axis("off")
        a.imshow(_img(test, ids[i]))
        a.set_title("truth: " + names[lab[i]][:24], fontsize=8.5, fontweight="bold")
        for j, k in enumerate(pr):
            ok = pr[k][i] == lab[i]
            a.text(0, -0.05 - 0.10 * j, ("OK  " if ok else "X   ") + k + ": " + names[pr[k][i]][:20], transform=a.transAxes,
                   fontsize=8, color="#1a7f37" if ok else "#c0392b", va="top", fontweight="bold")
    fig.suptitle("12 test photos spread evenly through the test set (not hand-picked), final round-02 models: "
                 "green = correct, red = wrong", fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(fig, "compare_07_samples.png")
    ok_s, ok_l = pr["CNN small"][idx] == lab[idx], pr["CNN large"][idx] == lab[idx]
    fact(f"of the 12 sample photos: small right on {int(ok_s.sum())}, large right on {int(ok_l.sum())}")


def cmp_race_gif(S, L, R0):
    series = [(R0["cnn_small"], C_SMALL, ":"), (R0["cnn_large"], C_LARGE, ":"), (S, C_SMALL, "-"), (L, C_LARGE, "-")]
    sm = [(r, c, ls, r["hist"].set_index("epoch").val_acc1.rolling(9, center=True, min_periods=1).mean(),
           r["hist"].set_index("epoch").clean_acc1.dropna()) for r, c, ls in series]
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
    for r, c, ls, v, cl in sm:
        lw = 2.4 if ls == "-" else 1.5
        l1, = ax[0].plot([], [], color=c, lw=lw, ls=ls, label=f"{r['name']} {RUNS[r['run']][0]}")
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
        ttl.set_text(f"Round 02 race, epoch {e}/300: solid = final run, dotted = r0 reference")
        return []

    FuncAnimation(fig, update, frames=frames, blit=False).save(ANI / "compare_race.gif", writer=PillowWriter(fps=12),
                                                               dpi=80)
    plt.close(fig)
    print("wrote", ANI / "compare_race.gif")


def cmp_reveal_gif(S, L):
    ids, lab, pr, names, test = _samples(S, L)
    idx = np.linspace(0, len(ids) - 1, 12).astype(int)
    fig = plt.figure(figsize=(10, 4.2))
    ax_i = fig.add_axes([0.02, 0.05, 0.40, 0.82])
    ax_t = fig.add_axes([0.45, 0.05, 0.53, 0.82])

    def draw(f):
        i = idx[f]
        ax_i.clear()
        ax_t.clear()
        ax_i.imshow(_img(test, ids[i], 360))
        ax_i.axis("off")
        ax_t.axis("off")
        ax_t.set_xlim(0, 1)
        ax_t.set_ylim(0, 1)
        ax_t.text(0, 0.95, f"Photo {f + 1} of {len(idx)}", fontsize=10, color=MUTED, va="top")
        ax_t.text(0, 0.82, "Truth:  " + names[lab[i]], fontsize=13, fontweight="bold", va="top")
        for j, k in enumerate(pr):
            ok = pr[k][i] == lab[i]
            ax_t.text(0, 0.60 - 0.18 * j, f"{'OK ' if ok else 'X  '}{k}: {names[pr[k][i]]}", fontsize=12, va="top",
                      color="#1a7f37" if ok else "#c0392b", fontweight="bold")
        done = idx[: f + 1]
        ax_t.text(0, 0.06, "Running score  " + "   ".join(f"{k}: {int((pr[k][done] == lab[done]).sum())}/{len(done)}"
                                                         for k in pr), fontsize=10)
        return []

    FuncAnimation(fig, draw, frames=len(idx), blit=False).save(ANI / "compare_prediction_reveal.gif",
                                                               writer=PillowWriter(fps=0.8), dpi=80)
    plt.close(fig)
    print("wrote", ANI / "compare_prediction_reveal.gif")


def compare(dfs):
    S = load_final("cnn_small", final_run(dfs["cnn_small"]))
    L = load_final("cnn_large", final_run(dfs["cnn_large"]))
    R0 = {m: load_final(m, "r0_reference") for m in MODELS}
    old = {m: json.loads((R01 / m / "seed42" / "test_metrics.json").read_text())["row"] for m in MODELS}
    fact(f"final runs: small {S['run']}, large {L['run']}")
    for r in (S, L):
        w = r["row"]
        fact(f"{r['name']} {r['run']}: val {w['val_acc1_at_best']:.2f} test {w['test_acc1']:.2f} top5 {w['test_acc5']:.2f} "
             f"f1 {w['test_macro_f1']:.2f} gap {w['overfit_gap_clean_minus_val']:.1f} ece {w['test_ece']:.2f} "
             f"best epoch {w['best_epoch']} train {w['train_time_s'] / 60:.1f} min {w['sec_per_epoch']:.1f} s/ep")
    n10, n01, p, both_r, both_w = mcnemar_pair(L["preds"], S["preds"])
    fact(f"final large vs final small: only large right {n10}, only small right {n01}, p={p:.2e}, both right {both_r}, "
         f"both wrong {both_w}")
    for m in MODELS:
        a, b = R0[m]["preds"], old[m]
        o = pd.read_csv(R01 / m / "seed42" / "test_predictions.csv")
        f = S if m == "cnn_small" else L
        x = mcnemar_pair(f["preds"], o)
        y = mcnemar_pair(a, o)
        fact(f"{m}: final vs round01 n10 {x[0]} n01 {x[1]} p {x[2]:.1e}; r0 vs round01 n10 {y[0]} n01 {y[1]} p {y[2]:.2f}; "
             f"round01 test {b['test_acc1']:.2f} top5 {b['test_acc5']:.2f} f1 {b['test_macro_f1']:.2f} "
             f"gap {b['overfit_gap_clean_minus_val']:.1f} ece {b['test_ece']:.2f} val {b['val_acc1_at_best']:.2f}")
    n10, n01, p, *_ = mcnemar_pair(R0["cnn_large"]["preds"], R0["cnn_small"]["preds"])
    fact(f"r0 large vs r0 small: {n10} vs {n01}, p={p:.2e}")
    cmp_results(S, L, old)
    cmp_effects(dfs)
    cmp_curves(S, L, R0)
    cmp_calibration(S, L)
    cmp_species(S, L)
    cmp_confusions(S, L)
    cmp_samples(S, L)
    cmp_race_gif(S, L, R0)
    cmp_reveal_gif(S, L)
    (HERE / "docs" / "facts.txt").write_text("\n".join(FACTS), encoding="utf-8")


if __name__ == "__main__":
    dfs = {}
    for model in MODELS:
        df, hist = load(model)
        if df.empty or "r0_reference" not in df.run.values:
            print(f"{model}: no finished runs yet - skipped")
            continue
        dfs[model] = df
        fig_overview(model, df)
        fig_gap(model, df)
        fig_dose(model, df)
        fig_curves(model, df, hist)
        fig_calibration(model, df)
        old = fig_before_after(model, df)
        print("improvement vs round 01:", fig_improvement(model, df))
        gif_race(model, hist)
        tables(model, df, old)
    if len(dfs) == len(MODELS):
        compare(dfs)
