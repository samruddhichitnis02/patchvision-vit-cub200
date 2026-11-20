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
    offset = {"r1_droppath0.2": (8, -15, "left"), "r4_labelsmooth0.2": (8, 8, "left"),
              "r5_wd0.1": (-8, 8, "right")}                     # hand-placed so close points never overlap
    for r in df.itertuples():
        ax.scatter(r.gap, r.val_top1, s=150, color=RUNS[r.run][2], edgecolor="white", linewidth=2, zorder=3)
        dx, dy, ha = offset.get(r.run, (8, 6, "left"))
        ax.annotate(r.label, (r.gap, r.val_top1), xytext=(dx, dy), textcoords="offset points", fontsize=9,
                    color=INK, ha=ha)
    ax.fill_between([df.gap.min() - 1, ref.gap], ref.val_top1, 52, color="#2ca02c", alpha=0.07, lw=0)
    ax.text(df.gap.min() - 0.9, 51.4, "goal: up and to the left\n(more accurate AND less memorizing)", fontsize=9,
            color="#2a7a2a", va="top")
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
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.3))
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


if __name__ == "__main__":
    for model in MODELS:
        df, hist = load(model)
        if df.empty or "r0_reference" not in df.run.values:
            print(f"{model}: no finished runs yet - skipped")
            continue
        fig_overview(model, df)
        fig_gap(model, df)
        fig_dose(model, df)
        fig_curves(model, df, hist)
        fig_calibration(model, df)
        old = fig_before_after(model, df)
        gif_race(model, hist)
        tables(model, df, old)
