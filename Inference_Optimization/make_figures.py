"""Figures and animations for the inference-optimization study (README section 10).

Reads only the CSV files written by the step scripts, so it can be re-run at any time:
    python Inference_Optimization/make_figures.py
Writes Inference_Optimization/docs/figures/*.png and docs/animations/*.gif

Colours follow the rest of the repository (assets/make_assets.py): every model keeps its colour in every figure.
Identity is never colour-only: each model also has its own marker and a direct label.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
S1 = HERE / "results" / "step1_baseline"
FIG = HERE / "docs" / "figures"
ANI = HERE / "docs" / "animations"
FIG.mkdir(parents=True, exist_ok=True)
ANI.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID = "#222222", "#666666", "#e6e6e6"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100,
                     "axes.edgecolor": "#999999", "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.titleweight": "bold", "axes.titlesize": 11.5})

MODELS = ["ViT base", "PatchVision-X", "CNN small", "CNN large"]
COLOR = {"ViT base": "#8c8c8c", "PatchVision-X": "#1f77b4", "CNN small": "#ff7f0e", "CNN large": "#2ca02c"}
MARKER = {"ViT base": "s", "PatchVision-X": "o", "CNN small": "^", "CNN large": "D"}
PARAMS = {"ViT base": "4.31 M", "PatchVision-X": "4.27 M", "CNN small": "4.22 M", "CNN large": "11.28 M"}
GPU_BUSY, GPU_IDLE = "#1f77b4", "#d9d9d9"


def savefig(fig, name):
    fig.savefig(FIG / name, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", FIG / name)


def header(fig, title, sub, top=0.9):
    """Figure title + muted subtitle on separate lines (never overlapping); the plots sit below `top`."""
    fig.suptitle(title, x=0.01, y=0.995, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.925, sub, fontsize=9, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0, 1, top))


def grid(a, axis="y"):
    a.grid(axis=axis, color=GRID, lw=0.8)
    a.set_axisbelow(True)


speed = pd.read_csv(S1 / "speed.csv")
acc = pd.read_csv(S1 / "accuracy.csv")
prof = pd.read_csv(S1 / "profile.csv")
rep = pd.read_csv(S1 / "repeatability_bs1.csv")
env = json.loads((S1 / "env.json").read_text())
lat1 = rep.groupby("model")["latency_ms_median"].median()            # robust batch-1 latency (5 repeats)
lat1_lo = rep.groupby("model")["latency_ms_median"].min()
lat1_hi = rep.groupby("model")["latency_ms_median"].max()
p95_1 = rep.groupby("model")["latency_ms_p95"].median()
thr256 = speed[speed.batch_size == 256].set_index("model")["throughput_img_s"]
GPU_NAME = env["gpu"].replace("NVIDIA GeForce ", "")
SUB = f"{GPU_NAME} | PyTorch {env['torch'].split('+')[0]} eager FP32 | same machine, same settings for all models"


# ------------------------------------------------------------------------------------------------------------
# 1. headline: batch-1 latency and batch-256 throughput
# ------------------------------------------------------------------------------------------------------------
def fig_overview():
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
    y = np.arange(len(MODELS))[::-1]
    a = ax[0]
    v = [lat1[m] for m in MODELS]
    a.barh(y, v, color=[COLOR[m] for m in MODELS], height=0.62, edgecolor="white", linewidth=2)
    a.errorbar(v, y, xerr=[[lat1[m] - lat1_lo[m] for m in MODELS], [lat1_hi[m] - lat1[m] for m in MODELS]],
               fmt="none", ecolor=INK, capsize=3, lw=1)
    for yi, m in zip(y, MODELS):
        a.text(lat1[m] + 0.25, yi, f"{lat1[m]:.2f} ms   (p95 {p95_1[m]:.2f})", va="center", color=INK, fontsize=9.5)
    a.set_yticks(y, [f"{m}\n{PARAMS[m]} params" for m in MODELS])
    a.set_xlim(0, max(v) * 1.55)
    a.set_xlabel("milliseconds per photo  (lower is better)")
    a.set_title("Latency: one photo at a time (batch 1)", loc="left")
    ratio = lat1["PatchVision-X"] / lat1["CNN small"]
    a.annotate(f"PatchVision-X is {ratio:.1f}x slower\nthan the same-size CNN", xy=(lat1["PatchVision-X"], y[1]),
               xytext=(lat1["PatchVision-X"] * 1.05, y[2] - 0.05), fontsize=9.5, color=INK,
               arrowprops=dict(arrowstyle="->", color=MUTED))
    grid(a, "x")

    a = ax[1]
    v = [thr256[m] for m in MODELS]
    a.barh(y, v, color=[COLOR[m] for m in MODELS], height=0.62, edgecolor="white", linewidth=2)
    for yi, m in zip(y, MODELS):
        a.text(thr256[m] + 250, yi, f"{thr256[m]:,.0f} photos/s", va="center", color=INK, fontsize=9.5)
    a.set_yticks(y, [""] * len(MODELS))
    a.set_xlim(0, max(v) * 1.35)
    a.set_xlabel("photos per second  (higher is better)")
    a.set_title("Throughput: large batches (batch 256)", loc="left")
    ratio = thr256["CNN small"] / thr256["PatchVision-X"]
    a.text(0.98, 0.97, f"CNN small vs PatchVision-X: the gap shrinks\nto {ratio:.1f}x when the GPU is kept busy",
           transform=a.transAxes, ha="right", va="top", fontsize=9.5, color=INK)
    grid(a, "x")
    header(fig, "Step 1 baseline: how fast is each model?",
           SUB + "   (batch-1 bars: median of 5 repeats, whisker = min-max)")
    savefig(fig, "s1_01_speed_overview.png")


# ------------------------------------------------------------------------------------------------------------
# 2. batch-size sweep: latency and throughput (two charts, no dual axis)
# ------------------------------------------------------------------------------------------------------------
def fig_sweep():
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for m in MODELS:
        d = speed[speed.model == m].sort_values("batch_size")
        kw = dict(color=COLOR[m], marker=MARKER[m], lw=2, ms=7, markeredgecolor="white", markeredgewidth=1.2)
        ax[0].plot(d.batch_size, d.latency_ms_median, label=m, **kw)
        ax[1].plot(d.batch_size, d.throughput_img_s, label=m, **kw)
        nudge = {"ViT base": 500, "PatchVision-X": -500}.get(m, 0)
        ax[1].text(d.batch_size.iloc[-1] * 1.12, d.throughput_img_s.iloc[-1] + nudge, m, va="center", fontsize=9,
                   color=INK)
    for a in ax:
        a.set_xscale("log", base=2)
        a.set_xticks([1, 8, 32, 64, 128, 256], ["1", "8", "32", "64", "128", "256"])
        a.set_xlabel("batch size (photos per call)")
        grid(a, "both")
    ax[0].set_yscale("log")
    ax[0].minorticks_off()
    ax[0].set_yticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
    ax[0].set_ylabel("milliseconds per call (log scale)")
    ax[0].set_title("Latency per call", loc="left")
    pv = speed[speed.model == "PatchVision-X"].set_index("batch_size")["latency_ms_median"]
    ax[0].axvspan(1, 32, color="#1f77b4", alpha=0.07, lw=0)
    ax[0].text(1.15, pv.loc[256] * 0.92, f"ViTs: 1 photo = {pv.loc[1]:.1f} ms,\n32 photos = {pv.loc[32]:.1f} ms\n"
               "-> the GPU is waiting,\n    not computing", fontsize=9, color=INK, va="top")
    ax[0].legend(frameon=False, loc="lower right", fontsize=9)
    ax[1].set_ylabel("photos per second")
    ax[1].set_title("Throughput", loc="left")
    ax[1].set_xlim(right=256 * 3.2)
    header(fig, "Step 1 baseline: what happens as the batch grows", SUB)
    savefig(fig, "s1_02_batch_sweep.png")


# ------------------------------------------------------------------------------------------------------------
# 3. where the time goes: GPU computing vs GPU idle (profiler)
# ------------------------------------------------------------------------------------------------------------
def fig_where_time_goes():
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    a = ax[0]
    y = np.arange(len(MODELS))[::-1]
    p1 = prof[prof.batch_size == 1].set_index("model")
    for yi, m in zip(y, MODELS):
        busy = p1.loc[m, "gpu_kernel_ms_per_forward"]
        total = lat1[m]
        idle = max(0.0, total - busy)
        a.barh(yi, busy, color=GPU_BUSY, height=0.6, edgecolor="white", linewidth=2)
        a.barh(yi, idle, left=busy, color=GPU_IDLE, height=0.6, edgecolor="white", linewidth=2, hatch="///")
        a.text(total + 0.2, yi, f"GPU busy {busy:.1f} of {total:.1f} ms  ({busy / total * 100:.0f}%)", va="center",
               fontsize=9.5, color=INK)
    a.set_yticks(y, MODELS)
    a.set_xlim(0, lat1.max() * 1.75)
    a.set_xlabel("milliseconds per photo (batch 1)")
    a.set_title("Where one batch-1 forward pass spends its time", loc="left")
    a.legend(handles=[Rectangle((0, 0), 1, 1, color=GPU_BUSY), Rectangle((0, 0), 1, 1, fc=GPU_IDLE, hatch="///",
                                                                            ec="white")],
             labels=["GPU computing (sum of kernel times)", "GPU idle: waiting for the CPU to launch work"],
             frameon=False, loc="lower right", fontsize=9)
    grid(a, "x")

    a = ax[1]
    k = [p1.loc[m, "kernels_per_forward"] for m in MODELS]
    a.barh(y, k, color=[COLOR[m] for m in MODELS], height=0.6, edgecolor="white", linewidth=2)
    for yi, m, kv in zip(y, MODELS, k):
        a.text(kv + 12, yi, f"{kv:.0f}", va="center", fontsize=9.5, color=INK)
    a.set_yticks(y, [""] * len(MODELS))
    a.set_xlim(0, max(k) * 1.2)
    a.set_xlabel("GPU kernels launched per forward pass")
    a.set_title("...because the ViTs launch 5x more kernels", loc="left")
    grid(a, "x")
    header(fig, "Step 1 diagnosis: the ViTs are launch-bound, not compute-bound",
           "torch.profiler, 30 forward passes averaged; latency = median of 5 timing repeats")
    savefig(fig, "s1_03_where_time_goes.png")


# ------------------------------------------------------------------------------------------------------------
# 4. accuracy reproduction check
# ------------------------------------------------------------------------------------------------------------
def fig_accuracy_check():
    fig, ax = plt.subplots(figsize=(12.5, 4.3))
    d = acc.copy()
    x = np.arange(len(d))
    one_photo = 100 / 5794
    ax.axhspan(-10 * one_photo, 10 * one_photo, color="#2ca02c", alpha=0.08, lw=0)
    ax.axhline(0, color=INK, lw=1)
    ax.text(len(d) - 0.45, 10 * one_photo * 0.95, "band = within 10 photos of 5,794", ha="right", va="top",
            fontsize=9, color=MUTED)
    ax.scatter(x - 0.12, d.bf16amp_vs_recorded_top1, s=70, marker="o", color="#1f77b4", edgecolor="white",
               zorder=3, label="bf16 autocast (how training evaluated)")
    ax.scatter(x + 0.12, d.fp32_vs_recorded_top1, s=70, marker="D", color="#9467bd", edgecolor="white", zorder=3,
               label="fp32 (reference for all later steps)")
    for xi, r in zip(x, d.itertuples()):
        ax.text(xi, -0.205, f"{r.recorded_top1:.2f}%", ha="center", fontsize=8.5, color=MUTED)
    ax.set_xticks(x, [n.replace("_", "\n", 1) for n in d.run], fontsize=8.5)
    ax.set_ylim(-0.23, 0.23)
    ax.set_ylabel("re-measured minus recorded\ntest top-1 (points)")
    ax.legend(frameon=False, loc="upper left", fontsize=9, ncol=2)
    grid(ax, "y")
    big = d[["bf16amp_vs_recorded_top1", "fp32_vs_recorded_top1"]].abs().max().max()
    header(fig, "Step 1 sanity check: all 10 trained models reproduce their recorded test accuracy",
           f"grey numbers = recorded top-1.  1 photo = 0.017 points.  Largest difference: {big:.2f} points "
           f"({big / one_photo:.0f} photos)", top=0.9)
    savefig(fig, "s1_04_accuracy_check.png")


# ------------------------------------------------------------------------------------------------------------
# 5. accuracy vs latency: the starting point of the optimization story
# ------------------------------------------------------------------------------------------------------------
def fig_tradeoff():
    a1 = acc.set_index("run")["fp32_top1"]
    pts = [("ViT base", a1["base"], "ViT base"), ("PatchVision-X", a1["full"], "PatchVision-X (full, 47.5%)"),
           ("PatchVision-X", a1["ovf5_dp0.3_autoaug"], "PatchVision-X (E4 best, 53.0%)"),
           ("CNN small", a1["cnn_small"], "CNN small"), ("CNN large", a1["cnn_large"], "CNN large")]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for m, acc1, lab in pts:
        hollow = "full" in lab
        ax.scatter(lat1[m], acc1, s=160, marker=MARKER[m], zorder=3, linewidths=2,
                   facecolor="white" if hollow else COLOR[m], edgecolor=COLOR[m])
        ax.text(lat1[m] * 1.07, acc1, lab, va="center", fontsize=9.5, color=INK)
    best = a1["ovf5_dp0.3_autoaug"]
    ax.annotate("", xy=(lat1["CNN small"] * 1.05, best), xytext=(lat1["PatchVision-X"] * 0.93, best),
                arrowprops=dict(arrowstyle="-|>", color="#1f77b4", lw=2, ls="--"))
    ax.text(np.sqrt(lat1["CNN small"] * lat1["PatchVision-X"]), best + 0.7,
            "the goal of steps 2-7:\nmove PatchVision-X left, keep it up", ha="center", fontsize=9.5,
            color="#1f77b4", fontweight="bold")
    ax.axvspan(0.5, lat1["CNN large"] * 1.2, color="#2ca02c", alpha=0.06, lw=0)
    ax.text(0.62, 37.2, "CNN speed zone", fontsize=9, color=MUTED)
    ax.set_xscale("log")
    ax.set_xlim(0.6, 25)
    ax.set_xticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
    ax.set_ylim(36, 56)
    ax.set_xlabel("batch-1 latency, ms (log scale, lower is better)")
    ax.set_ylabel("test top-1 accuracy, % (higher is better)")
    grid(ax, "both")
    header(fig, "Step 1: accuracy vs speed, before any optimization",
           "hollow marker = same architecture trained with the original recipe (identical speed)", top=0.9)
    savefig(fig, "s1_05_accuracy_vs_latency.png")


# ------------------------------------------------------------------------------------------------------------
# GIF 1: one-second race at batch 1
# ------------------------------------------------------------------------------------------------------------
def gif_race():
    per_s = {m: 1000.0 / lat1[m] for m in MODELS}
    frames, fps = 60, 15
    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = np.arange(len(MODELS))[::-1]
    bars = ax.barh(y, [0] * len(MODELS), color=[COLOR[m] for m in MODELS], height=0.6, edgecolor="white", lw=2)
    labels = [ax.text(0, yi, "", va="center", fontsize=10, color=INK) for yi in y]
    clock = ax.text(0.99, 1.02, "", transform=ax.transAxes, ha="right", va="bottom", fontsize=11, color=INK,
                    fontweight="bold")
    ax.set_yticks(y, MODELS)
    ax.set_xlim(0, max(per_s.values()) * 1.3)
    ax.set_xlabel("photos classified so far (batch 1, one photo per call)")
    ax.set_title("One second on the RTX 5090: who classifies more birds?", loc="left", fontsize=12)
    grid(ax, "x")
    fig.tight_layout()

    def update(f):
        t = min(1.0, (f + 1) / (frames - 12))           # last 12 frames hold the final state
        for b, lb, yi, m in zip(bars, labels, y, MODELS):
            n = int(per_s[m] * t)
            b.set_width(n)
            lb.set_position((n + 8, yi))
            lb.set_text(f"{n}" + (f"  ({per_s[m] / per_s['PatchVision-X']:.1f}x PatchVision-X)"
                                  if t >= 1.0 and m.startswith("CNN") else ""))
        clock.set_text(f"t = {t * 1000:4.0f} ms")
        return [*bars, *labels, clock]

    FuncAnimation(fig, update, frames=frames, blit=False).save(ANI / "s1_one_second_race.gif",
                                                               writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
    print("wrote", ANI / "s1_one_second_race.gif")


# ------------------------------------------------------------------------------------------------------------
# GIF 2: the real GPU timeline of one forward pass (profiler trace)
# ------------------------------------------------------------------------------------------------------------
def gif_timeline():
    tl = pd.read_csv(S1 / "kernel_timeline_bs1.csv")
    lanes = ["PatchVision-X", "CNN small"]
    t_end = tl.end_us.max() / 1000.0
    frames, fps = 70, 14
    fig, ax = plt.subplots(figsize=(11, 4.0))
    # two rows per model: the kernel ticks (when the GPU starts each kernel) and a "busy so far" bar
    y = {"PatchVision-X": 3.0, "CNN small": 1.0}
    k = {m: tl[tl.model == m].sort_values("start_us") for m in lanes}
    end = {m: k[m].end_us.max() / 1000 for m in lanes}
    for m in lanes:
        ax.add_patch(Rectangle((0, y[m] - 0.05), end[m], 0.6, color=GPU_IDLE, lw=0))      # idle background
        ax.add_patch(Rectangle((0, y[m] - 0.75), end[m], 0.4, color="#f2f2f2", lw=0))     # busy-bar track
    ticks = {m: ax.vlines([], [], [], colors=COLOR[m], lw=0.7) for m in lanes}
    busy_bar = {m: Rectangle((0, y[m] - 0.75), 0, 0.4, color=COLOR[m], lw=0) for m in lanes}
    for p in busy_bar.values():
        ax.add_patch(p)
    head = ax.axvline(0, color=INK, lw=1.2)
    stats = {m: ax.text(t_end * 1.01, y[m] - 0.1, "", va="center", fontsize=9.5, color=INK) for m in lanes}
    ax.set_yticks([y[m] + 0.25 for m in lanes] + [y[m] - 0.55 for m in lanes],
                  [f"{m}\nkernels" for m in lanes] + ["GPU busy\nso far" for m in lanes], fontsize=8.5)
    ax.set_ylim(-0.1, 3.9)
    ax.set_xlim(0, t_end * 1.32)
    ax.set_xlabel("time since the forward pass started (ms)")
    ax.set_title("Real GPU timeline of ONE photo (batch 1): each tick = one GPU kernel, grey = GPU idle",
                 loc="left", fontsize=11.5)
    ax.text(0, -0.3, "CUPTI device trace via torch.profiler. The profiler slows the CPU side down "
            "(14.8 ms here vs 9.7 ms measured without it); the GPU work itself is the same.",
            transform=ax.transAxes, fontsize=8.5, color=MUTED)
    fig.tight_layout()

    def update(f):
        t = min(1.0, (f + 1) / (frames - 14)) * t_end
        head.set_xdata([t, t])
        for m in lanes:
            d = k[m][k[m].start_us / 1000 <= t]
            xs = (d.start_us / 1000).to_numpy()
            ticks[m].set_segments([[(xv, y[m] - 0.05), (xv, y[m] + 0.55)] for xv in xs])
            busy = ((d.end_us.clip(upper=t * 1000) - d.start_us).clip(lower=0)).sum() / 1000
            busy_bar[m].set_width(busy)
            el = min(t, end[m])
            stats[m].set_text(f"{len(d)}/{len(k[m])} kernels\nGPU busy {busy:.2f} of {el:.2f} ms "
                              f"({busy / max(el, 1e-6) * 100:.0f}%)")
        return []

    FuncAnimation(fig, update, frames=frames, blit=False).save(ANI / "s1_gpu_timeline.gif",
                                                               writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
    print("wrote", ANI / "s1_gpu_timeline.gif")


if __name__ == "__main__":
    fig_overview()
    fig_sweep()
    fig_where_time_goes()
    fig_accuracy_check()
    fig_tradeoff()
    gif_race()
    gif_timeline()
