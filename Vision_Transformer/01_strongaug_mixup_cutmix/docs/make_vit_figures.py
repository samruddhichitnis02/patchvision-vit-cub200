"""Figures and animations for the round-01 README of the plain Vision Transformer (`base` run only).

    python docs/make_vit_figures.py        (from inside Vision_Transformer/01_strongaug_mixup_cutmix/)

Reads results/base/seed42/ and writes docs/figures/v*.png, docs/animations/v*.gif and docs/vit_facts.txt.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "results" / "base" / "seed42"
FIG, ANI = HERE / "figures", HERE / "animations"
FIG.mkdir(exist_ok=True)
ANI.mkdir(exist_ok=True)

BLUE, GREEN, GREY, ORANGE, PINK = "#2a6fdb", "#2e9e5b", "#8a8f98", "#e07b28", "#f2b8c6"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})

h = pd.read_csv(RUN / "history.csv")
m =json.loads((RUN / "test_metrics.json").read_text())["row"]
pc = pd.read_csv(RUN / "test_per_class.csv")
pred = pd.read_csv(RUN / "test_predictions.csv")
rel = pd.read_csv(RUN / "test_reliability.csv")
best_ep, last_ep = int(m["best_epoch"]), int(m["epochs_run"])
clean = h.dropna(subset=["clean_acc1"])
facts = []

# ---------------------------------------------------------------- v01 results
fig, ax = plt.subplots(figsize=(7, 3.6))
names = ["test top-1", "test top-5", "test macro-F1", "val top-1\n(chosen epoch)"]
vals = [m["test_acc1"], m["test_acc5"], m["test_macro_f1"], m["val_acc1_at_best"]]
bars = ax.bar(names, vals, color=[BLUE, BLUE, BLUE, GREY], width=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.2f} %", ha="center", fontsize=10)
ax.axhline(0.5, color="k", lw=0.8, ls=":")
ax.text(3.45, 2.0, "chance: 0.5 %", ha="right", fontsize=8, color="k")
ax.set_ylim(0, 80)
ax.set_ylabel("%")
ax.set_title("Plain ViT (base), seed 42: 5,794 test photos")
fig.tight_layout()
fig.savefig(FIG / "v01_results.png")
plt.close(fig)

# ---------------------------------------------------------------- v02 training curves
fig, axs = plt.subplots(2, 2, figsize=(11, 7))
ax = axs[0, 0]
ax.plot(h.epoch, h.train_loss, color=GREY, label="training loss (augmented, Mixup/CutMix targets)")
ax.plot(h.epoch, h.val_loss, color=BLUE, label="validation loss")
vmin = h.loc[h.val_loss.idxmin()]
ax.scatter([vmin.epoch], [vmin.val_loss], color=BLUE, zorder=3)
ax.annotate(f"lowest {vmin.val_loss:.2f}\n@ epoch {int(vmin.epoch)}", (vmin.epoch, vmin.val_loss),
            xytext=(10, -35), textcoords="offset points", fontsize=8, color=BLUE)
ax.set_title("Loss")
ax.set_xlabel("epoch")
ax.legend(fontsize=8)

ax = axs[0, 1]
ax.plot(clean.epoch, clean.clean_acc1, color=GREEN, marker=".", label="clean training photos (no augmentation)")
ax.plot(h.epoch, h.val_acc1, color=BLUE, label="validation photos")
ax.fill_between(clean.epoch, np.interp(clean.epoch, h.epoch, h.val_acc1), clean.clean_acc1, color=PINK, alpha=0.6,
                label="memorizing gap")
ax.axvline(best_ep, color="k", ls="--", lw=0.8, label=f"chosen epoch ({best_ep})")
ax.set_ylim(0, 102)
ax.set_title("Accuracy (top-1, %)")
ax.set_xlabel("epoch")
ax.legend(fontsize=8, loc="lower right")

ax = axs[1, 0]
gap = clean.clean_acc1 - np.interp(clean.epoch, h.epoch, h.val_acc1)
ax.plot(clean.epoch, gap, color="#c0395b", marker=".")
ax.set_title("Memorizing gap: clean-train minus validation (points)")
ax.set_xlabel("epoch")
ax.set_ylim(0, 70)

ax = axs[1, 1]
ax.plot(h.epoch, h.train_lr_first, color=BLUE)
ax.axvspan(0, 10, color="#ffe9a8", alpha=0.7, label="10 warm-up epochs")
ax.axvline(last_ep, color=ORANGE, ls="--", lw=1)
ax.text(last_ep - 4, 3.5e-4, f"early stop\n@ epoch {last_ep}", ha="right", fontsize=8, color=ORANGE)
ax.set_xlim(0, 300)
ax.set_title("Learning rate (cosine; the schedule is planned for 300 epochs)")
ax.set_xlabel("epoch")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "v02_training_curves.png")
plt.close(fig)
facts += [f"lowest val loss {vmin.val_loss:.3f} @ {int(vmin.epoch)}; final val loss {h.val_loss.iloc[-1]:.3f}",
          f"clean-train final {clean.clean_acc1.iloc[-1]:.1f}; max {clean.clean_acc1.max():.1f}",
          f"gap at chosen epoch (test_metrics) {m['overfit_gap_clean_minus_val']:.1f}; final gap {gap.iloc[-1]:.1f}",
          f"best val {h.val_acc1.max():.2f} @ {int(h.loc[h.val_acc1.idxmax(), 'epoch'])}"]
for thr in (90, 99):
    e = clean.loc[clean.clean_acc1 >= thr, "epoch"]
    facts.append(f"clean-train first >= {thr}%: epoch {int(e.iloc[0]) if len(e) else None}")

# ---------------------------------------------------------------- v03 memorizing gap animation
fig, ax = plt.subplots(figsize=(7, 4))
ax.set_xlim(0, last_ep + 4)
ax.set_ylim(0, 102)
ax.set_xlabel("epoch")
ax.set_ylabel("top-1 accuracy (%)")
ax.set_title("Plain ViT: what it learned vs what it memorized")
lc, = ax.plot([], [], color=GREEN, marker=".", label="clean training photos")
lv, = ax.plot([], [], color=BLUE, label="validation photos")
ax.legend(loc="center right", fontsize=9)
txt = ax.text(5, 92, "", fontsize=10)
fill = [None]
frames = list(range(5, last_ep + 1, 3)) + [last_ep] * 12


def draw(ep):
    c = clean[clean.epoch <= ep]
    v = h[h.epoch <= ep]
    lc.set_data(c.epoch, c.clean_acc1)
    lv.set_data(v.epoch, v.val_acc1)
    if fill[0] is not None:
        fill[0].remove()
    fill[0] = ax.fill_between(c.epoch, np.interp(c.epoch, h.epoch, h.val_acc1), c.clean_acc1, color=PINK, alpha=0.6)
    if len(c):
        g = c.clean_acc1.iloc[-1] - np.interp(c.epoch.iloc[-1], h.epoch, h.val_acc1)
        txt.set_text(f"epoch {ep}   gap {g:.0f} points")
    return lc, lv


FuncAnimation(fig, draw, frames=frames, interval=80).save(ANI / "v03_memorizing_gap.gif", writer=PillowWriter(fps=12))
plt.close(fig)

# ---------------------------------------------------------------- v04 augmentation (strong tier rows only)
src = Image.open(FIG / "augmentations.png")
w, hgt = src.size
src.crop((0, 0, w, int(hgt * 0.705))).save(FIG / "v04_strong_augmentation.png")

# ---------------------------------------------------------------- v05 species best/worst
pcs = pc.sort_values("f1")
worst, best = pcs.head(10), pcs.tail(10)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, d, col, t in [(axs[0], best, BLUE, "10 easiest species (test F1)"), (axs[1], worst, "#c0395b", "10 hardest species (test F1)")]:
    ax.barh(d.class_name, d.f1, color=col)
    for y, v in enumerate(d.f1):
        ax.text(v + 1, y, f"{v:.1f}", va="center", fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_title(t)
    ax.tick_params(axis="y", labelsize=8)
fig.tight_layout()
fig.savefig(FIG / "v05_species_best_worst.png")
plt.close(fig)
facts += [f"median F1 {pc.f1.median():.1f}; >80: {(pc.f1 > 80).sum()}; <20: {(pc.f1 < 20).sum()}; ==0: {(pc.f1 == 0).sum()}",
          "best: " + ", ".join(f"{r.class_name} {r.f1:.1f}" for r in best.iloc[::-1].itertuples()),
          "worst: " + ", ".join(f"{r.class_name} {r.f1:.1f}" for r in worst.itertuples())]

# ---------------------------------------------------------------- v06 top confusions
names = dict(zip(pc.class_id, pc.class_name))
wrong = pred[pred.pred != pred.label]
top = wrong.groupby(["label", "pred"]).size().sort_values(ascending=False).head(12)
support = pc.set_index("class_id").support
fig, ax = plt.subplots(figsize=(9, 4.5))
lab = [f"{names[a]} → {names[b]}" for a, b in top.index]
frac = [100 * n / support[a] for (a, b), n in top.items()]
ax.barh(lab[::-1], list(top.values)[::-1], color=BLUE)
for y, (n, f) in enumerate(zip(list(top.values)[::-1], frac[::-1])):
    ax.text(n + 0.2, y, f"{n} photos ({f:.0f} % of the species)", va="center", fontsize=8)
ax.set_xlabel("test photos")
ax.set_xlim(0, max(top.values) * 1.6)
ax.set_title("Most common mistakes (true species → predicted species)")
ax.tick_params(axis="y", labelsize=8)
fig.tight_layout()
fig.savefig(FIG / "v06_confusions.png")
plt.close(fig)
facts.append("top confusions: " + "; ".join(f"{l} {n}" for l, n in zip(lab, top.values)))

# ---------------------------------------------------------------- v07 calibration
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
ax = axs[0]
ax.plot([0, 1], [0, 1], color="k", ls=":", lw=1, label="perfect calibration")
r = rel[rel.fraction > 0]
ax.plot(r.confidence, r.accuracy, color=BLUE, marker="o", label="plain ViT")
ax.set_xlabel("confidence of the top guess")
ax.set_ylabel("accuracy")
ax.set_title(f"Reliability (ECE {m['test_ece']:.2f} %)")
ax.legend(fontsize=8)
ax = axs[1]
ax.bar((rel.bin_lo + rel.bin_hi) / 2, 100 * rel.fraction, width=(rel.bin_hi - rel.bin_lo) * 0.9, color=BLUE)
ax.set_xlabel("confidence of the top guess")
ax.set_ylabel("% of test photos")
ax.set_title("How confident the model is")
fig.tight_layout()
fig.savefig(FIG / "v07_calibration.png")
plt.close(fig)
conf = pred.confidence
facts.append(f"mean confidence {100 * conf.mean():.1f}; acc {m['test_acc1']:.2f}; share conf>0.5 {100 * (conf > 0.5).mean():.1f}; "
             f"acc when conf>0.5 {100 * (pred[conf > 0.5].pred == pred[conf > 0.5].label).mean():.1f}")

# ---------------------------------------------------------------- v08 attention over training (snapshots)
snaps = sorted((RUN / "attention_snapshots").glob("epoch_*.png"))
frames = []
for p in snaps:
    im = Image.open(p).convert("RGB")
    W, H = im.size
    im = im.crop((0, 0, W, int(H * 0.5)))              # the first 8 photos: image, last block, mean of blocks
    im = im.resize((900, int(900 * im.size[1] / W)), Image.LANCZOS)
    frames.append(im.convert("P", palette=Image.ADAPTIVE, colors=160))
frames[0].save(ANI / "v08_attention_over_training.gif", save_all=True, append_images=frames[1:] + [frames[-1]] * 3,
               duration=450, loop=0, optimize=True)

# ---------------------------------------------------------------- facts
facts += [f"in-box attention test {m['test_attn_in_box']:.3f}; box area frac val {h.val_box_area_frac.mean():.3f}",
          f"gate mean last {h.val_gate_mean.iloc[-1]:.3f}; gate sparsity last {h.val_gate_sparsity.iloc[-1]:.4f}",
          f"time {m['train_time_s'] / 60:.1f} min; {m['sec_per_epoch']:.2f} s/epoch"]
(HERE / "vit_facts.txt").write_text("\n".join(facts), encoding="utf-8")
print("\n".join(facts))
