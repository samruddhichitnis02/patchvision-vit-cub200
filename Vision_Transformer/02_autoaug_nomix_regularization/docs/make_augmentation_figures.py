"""Augmentation figures for the round-02 README (docs/figures/augmentation/*.jpg).

    python docs/make_augmentation_figures.py      (from inside Vision_Transformer/02_autoaug_nomix_regularization/)

 Uses the project's real transforms (auto tier) and torchvision's
TrivialAugmentWide operations with their real ranges."""
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2 as T

ROOT = Path(__file__).resolve().parents[3]                        # repository root
sys.path.insert(0, str(ROOT / "Vision_Transformer"))
from patchvision.data import build_transforms  # noqa: E402

OUT = Path(__file__).parent / "figures" / "augmentation"
OUT.mkdir(exist_ok=True)
CUB = ROOT / "CUB_200_2011"
train = pd.read_csv(ROOT / "DataPreprocessing_Results" / "train.csv")
MEAN = [0.4859701097011566, 0.49921685457229614, 0.43137678503990173]
STD = [0.22647330164909363, 0.2219720035791397, 0.2609000504016876]
INK, MUTED = "#222222", "#666666"
plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})
TA = T.TrivialAugmentWide()
SPACE = TA._AUGMENTATION_SPACE
OPS = list(SPACE)
GEOM = {"ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate"}
FACTS = {}


def path(fp):
    return CUB / fp.replace("\\", "/").split("CUB_200_2011/", 1)[1]


def load(fp):
    return tv_tensors.Image(T.functional.pil_to_tensor(Image.open(path(fp)).convert("RGB")))


def show(t):
    if t.dtype == torch.uint8:
        return (t.float() / 255).clamp(0, 1).permute(1, 2, 0).numpy()
    t = t.detach().float()
    m, s = torch.tensor(MEAN)[:, None, None], torch.tensor(STD)[:, None, None]
    return (t * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


def save(fig, name):
    tmp = OUT / name
    fig.savefig(tmp, dpi=170, bbox_inches="tight", facecolor="white")
    Image.open(tmp).convert("RGB").save(tmp.with_suffix(".jpg"), quality=85)   # JPEG keeps the repository small
    tmp.unlink()
    plt.close(fig)
    print("wrote", name)


def apply_op(img, op, mag):
    return TA._apply_image_or_video_transform(img, op, mag, interpolation=TA.interpolation, fill=TA._fill)


def magnitude(op, b, h=224, w=224):
    m = SPACE[op][0](31, h, w)
    return 0.0 if m is None else float(m[b])


def describe(op, mag):
    if op in ("ShearX", "ShearY"):
        import math
        return f"{op} {math.degrees(math.atan(mag)):+.0f} deg"
    if op in ("TranslateX", "TranslateY"):
        return f"{op} {mag:+.0f} px"
    if op == "Rotate":
        return f"Rotate {mag:+.0f} deg"
    if op in ("Brightness", "Color", "Contrast", "Sharpness"):
        return f"{op} x{1 + mag:.2f}"
    if op == "Posterize":
        return f"Posterize {int(mag)} bits"
    if op == "Solarize":
        return f"Solarize above {mag * 255:.0f}"
    return op


crop = T.RandomResizedCrop((224, 224), scale=(0.35, 1.0), antialias=True)
to_float, norm = T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)
erase = T.RandomErasing(p=0.25)


def auto_traced(img):
    """The auto chain step by step (same operations, same order as build_transforms('auto')), returning the op used."""
    x = crop(img)
    flipped = bool(torch.rand(()) < 0.5)
    if flipped:
        x = T.functional.horizontal_flip(x)
    op = OPS[int(torch.randint(len(OPS), ()))]
    mags = SPACE[op][0](31, 224, 224)
    mag = 0.0
    if mags is not None:
        mag = float(mags[int(torch.randint(31, ()))])
        if SPACE[op][1] and torch.rand(()) <= 0.5:
            mag = -mag
    x = apply_op(x, op, mag)
    x = norm(to_float(x))
    erased = bool(torch.rand(()) < 0.25)
    if erased:
        x = T.RandomErasing(p=1.0)(x)
    return x, op, mag, flipped, erased


row = train[train.class_name == "Painted Bunting"].iloc[0]
orig = load(row.filepath)

# ---------------------------------------------------------------- 1. the chain, one step at a time
torch.manual_seed(7)
random.seed(7)
s1 = crop(orig)
s2 = T.functional.horizontal_flip(s1)                         # forced ON for the demo (50 % in training)
op_demo, mag_demo = "Rotate", magnitude("Rotate", 10)
s3 = apply_op(s2, op_demo, mag_demo)
s4 = norm(to_float(s3))
s5 = T.RandomErasing(p=1.0)(s4)                                # forced ON for the demo (25 % in training)
panels = [(orig, "0. Original JPEG\n(from disk, any size)"), (s1, "1. Random crop\n35-100 % -> 224x224"),
          (s2, "2. Flip left-right\n(50 % chance)"), (s3, f"3. TrivialAugment: ONE op\nhere: {describe(op_demo, mag_demo)}"),
          (s4, "4. Normalize\n(numbers change, look same)"), (s5, "5. Random erase\n(25 % chance)")]
fig, ax = plt.subplots(1, 6, figsize=(15, 3.4))
for a, (t, title) in zip(ax, panels):
    a.imshow(show(t))
    a.set_title(title, fontsize=9)
    a.axis("off")
fig.suptitle(f"One photo through the round-02 chain ({row.class_name}). Each step works on the previous step's "
             "output; only the last image goes to the model.", fontsize=11, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.9))
save(fig, "f1_chain.png")

# ---------------------------------------------------------------- 2. all 14 operations, medium and maximum strength
torch.manual_seed(11)
base = crop(orig)
for part, ops in (("a", OPS[:7]), ("b", OPS[7:])):
    fig, ax = plt.subplots(2, 7, figsize=(17, 6.4))
    for j, op in enumerate(ops):
        for i, (b, lab) in enumerate(((15, "medium"), (30, "maximum"))):
            mag = magnitude(op, b)
            if op in ("Posterize", "Solarize"):
                mag = magnitude(op, b)
            a = ax[i, j]
            a.imshow(show(apply_op(base, op, mag)))
            name = describe(op, mag) if SPACE[op][0](31, 224, 224) is not None else op
            a.set_title(f"{name}\n({lab})" if SPACE[op][0](31, 224, 224) is not None else f"{op}\n(no strength)",
                        fontsize=8.5, color="#1a5e9a" if op in GEOM else ("#7a3d00" if op != "Identity" else INK))
            a.axis("off")
    fig.suptitle(f"The 14 TrivialAugment operations ({'1-7' if part == 'a' else '8-14'}), applied to the same cropped "
                 "photo. Top row: medium strength (level 15 of 31). Bottom row: maximum strength (level 30 of 31).",
                 fontsize=11, x=0.01, ha="left")
    fig.text(0.01, 0.925, "Blue titles = geometric (moves pixels; empty corners become black).  Brown titles = colour / "
             "brightness (pixels stay in place).", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.subplots_adjust(hspace=0.32)
    save(fig, f"f2{part}_ops.png")

# ---------------------------------------------------------------- 3. same photo, 8 epochs, with the op that was drawn
fig, ax = plt.subplots(1, 8, figsize=(17, 3.0))
torch.manual_seed(1)
for e, a in enumerate(ax, 1):
    x, op, mag, fl, er = auto_traced(orig)
    a.imshow(show(x))
    extra = (" + flip" if fl else "") + (" + erase" if er else "")
    a.set_title(f"epoch {e}\n{describe(op, mag)}{extra}", fontsize=8)
    a.axis("off")
fig.suptitle("The same photo in 8 different epochs: a new crop and a new random operation every time (the original "
             "itself is never shown to the model during training)", fontsize=11, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.86))
save(fig, "f3_epochs.png")

# ---------------------------------------------------------------- 4. one real batch of 64 (no mixing)
g = torch.Generator().manual_seed(42)
order = torch.randperm(len(train), generator=g)[:64].tolist()
batch = train.iloc[order].reset_index(drop=True)
n_species = batch.label.nunique()
dup = batch.label.value_counts()
fig, ax = plt.subplots(4, 16, figsize=(17, 5.2))
torch.manual_seed(3)
counts = {}
for i, (a, r) in enumerate(zip(ax.ravel(), batch.itertuples())):
    x, op, mag, fl, er = auto_traced(load(r.filepath))
    counts[op] = counts.get(op, 0) + 1
    a.imshow(show(x))
    a.set_title(f"{i + 1}: {op}", fontsize=6.3, color=MUTED)
    a.axis("off")
fig.suptitle(f"One real batch of 64 (first shuffle, seed 42): {n_species} different species, every photo with its own "
             "random operation, NO mixing between photos", fontsize=11, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.93))
save(fig, "f4_batch.png")
FACTS["batch"] = (n_species, int((dup == 1).sum()), int((dup == 2).sum()), int((dup == 3).sum()))
FACTS["batch_ops"] = counts

# ---------------------------------------------------------------- 5. the label (target) with and without smoothing
fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
for a, (ls, title) in zip(ax, ((0.0, "No smoothing (not used)"), (0.1, "Label smoothing 0.1 (r0 and most runs)"),
                               (0.2, "Label smoothing 0.2 (run r4)"))):
    correct = (1 - ls + ls / 200) * 100
    other = ls / 200 * 100
    a.bar([0], [correct], color="#2c7fb8", width=0.6)
    a.bar([1], [other * 199], color="#cccccc", width=0.6)
    a.text(0, correct + 2, f"{correct:.2f} %", ha="center", fontsize=10, fontweight="bold")
    a.text(1, other * 199 + 2, f"{other * 199:.2f} % in total\n= {other:.2f} % each", ha="center", fontsize=9)
    a.set_xticks([0, 1], ["Painted Bunting\n(the correct species)", "the other 199\nspecies together"], fontsize=8.5)
    a.set_ylim(0, 115)
    a.set_yticks([0, 25, 50, 75, 100])
    a.set_title(title, fontsize=10)
    a.spines[["top", "right"]].set_visible(False)
ax[0].set_ylabel("target the model is trained toward, %")
fig.tight_layout()
save(fig, "f5_label.png")

# ---------------------------------------------------------------- 6. pipeline diagram
fig, ax = plt.subplots(figsize=(15, 4.2))
ax.set_xlim(0, 100)
ax.set_ylim(-2, 30)
ax.axis("off")
boxes = [(1, "1. Shuffle\n5,394 photos\n(new order\nevery epoch)", "#eeeeee"),
         (17, "2. 6 CPU workers\nopen JPEG +\naugment\n(crop, flip, ONE op,\nerase)", "#dbe9f6"),
         (33, "3. Batch\nof 64 images\n(many species,\nnot mixed)", "#dbe9f6"),
         (49, "4. Model\nforward + loss\n+ weight update\n(AdamW, + SAM\nin r7)", "#e5f5e0"),
         (65, "5. Repeat\n84 batches\n= 1 epoch", "#eeeeee"),
         (81, "6. Check on\n600 original\nvalidation photos\n(no update)", "#fff3cd")]
for x0, txt, col in boxes:
    ax.add_patch(FancyBboxPatch((x0, 7), 14, 17, boxstyle="round,pad=0.4", fc=col, ec="#888888"))
    ax.text(x0 + 7, 15.5, txt, ha="center", va="center", fontsize=9.5)
for x0 in (15.4, 31.4, 47.4, 63.4, 79.4):
    ax.annotate("", xy=(x0 + 1.4, 15.5), xytext=(x0 + 0.1, 15.5), arrowprops=dict(arrowstyle="-|>", lw=1.5))
ax.annotate("", xy=(8, 6.0), xytext=(88, 6.0), arrowprops=dict(arrowstyle="-|>", lw=1.2, color=MUTED,
                                                              connectionstyle="arc3,rad=-0.08"))
ax.text(48, -1.2, "7. next epoch: start again from the original files (new shuffle, new crops, new random operations); "
        "300 epochs, no early stopping", ha="center", fontsize=9.5, color=MUTED)
ax.text(1, 28.5, "Round 02, one epoch: what happens in order", fontsize=12, fontweight="bold")
ax.text(1, 26, "blue = runs on the CPU (in parallel)   green = training on the GPU   yellow = measuring only.   "
        "Compared with round 01 there is NO mixing step.", fontsize=8.5, color=MUTED)
save(fig, "f6_pipeline.png")

# ---------------------------------------------------------------- 7. round 01 vs round 02, same photo
strong = build_transforms("strong", 224, MEAN, STD, train=True)
torch.manual_seed(5)
fig, ax = plt.subplots(2, 6, figsize=(16, 6.8))
for j in range(6):
    ax[0, j].imshow(show(strong(orig)))
    ax[0, j].set_title(f"round 01, view {j + 1}", fontsize=9)
    x, op, mag, fl, er = auto_traced(orig)
    ax[1, j].imshow(show(x))
    ax[1, j].set_title(f"round 02, view {j + 1}\n{describe(op, mag)}", fontsize=9)
for a in ax.ravel():
    a.axis("off")
fig.suptitle("Same photo, 6 random training views per round (before any mixing). Round 01: small changes, all stacked. "
             "Round 02: one change, sometimes strong.", fontsize=11, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.subplots_adjust(hspace=0.3)
save(fig, "f7_round01_vs_02.png")

# ---------------------------------------------------------------- facts: how often each kind happens
fr = {"identity": 1 / 14, "geometric": 5 / 14, "colour": 8 / 14,
      "no_change_after_crop": 1 / 14 * 0.5 * 0.75}
FACTS["fractions"] = fr
(OUT / "facts.txt").write_text("\n".join(f"{k}: {v}" for k, v in FACTS.items()) + "\n")
print((OUT / "facts.txt").read_text())
