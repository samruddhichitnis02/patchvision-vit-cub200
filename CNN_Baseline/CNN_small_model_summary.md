# CNN small: model summary

CNN small is a compact ResNet built from scratch (no pretrained weights). It is the parameter-matched CNN baseline
for PatchVision-X: 4.22 M parameters against PatchVision-X's 4.27 M. It uses the ResNet-18 layout (4 stages of 2
BasicBlocks each), with every stage's width scaled down so the parameter count matches the ViT.

All numbers in this file come from the code and from the saved run files:

| What | Where |
|---|---|
| Model code | [`02_autoaug_nomix_regularization/cnn/model.py`](02_autoaug_nomix_regularization/cnn/model.py) |
| Model and run settings | [`02_autoaug_nomix_regularization/cnn/config.py`](02_autoaug_nomix_regularization/cnn/config.py) (`MODELS["cnn_small"]`, `BASE_TRAIN`, `RUNS`, `COMBINED`) |
| Training loop | [`02_autoaug_nomix_regularization/cnn/engine.py`](02_autoaug_nomix_regularization/cnn/engine.py) |
| Optimizer, schedule, loss, EMA, data | Shared with the ViT study: `ViT_PatchVisionX/patchvision/` (`optim.py`, `losses.py`, `regularization.py`, `data.py`) |
| Saved settings of every run | `02_autoaug_nomix_regularization/results/cnn_small/<run>/seed42/config.json` |

The settings described here are the **round 02** recipe (`02_autoaug_nomix_regularization/`). Round 01 differences
are listed in [section 9](#9-differences-from-round-01).

---

## 1. At a glance

| Property | Value |
|---|---|
| Family | ResNet, BasicBlock (two 3×3 convs + shortcut) |
| Stage widths | 39 / 78 / 156 / 312 channels |
| Blocks per stage | 2 / 2 / 2 / 2 (8 blocks, 16 convs inside blocks) |
| Conv layers in total | 20 (1 stem + 16 block convs + 3 shortcut 1×1 convs) |
| Trainable parameters | **4,217,387 (4.22 M)** |
| BatchNorm running buffers | 5,850 (running mean + running var, not trained) |
| Input | 3 × 224 × 224 RGB |
| Output | 200 logits (one per CUB-200 bird species) |
| Compute | 1.40 GFLOPs per image |
| Weights file size (fp32) | about 16.9 MB |
| Latency (RTX 5090, batch 1) | 1.54 ms |
| Throughput (RTX 5090, batch 64) | about 14,100 images/s |
| Pretrained weights | none; trained from scratch on CUB-200-2011 only |

---

## 2. Architecture

```
input 3 x 224 x 224
  │
  ├─ STEM     Conv 7x7, 39 ch, stride 2, pad 3 (no bias) → BatchNorm → ReLU → MaxPool 3x3, stride 2   → 39 x 56 x 56
  │
  ├─ STAGE 1  BasicBlock(39 → 39,  stride 1) → BasicBlock(39 → 39)                                       → 39 x 56 x 56
  ├─ STAGE 2  BasicBlock(39 → 78,  stride 2, 1x1 shortcut) → BasicBlock(78 → 78)                         → 78 x 28 x 28
  ├─ STAGE 3  BasicBlock(78 → 156, stride 2, 1x1 shortcut) → BasicBlock(156 → 156)                       → 156 x 14 x 14
  ├─ STAGE 4  BasicBlock(156 → 312, stride 2, 1x1 shortcut) → BasicBlock(312 → 312)                      → 312 x 7 x 7
  │
  └─ HEAD     Global average pool → flatten → Linear(312 → 200)                                          → 200 logits
```

### 2.1 BasicBlock

```
x ──────────────────────────────────────────────┐ shortcut: identity, or Conv 1x1 (stride s) + BatchNorm
│                                               │           when the width or the resolution changes
Conv 3x3 (stride s, no bias) → BatchNorm → ReLU │
Conv 3x3 (stride 1, no bias) → BatchNorm        │
DropPath (training only)                        │
              └──────────────── + ──────────────┘
                                │
                               ReLU
```

- **Downsampling:** the first block of stages 2, 3 and 4 has stride 2, which halves height and width. That block's shortcut is a 1×1 conv + BatchNorm, so it matches the new width and size.
- **No bias in convolutions:** every conv is followed by BatchNorm, which has its own shift.
- **Activation:** ReLU everywhere.

### 2.2 Layer-by-layer parameter count

| Part | Output shape | Parameters | Share |
|---|---|---|---|
| Stem (conv 7×7 + BN) | 39 × 56 × 56 | 5,811 | 0.1 % |
| Stage 1 (2 blocks) | 39 × 56 × 56 | 55,068 | 1.3 % |
| Stage 2 (2 blocks) | 78 × 28 × 28 | 195,468 | 4.6 % |
| Stage 3 (2 blocks) | 156 × 14 × 14 | 780,312 | 18.5 % |
| Stage 4 (2 blocks) | 312 × 7 × 7 | 3,118,128 | 73.9 % |
| Head (Linear 312 → 200) | 200 | 62,600 | 1.5 % |
| **Total** | | **4,217,387** | 100 % |

Per block, in parameters:

| Block | conv1 | conv2 | BN (2 layers) | shortcut | Block total |
|---|---|---|---|---|---|
| Stage 1, block 1 | 13,689 | 13,689 | 156 | — | 27,534 |
| Stage 1, block 2 | 13,689 | 13,689 | 156 | — | 27,534 |
| Stage 2, block 1 | 27,378 | 54,756 | 312 | 3,198 | 85,644 |
| Stage 2, block 2 | 54,756 | 54,756 | 312 | — | 109,824 |
| Stage 3, block 1 | 109,512 | 219,024 | 624 | 12,480 | 341,640 |
| Stage 3, block 2 | 219,024 | 219,024 | 624 | — | 438,672 |
| Stage 4, block 1 | 438,048 | 876,096 | 1,248 | 49,296 | 1,364,688 |
| Stage 4, block 2 | 876,096 | 876,096 | 1,248 | — | 1,753,440 |

Most parameters (about 74 %) sit in stage 4, which works on the 7 × 7 feature map.

### 2.3 Weight initialization

| Layer | Initialization |
|---|---|
| Conv | Kaiming normal (`mode="fan_out"`, ReLU gain) |
| BatchNorm | weight = 1, bias = 0 |
| Last BatchNorm in every block (`bn2`) | weight = 0 (`zero_init_residual=True`), so each block starts as an identity and early training is stable |
| Linear head | PyTorch default (uniform, Kaiming-style) |

### 2.4 Stochastic depth (drop-path)

New in round 02. It mirrors the ViT's drop-path and is **off (0.0) unless a run turns it on**.

- **During training**, each block's residual branch is dropped for a random subset of the photos in the batch. For those photos the block passes its input straight through the shortcut. The surviving outputs are rescaled by 1 / (1 − p).
- **The drop rate rises linearly across the 8 blocks**, from 0 at the first block to the maximum at the last. With a maximum of 0.2 the rates are 0, 0.029, 0.057, 0.086, 0.114, 0.143, 0.171, 0.200.
- **At test time** every block is always used.

---

## 3. Data and preprocessing

| Item | Value |
|---|---|
| Dataset | CUB-200-2011, 200 bird species |
| Split | train 5,394 / validation 600 / test 5,794 (validation is held out from the official training set; test is the official test set) |
| Input size | 224 × 224 |
| Normalization | mean (0.4860, 0.4992, 0.4314), std (0.2265, 0.2220, 0.2609), computed on the CUB training split |
| Validation / test transform | Resize to 224 × 224 + Normalize (no crop, no augmentation) |
| Bounding boxes, part labels, attributes | **not used** by the CNN |

### 3.1 Training augmentation (`auto` tier)

These are applied in this order, newly at random every time a photo is loaded, so every epoch sees a different version of each photo:

| # | Augmentation | How often |
|---|---|---|
| 1 | `RandomResizedCrop(224, scale 0.35–1.0)`: a random part covering 35–100 % of the photo, resized to 224 × 224 | always |
| 2 | `RandomHorizontalFlip` | 50 % |
| 3 | `TrivialAugmentWide`: **one** operation picked at random from its list (rotation, shear, translation, brightness, contrast, colour, sharpness, posterize, solarize, auto-contrast, equalize, identity, …) at a random strength | always (exactly one op) |
| 4 | `RandomErasing`: a random rectangle is blanked | 25 % |

**No Mixup and no CutMix:** photos are never blended or pasted together.

---

## 4. Loss function

**Soft-target cross-entropy with label smoothing** (`SoftTargetCrossEntropy` + `smooth_one_hot` in `patchvision/losses.py`).

For a photo of class *y*, with smoothing ε and C = 200 classes:

- target for the true class = 1 − ε + ε / C
- target for every other class = ε / C

loss = −Σ<sub>c</sub> target<sub>c</sub> · log softmax(logits)<sub>c</sub>, averaged over the batch.

| Setting | Value |
|---|---|
| ε (label smoothing) | 0.1 by default; 0.2 in r4 and in r8 |
| Precision | computed in fp32, even when the forward pass runs in bf16 |
| Other losses | none. The ViT-only bounding-box attention loss and flip-consistency loss are off: a CNN has no attention maps |

---

## 5. Optimizer and schedule

### 5.1 Optimizer: AdamW

| Setting | Value |
|---|---|
| Optimizer | `torch.optim.AdamW` (decoupled weight decay) |
| Peak learning rate | 5e-4 |
| Betas | (0.9, 0.999) |
| Epsilon | 1e-8 (PyTorch default) |
| Weight decay | 0.05 by default; 0.1 in r5; 0.2 in r6 and r8 |
| Parameters that get weight decay | conv and linear weights: 4,211,337 parameters |
| Parameters without weight decay | BatchNorm weights/biases and the head bias: 6,050 parameters |
| Gradient clipping | global L2 norm clipped to 1.0 |

### 5.2 Learning-rate schedule

The learning rate is updated every step:

| Phase | Steps | Learning rate |
|---|---|---|
| Linear warm-up | first 10 epochs (840 steps) | 0 → 5e-4 |
| Cosine decay | remaining 290 epochs | 5e-4 → 1e-6 |

### 5.3 SAM (only in r7)

SAM is Sharpness-Aware Minimization (Foret et al., ICLR 2021), wrapped around AdamW with ρ = 0.05. For every batch:

1. Compute the gradient *g* at the current weights *w*.
2. Move to *w + ρ·g / ‖g‖*.
3. Compute the gradient there with BatchNorm running statistics frozen, so each batch updates them only once.
4. Go back to *w* and let AdamW step with that second gradient.

This costs two forward/backward passes per batch.

---

## 6. Other training settings

| Setting | Value |
|---|---|
| Epochs | 300, every run (early stopping **off**, so all runs get the same budget) |
| Batch size | 64 (last incomplete batch dropped) → 84 steps per epoch, 25,200 steps in total |
| Mixed precision | bf16 autocast for the forward pass (RTX 5090), loss in fp32, no gradient scaler needed |
| EMA of weights | yes, decay 0.999 (with a warm-up: decay = min(0.999, (1+n)/(10+n)) at update n). It averages all weights and BatchNorm statistics |
| Model that is evaluated and saved | the **EMA** copy |
| Model selection | the epoch with the best **validation top-1** (the test set is used only once, at the end, on that epoch) |
| Overfitting gap | top-1 on 600 un-augmented training photos minus validation top-1, measured every 5 epochs |
| Seed | 42 |
| Data-loader workers | 6 |
| Hardware | 1 × NVIDIA RTX 5090, Windows 11, PyTorch 2.11 (CUDA 12.8) |
| Time per run | about 32 minutes (about 6.3 s per epoch) |

---

## 7. Runs and results (round 02)

Each run changes **one** setting on top of r0. r8 combines the settings that beat r0 on validation by more than one photo.

| Run | Change from r0 | Validation top-1 | **Test top-1** | Gap | ECE |
|---|---|---|---|---|---|
| r0 reference | — | 44.67 | 43.75 | 55.3 | 2.22 |
| r1 | drop-path 0.2 | 46.33 | 43.82 | 54.0 | 2.28 |
| r2 | drop-path 0.4 | 42.67 | 42.30 | 55.7 | 2.02 |
| r3 | drop-path 0.5 | 40.67 | 40.16 | 53.7 | 1.80 |
| r4 | label smoothing 0.2 | 46.83 | 45.32 | 54.2 | 15.71 |
| r5 | weight decay 0.1 | 46.67 | 45.03 | 53.5 | 6.09 |
| r6 | weight decay 0.2 | 48.00 | 47.43 | 52.3 | 3.87 |
| r7 | SAM (ρ 0.05) | 44.83 | 43.11 | 53.3 | 1.98 |
| **r8 combined** | **weight decay 0.2 + label smoothing 0.2 + drop-path 0.2** | **48.67** | **47.65** | **51.5** | 14.27 |

All values are in %. Gap = clean training top-1 minus validation top-1. ECE = expected calibration error on test.

- **Final CNN small:** r8, **47.65 % test top-1**. That is +3.94 over r0 (p < 0.0001, exact McNemar test) and +4.0 over round 01 (43.65 %).
- **SAM** was left out of r8: it beat r0 by only one validation photo.
- **Label smoothing 0.2** makes the model under-confident (high ECE). Temperature scaling after training can fix this without changing accuracy.

---

## 8. Final model configuration (r8_combined)

```json
"model": {"img_size": 224, "num_classes": 200, "widths": [39, 78, 156, 312], "blocks": [2, 2, 2, 2],
          "zero_init_residual": true, "drop_path": 0.2},
"train": {"epochs": 300, "batch_size": 64, "lr": 0.0005, "min_lr": 1e-06, "warmup_epochs": 10,
          "weight_decay": 0.2, "betas": [0.9, 0.999], "grad_clip": 1.0, "label_smoothing": 0.2, "amp": true,
          "aug_tier": "auto", "use_mixup_cutmix": false, "use_ema": true, "ema_decay": 0.999,
          "best_metric": "acc1", "early_stop_patience": 0, "use_sam": false}
```

Full file: `02_autoaug_nomix_regularization/results/cnn_small/r8_combined/seed42/config.json`

To rebuild and load it:

```python
from cnn.config import get_experiment
from cnn.model import build_model
import torch

exp = get_experiment("cnn_small", "r8_combined")
model = build_model(exp.model)
ck = torch.load("results/cnn_small/r8_combined/seed42/best_model.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ck["state"])      # EMA weights of the best validation epoch
model.eval()
```

---

## 9. Differences from round 01

| Setting | Round 01 (`01_strongaug_mixup_cutmix/`) | Round 02 (final r8) |
|---|---|---|
| Architecture | same widths and blocks, no drop-path | same, drop-path 0.2 |
| Augmentation | `strong`: crop + flip + affine + colour jitter + erasing, all applied together | `auto`: crop + flip + **one** TrivialAugment op + erasing |
| Mixup / CutMix | on | **off** |
| Label smoothing | 0.1 | 0.2 |
| Weight decay | 0.05 | 0.2 |
| Test top-1 | 43.65 % | **47.65 %** |
