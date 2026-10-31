# CNN baselines

Two compact ResNet-style CNNs, trained **from scratch** on CUB-200-2011 (no pretrained weights), to compare with the Vision Transformer in [`../ViT_PatchVisionX`](../ViT_PatchVisionX).

| Experiment | Stage widths | Parameters | Purpose |
|---|---|---|---|
| `cnn_small` | 39, 78, 156, 312 | 4.22 M | Matched to PatchVision-X (4.27 M). The fair CNN-vs-ViT comparison. |
| `cnn_large` | 64, 128, 256, 512 | 11.28 M | ResNet-18 layout. Shows what extra CNN capacity buys. Not parameter-matched. |

Both: 7x7 stem + max-pool, four stages of two BasicBlocks (BatchNorm, ReLU, residual connections), global average pooling, 200-way linear head.

## Same conditions as the ViT `full` run

* Same split files, normalisation and augmentation (`DataPreprocessing_Results/`, `strong` tier, mixup/CutMix, label smoothing 0.1).
* AdamW (lr 5e-4, weight decay 0.05), 10 warm-up epochs then cosine decay, gradient clipping 1.0, bf16, batch 64, up to 300 epochs.
* EMA weights (decay 0.999), early stopping on validation (patience 60), best epoch chosen on **validation** only. The test set is used once, at the end.
* Switched off because they need attention maps or a token grid: bounding-box attention loss, flip-consistency loss, box-guided crop.

The data pipeline, optimiser, schedule, EMA, early stopping, metrics and plots are **imported from `../ViT_PatchVisionX/patchvision`** (`cnn/_vit.py` makes it importable), so both studies are measured identically. Nothing in the ViT folder is modified. This folder therefore needs `ViT_PatchVisionX/` next to it.

## Run it

Run everything from inside this folder.

```
cd CNN_Baseline
python train.py cnn_small 42
python train.py cnn_large 42
python watch_training.py cnn_small 42     # live monitor, second terminal
```

Or open `training.ipynb` (start Jupyter inside `CNN_Baseline/`): it holds the smoke test, the dry run, the training, and the CNN-vs-ViT comparison tables and figures. Keep `SMOKE_TEST = True` on the first pass (2 epochs on tiny synthetic data), then set it to `False`.

A finished run is skipped and an interrupted one resumes from `last.pt`.

## Layout

```
CNN_Baseline/
├── training.ipynb          run + results + CNN vs ViT comparison
├── train.py  watch_training.py
├── cnn/
│   ├── config.py           experiments (cnn_small, cnn_large) and settings
│   ├── model.py            the CNN
│   ├── engine.py           training / validation / test loop and everything it saves
│   ├── experiment.py       dry run (one batch forward/backward check)
│   ├── compare.py          CNN-vs-ViT table and figures
│   └── _vit.py             makes ../ViT_PatchVisionX/patchvision importable
└── results/<experiment>/seed<seed>/    same files as the ViT runs
```

Weights (`*.pt`) are ignored by Git, as in the ViT study.

## Results so far (seed 42, 5,794 test photos)

| Model | Params | GFLOPs | Top-1 | Top-5 | Macro-F1 | ECE | Train-val gap | Best epoch / run | Train time |
|---|---|---|---|---|---|---|---|---|---|
| CNN small | 4.22 M | 1.40 | 43.65 % | 69.02 % | 43.54 % | 7.07 % | 54.7 pts | 163 / 223 | 33.5 min |
| CNN large | 11.28 M | 3.63 | 45.86 % | 71.16 % | 45.79 % | 8.56 % | 53.3 pts | 140 / 200 | 30.0 min |
| ViT `base` | 4.31 M | 1.55 | 39.52 % | 65.12 % | 39.12 % | 2.49 % | 59.8 pts | 186 / 246 | 48.4 min |
| PatchVision-X (`full`) | 4.27 M | 1.55 | **47.39 %** | **73.59 %** | **47.05 %** | 5.95 % | 49.8 pts | 153 / 213 | 71.2 min |

The full table, figures and per-species comparison are in `training.ipynb` and in `results/` (`comparison.png`, `comparison_table.csv`, `per_class_cnn_vs_vit.png`).

How to read this:

* PatchVision-X is 3.7 points ahead of the parameter-matched CNN and 1.5 points ahead of the 11 M CNN. The second gap is within the test-set sampling noise (about plus/minus 1.3 points).
* Every model memorizes the training photos (gaps of 50 to 60 points), so this looks like a small-data effect and not a ViT-specific one.
* **One seed each.** The CNNs also use the ViT's learning rate and weight decay without tuning. Three seeds per model are still needed before quoting exact numbers.
