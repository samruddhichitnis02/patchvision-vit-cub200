# CNN baselines

The CNN side of the PatchVision study: compact ResNets trained from scratch on CUB-200-2011. Each experiment
round has its own folder. The ViT (PatchVision-X) goes through the same rounds with the same settings, so the
two architectures stay comparable.

| Folder | Training recipe | Status |
|---|---|---|
| [`01_strongaug_mixup_cutmix/`](01_strongaug_mixup_cutmix) | `strong` augmentation (stacked) + Mixup/CutMix, label smoothing 0.1, weight decay 0.05, no drop-path. The recipe of the original ViT-vs-CNN comparison. | done: CNN small 43.65 %, CNN large 45.86 % test top-1 |
| `02_autoaug_nomix_regularization/` | `auto` augmentation (TrivialAugment, one op at a time), **no Mixup/CutMix**, plus a one-change-at-a-time search over drop-path (stochastic depth), label smoothing, weight decay and SAM | planned |

Shared data (`CUB_200_2011/`, `DataPreprocessing_Results/`) is read from the repository root.
