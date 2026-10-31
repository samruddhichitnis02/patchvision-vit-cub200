"""Makes the ViT study's `patchvision` package importable from the CNN study.

The CNN reuses the ViT's data pipeline (splits, normalisation, augmentation, mixup/CutMix), optimiser and
schedule, EMA, early stopping, metrics and plots, so that both studies are measured in exactly the same way.
Nothing in ViT_PatchVisionX is modified.

Importing this module (any `from . import _vit`) adds <repo>/ViT_PatchVisionX to sys.path.
"""
import sys
from pathlib import Path

VIT_DIR = Path(__file__).resolve().parents[2] / "ViT_PatchVisionX"
if not (VIT_DIR / "patchvision").is_dir():
    raise ImportError(f"{VIT_DIR / 'patchvision'} not found. The CNN study reuses the ViT study's `patchvision` "
                      f"package (data, metrics, optimiser); keep the folder ViT_PatchVisionX next to CNN_Baseline.")
if str(VIT_DIR) not in sys.path:
    sys.path.insert(0, str(VIT_DIR))
