"""Experiment definitions for round 02 of the CNN study: `auto` augmentation, NO Mixup/CutMix, regularization search.

New base recipe (the same for every run and, later, for the ViT and PatchVision-X):
    * augmentation `auto` exactly as defined in the preprocessing notebook (section E):
      RandomResizedCrop(0.35-1.0) + horizontal flip + TrivialAugmentWide (ONE random op per photo) + RandomErasing(0.25)
    * NO Mixup and NO CutMix: photos are never mixed
    * everything else unchanged from round 01: AdamW lr 5e-4, 10 warm-up epochs + cosine, batch 64, label smoothing 0.1,
      weight decay 0.05, EMA weights, grad-clip 1.0, bf16 autocast, best epoch chosen on VALIDATION top-1
    * early stopping OFF: every run trains the full 300 epochs, so all runs get the same budget and stay comparable
      (in E4 the one run that happened to train longest had an unfair advantage)

Round plan (one change at a time on top of r0, then the winners combined):
    r0_reference        new base recipe only
    r1_droppath0.2      stochastic depth 0.2           r2_droppath0.4   r3_droppath0.5
    r4_labelsmooth0.2   label smoothing 0.1 -> 0.2
    r5_wd0.1            weight decay 0.05 -> 0.1       r6_wd0.2
    r7_sam              SAM optimizer (rho 0.05)
    r8_combined         the winners of r1-r7 for THIS model (chosen on validation; added after round 2)
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Optional, Tuple

from . import _vit  # noqa: F401  (adds ViT_PatchVisionX to sys.path)
from patchvision.config import PathsConfig, TrainConfig  # noqa: E402,F401


@dataclass
class CNNConfig:
    """Compact ResNet (BasicBlock). widths = channels of the 4 stages, blocks = BasicBlocks per stage."""
    img_size: int = 224
    num_classes: int = 200
    widths: Tuple[int, ...] = (39, 78, 156, 312)
    blocks: Tuple[int, ...] = (2, 2, 2, 2)
    zero_init_residual: bool = True      # last BatchNorm of every block starts at 0 (standard trick for stable training)
    drop_path: float = 0.0               # NEW: stochastic depth, max rate (linear 0 -> drop_path over the 8 blocks)


@dataclass
class CNNTrainConfig(TrainConfig):
    """The shared TrainConfig plus the SAM optimizer switch (new in round 02)."""
    use_sam: bool = False                # Sharpness-Aware Minimization (Foret et al., ICLR 2021) around AdamW
    sam_rho: float = 0.05                # size of the "look-around" step


@dataclass
class CNNExperiment:
    name: str = "cnn_small/r0_reference"
    seed: int = 42
    model: CNNConfig = field(default_factory=CNNConfig)
    train: CNNTrainConfig = field(default_factory=CNNTrainConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Base recipe of round 02. ViT-only components stay off (a CNN has no attention maps / token grid); EMA stays on.
BASE_TRAIN: Dict[str, Any] = {
    "use_bbox_loss": False, "use_consistency": False, "use_box_crop": False, "use_ema": True,
    "aug_tier": "auto", "use_mixup_cutmix": False,
    "early_stop_patience": 0,
}

MODELS: Dict[str, Dict[str, Any]] = {
    "cnn_small": {"widths": (39, 78, 156, 312)},     # 4.22 M parameters, matched to PatchVision-X
    "cnn_large": {"widths": (64, 128, 256, 512)},    # 11.28 M parameters, ResNet-18 layout
}

# run name -> (model overrides, train overrides), applied on top of the base recipe
RUNS: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {
    "r0_reference": ({}, {}),
    "r1_droppath0.2": ({"drop_path": 0.2}, {}),
    "r2_droppath0.4": ({"drop_path": 0.4}, {}),
    "r3_droppath0.5": ({"drop_path": 0.5}, {}),
    "r4_labelsmooth0.2": ({}, {"label_smoothing": 0.2}),
    "r5_wd0.1": ({}, {"weight_decay": 0.1}),
    "r6_wd0.2": ({}, {"weight_decay": 0.2}),
    "r7_sam": ({}, {"use_sam": True}),
}


def _build(cls, d: Dict[str, Any]):
    names = {f.name for f in fields(cls)}
    return cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in names})


def get_experiment(model: str, run: str, seed: int = 42, model_overrides: Optional[Dict[str, Any]] = None,
                   train_overrides: Optional[Dict[str, Any]] = None) -> CNNExperiment:
    """e.g. get_experiment("cnn_small", "r1_droppath0.2"). The run is saved under results/<model>/<run>/seed<seed>/.
    A run not in RUNS (e.g. r8_combined) can be built by passing its overrides explicitly."""
    if model not in MODELS:
        raise KeyError(f"unknown model {model!r}. Available: {list(MODELS)}")
    m_run, t_run = copy.deepcopy(RUNS.get(run, ({}, {})))
    if run not in RUNS and not (model_overrides or train_overrides):
        raise KeyError(f"unknown run {run!r}. Available: {list(RUNS)} (or pass overrides)")
    m = {**MODELS[model], **m_run, **(model_overrides or {})}
    t = {**BASE_TRAIN, **t_run, **(train_overrides or {})}
    return CNNExperiment(name=f"{model}/{run}", seed=seed, model=_build(CNNConfig, m),
                         train=_build(CNNTrainConfig, t))
