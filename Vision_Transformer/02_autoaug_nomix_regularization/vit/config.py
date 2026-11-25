"""Experiment definitions for round 02 of the ViT study: `auto` augmentation, NO Mixup/CutMix, regularization search.
The SAME round as the CNNs (CNN_Baseline/02_autoaug_nomix_regularization/cnn/config.py): same base recipe, same runs,
same values, same trainer.

Base recipe (r0), identical to the CNNs' round-02 r0:
    * augmentation `auto`: RandomResizedCrop(0.35-1.0) + horizontal flip + TrivialAugmentWide (ONE op) + RandomErasing
    * NO Mixup and NO CutMix
    * AdamW lr 5e-4, 10 warm-up epochs + cosine, batch 64, label smoothing 0.1, weight decay 0.05, EMA weights,
      grad-clip 1.0, bf16 autocast, best epoch chosen on VALIDATION top-1
    * early stopping OFF: every run trains the full 300 epochs
    * the ViT-only extras (bird-box attention loss, flip-consistency loss, box-guided crop) are OFF, as for the CNNs
    * drop-path 0.1: the ViT's own default (the CNNs had none before round 02, so their r0 used 0)

Models:
    vit_base        the plain graph-attention ViT ("base" in the round-01 ablation: conv tokenizer + sparse graph
                    attention + position embedding; no LFE, no multi-scale heads)
    patchvision_x   the full architecture (LFE + multi-scale heads)   -- later, only when asked

Runs (one change at a time on top of r0, then the winners combined):
    r0_reference   r1_droppath0.2   r2_droppath0.4   r3_droppath0.5   r4_labelsmooth0.2   r5_wd0.1   r6_wd0.2   r7_sam
    r8_combined    the winners of r1-r7 for THIS model (chosen on validation; added after r0-r7)
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Optional, Tuple

from . import _pv  # noqa: F401
from patchvision.config import ModelConfig, PathsConfig, TrainConfig  # noqa: E402,F401


@dataclass
class ViTTrainConfig(TrainConfig):
    """The shared TrainConfig plus the SAM optimizer switch (as in the CNN round 02)."""
    use_sam: bool = False
    sam_rho: float = 0.05


@dataclass
class ViTExperiment:
    name: str = "vit_base/r0_reference"
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    train: ViTTrainConfig = field(default_factory=ViTTrainConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


BASE_TRAIN: Dict[str, Any] = {
    "use_bbox_loss": False, "use_consistency": False, "use_box_crop": False, "use_ema": True,
    "aug_tier": "auto", "use_mixup_cutmix": False,
    "early_stop_patience": 0,
}

MODELS: Dict[str, Dict[str, Any]] = {
    "vit_base": {"use_lfe": False, "use_multiscale": False, "drop_path": 0.1},
    "patchvision_x": {"drop_path": 0.1},
}

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

# r8_combined: filled in after r0-r7, with the same rule as the CNNs (a change counts as a winner only if it beat
# r0 on validation by more than one photo; one value per setting).
COMBINED: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {
    # r2 drop-path 0.4 (+21 val photos; ties r3 0.5, tie goes to the smaller change), r4 label smoothing 0.2 (+4),
    # r6 weight decay 0.2 (+7). Not included: r5 weight decay 0.1 (-7), r7 SAM (-9).
    "vit_base": ({"drop_path": 0.4}, {"label_smoothing": 0.2, "weight_decay": 0.2}),
}


def _build(cls, d: Dict[str, Any]):
    names = {f.name for f in fields(cls)}
    return cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in names})


def get_experiment(model: str, run: str, seed: int = 42, model_overrides: Optional[Dict[str, Any]] = None,
                   train_overrides: Optional[Dict[str, Any]] = None) -> ViTExperiment:
    """e.g. get_experiment("vit_base", "r1_droppath0.2"). Saved under results/<model>/<run>/seed<seed>/."""
    if model not in MODELS:
        raise KeyError(f"unknown model {model!r}. Available: {list(MODELS)}")
    if run == "r8_combined":
        if model not in COMBINED:
            raise KeyError(f"r8_combined for {model} is not defined yet: pick its winners from r1-r7 first")
        m_run, t_run = copy.deepcopy(COMBINED[model])
    else:
        if run not in RUNS and not (model_overrides or train_overrides):
            raise KeyError(f"unknown run {run!r}. Available: {list(RUNS)} (or pass overrides)")
        m_run, t_run = copy.deepcopy(RUNS.get(run, ({}, {})))
    m = {**MODELS[model], **m_run, **(model_overrides or {})}
    t = {**BASE_TRAIN, **t_run, **(train_overrides or {})}
    return ViTExperiment(name=f"{model}/{run}", seed=seed, model=_build(ModelConfig, m),
                         train=_build(ViTTrainConfig, t))
