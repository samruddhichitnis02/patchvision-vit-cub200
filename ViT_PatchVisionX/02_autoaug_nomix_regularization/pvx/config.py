"""Experiment definitions for round 02 of PatchVision-X: `auto` augmentation, NO Mixup/CutMix, regularization search.
The same round, runs and values as the CNNs and the plain ViT (CNN_Baseline/02_... and Vision_Transformer/02_...).

Base recipe (r0) = PatchVision-X round 01 `full`, with ONLY the round-02 data changes:
    changed (as for the CNNs and the plain ViT):
        * augmentation `auto` instead of `strong`: RandomResizedCrop(0.35-1.0) + flip + TrivialAugmentWide (ONE op)
          + RandomErasing
        * NO Mixup and NO CutMix
        * early stopping OFF: every run trains the full 300 epochs
    kept exactly as in round 01 `full`:
        * model: LFE + multi-scale heads + sparse gate, drop-path 0.1
        * extras ON: bird-box attention loss (0.1), flip-consistency loss, box-guided crop (30 %), EMA 0.999
        * AdamW lr 5e-4, weight decay 0.05, label smoothing 0.1, batch 64, 10 warm-up epochs + cosine, grad-clip 1.0,
          bf16, best epoch chosen on VALIDATION top-1
        * trainer: Vision_Transformer/patchvision/engine.py (round 01's trainer; SAM is used only by r7)

Runs (one change at a time on top of r0, then the winners combined):
    r0_reference   r1_droppath0.2   r2_droppath0.4   r3_droppath0.5   r4_labelsmooth0.2   r5_wd0.1   r6_wd0.2   r7_sam
    r8_combined    the winners of r1-r7 (chosen on validation; added after r0-r7)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from . import _pv  # noqa: F401
from patchvision.config import ExperimentConfig, ModelConfig, PathsConfig, TrainConfig, _build  # noqa: E402,F401

MODEL = "patchvision_x"


@dataclass
class PVXTrainConfig(TrainConfig):
    """Round 01's TrainConfig plus the SAM switch (off except in r7)."""
    use_sam: bool = False
    sam_rho: float = 0.05


# round 01 `full` = the dataclass defaults; round 02 changes only these
BASE_TRAIN: Dict[str, Any] = {"aug_tier": "auto", "use_mixup_cutmix": False, "early_stop_patience": 0}

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

# r8_combined: the winners of r1-r7 on validation (beat r0 = 46.67 % by more than one photo; one value per setting):
#   drop-path 0.4 (r2, 50.17 %), label smoothing 0.2 (r4, 49.00 %), weight decay 0.2 (r6, 48.00 %).
#   SAM (r7, 47.00 %, +2 photos) only just passed the rule; left out by choice: the gain is within noise and it
#   doubles the training time.
COMBINED: Tuple[Dict[str, Any], Dict[str, Any]] | None = (
    {"drop_path": 0.4},
    {"label_smoothing": 0.2, "weight_decay": 0.2},
)


def get_experiment(run: str, seed: int = 42) -> ExperimentConfig:
    """e.g. get_experiment("r1_droppath0.2"). Saved under results/patchvision_x/<run>/seed<seed>/."""
    if run == "r8_combined":
        if COMBINED is None:
            raise KeyError("r8_combined is not defined yet: pick the winners from r1-r7 first")
        m_run, t_run = copy.deepcopy(COMBINED)
    else:
        if run not in RUNS:
            raise KeyError(f"unknown run {run!r}. Available: {list(RUNS)}")
        m_run, t_run = copy.deepcopy(RUNS[run])
    return ExperimentConfig(name=f"{MODEL}/{run}", seed=seed, model=_build(ModelConfig, m_run),
                            train=_build(PVXTrainConfig, {**BASE_TRAIN, **t_run}))
