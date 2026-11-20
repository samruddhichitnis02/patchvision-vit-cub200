"""Load the trained models (ViT and CNN) from their `best_model.pt` checkpoints.

Both training codes save {"exp": <config dict>, "state": <EMA weights when EMA was used>, ...}.
The ViT returns (logits, aux dict); the CNN returns logits. For inference we only ever want logits, so every
model is wrapped in `LogitsOnly`. NOTE: in the *baseline* the ViT still computes its training-only aux outputs
(attention maps, gate statistics) inside forward(); removing that work is one of the optimizations, not part
of the baseline.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
VIT_DIR = REPO / "ViT_PatchVisionX"
CNN_DIR = REPO / "CNN_Baseline" / "01_strongaug_mixup_cutmix"
for p in (VIT_DIR, CNN_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from patchvision.config import ExperimentConfig  # noqa: E402
from patchvision.models import build_model as build_vit  # noqa: E402
from cnn.config import CNNConfig, _build as _build_cnn_cfg  # noqa: E402
from cnn.model import build_model as build_cnn  # noqa: E402

# run name -> (family, results folder)
RUNS: Dict[str, Tuple[str, Path]] = {
    "base": ("vit", VIT_DIR / "results/base/seed42"),
    "full": ("vit", VIT_DIR / "results/full/seed42"),
    "ovf1_droppath0.3": ("vit", VIT_DIR / "results/ovf1_droppath0.3/seed42"),
    "ovf2_wd0.1": ("vit", VIT_DIR / "results/ovf2_wd0.1/seed42"),
    "ovf3_autoaug": ("vit", VIT_DIR / "results/ovf3_autoaug/seed42"),
    "ovf4_dropout0.1": ("vit", VIT_DIR / "results/ovf4_dropout0.1/seed42"),
    "ovf5_dp0.3_autoaug": ("vit", VIT_DIR / "results/ovf5_dp0.3_autoaug/seed42"),
    "ovf6_all_four": ("vit", VIT_DIR / "results/ovf6_all_four/seed42"),
    "cnn_small": ("cnn", CNN_DIR / "results/cnn_small/seed42"),
    "cnn_large": ("cnn", CNN_DIR / "results/cnn_large/seed42"),
}

# The 4 distinct ARCHITECTURES. ovf1..ovf6 share the `full` architecture (they differ only in training-time
# regularization: drop-path / dropout / weight decay / augmentation, all inactive at inference), so they have
# identical speed. For speed we use one representative run per architecture; the weights do not affect speed.
ARCHITECTURES: Dict[str, str] = {
    "ViT base": "base",
    "PatchVision-X": "ovf5_dp0.3_autoaug",      # best PatchVision-X run (E4 winner); same arch as `full`
    "CNN small": "cnn_small",
    "CNN large": "cnn_large",
}


class LogitsOnly(nn.Module):
    """Uniform interface: forward(x) -> logits for both model families."""

    def __init__(self, model: nn.Module, family: str):
        super().__init__()
        self.model, self.family = model, family

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out[0] if self.family == "vit" else out


@dataclass
class LoadedRun:
    name: str
    family: str
    run_dir: Path
    model: nn.Module          # LogitsOnly wrapper, eval mode, on `device`
    raw: nn.Module            # the unwrapped model
    exp: dict                 # training config saved in the checkpoint
    best_epoch: int


def load_run(name: str, device: torch.device | str = "cuda") -> LoadedRun:
    family, run_dir = RUNS[name]
    ck = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    if family == "vit":
        raw = build_vit(ExperimentConfig.from_dict(ck["exp"]).model)
    else:
        raw = build_cnn(_build_cnn_cfg(CNNConfig, ck["exp"]["model"]))
    raw.load_state_dict(ck["state"], strict=True)
    raw = raw.to(device).eval()
    return LoadedRun(name=name, family=family, run_dir=run_dir, model=LogitsOnly(raw, family).eval(), raw=raw,
                     exp=ck["exp"], best_epoch=int(ck.get("epoch", -1)))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
