"""Weight EMA and early stopping."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


class ModelEMA:
    """R2. Exponential moving average of ALL floating-point state (parameters and BatchNorm statistics).
    The EMA copy is what gets evaluated: it is a smoother, better-generalising solution than the raw weights,
    especially in the noisy late phase of small-data training."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.updates = decay, 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))      # warm-up of the decay
        src = model.state_dict()
        for k, v in self.module.state_dict().items():
            s = src[k].detach()
            if v.dtype.is_floating_point:
                v.mul_(d).add_(s, alpha=1.0 - d)
            else:
                v.copy_(s)

    def state_dict(self) -> dict:
        return {"module": self.module.state_dict(), "updates": self.updates}

    def load_state_dict(self, sd: dict) -> None:
        self.module.load_state_dict(sd["module"])
        self.updates = sd["updates"]


class EarlyStopping:
    """Stops when the monitored validation metric has not improved for `patience` epochs (after `start`)."""

    def __init__(self, patience: int, start: int = 0, mode: str = "max"):
        self.patience, self.start, self.mode = patience, start, mode
        self.best, self.wait = None, 0

    def step(self, value: float, epoch: int) -> bool:
        better = self.best is None or (value > self.best if self.mode == "max" else value < self.best)
        if better:
            self.best, self.wait = value, 0
        else:
            self.wait += 1
        return self.patience > 0 and epoch >= self.start and self.wait >= self.patience

    def state_dict(self) -> dict:
        return {"best": self.best, "wait": self.wait}

    def load_state_dict(self, sd: dict) -> None:
        self.best, self.wait = sd["best"], sd["wait"]
