"""The ViT for round 02: the project's graph-attention ViT (patchvision/models.py), unchanged.

The ViT returns (logits, aux) where aux holds attention maps for the bird-box and consistency losses. Those losses
are off in round 02 (as for the CNNs), so a thin wrapper returns the logits only and the shared trainer can treat the
ViT exactly like a CNN. The wrapped network is `self.net`, so saved state dicts have keys "net.<...>".
"""
import torch.nn as nn

from . import _pv  # noqa: F401
from patchvision.models import PatchVisionX  # noqa: E402


class LogitsOnly(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = PatchVisionX(cfg)

    def forward(self, x):
        return self.net(x)[0]


def build_model(cfg) -> nn.Module:
    return LogitsOnly(cfg)
