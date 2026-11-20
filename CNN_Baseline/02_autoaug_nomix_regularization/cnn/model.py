"""Compact ResNet written from scratch (no torchvision weights, no pretrained anything).

    stem   7x7 conv /2 + BN + ReLU + 3x3 max-pool /2        224 -> 56
    stage1 BasicBlock x2, width w1, stride 1                  56
    stage2 BasicBlock x2, width w2, stride 2                  28
    stage3 BasicBlock x2, width w3, stride 2                  14
    stage4 BasicBlock x2, width w4, stride 2                   7
    head   global average pool + Linear(w4, 200)

widths (39, 78, 156, 312) -> 4.22 M parameters (matched to the ViT);  (64, 128, 256, 512) -> 11.28 M (ResNet-18 layout).

NEW in round 02: stochastic depth ("drop-path"), the CNN counterpart of the ViT's drop-path.
During TRAINING each residual block's branch is skipped for a random subset of the photos in the batch (the block
then just passes its input through the shortcut), so the network cannot rely on any single block. The skip
probability grows linearly from 0 (first block) to `drop_path` (last block), exactly as in the ViT.
At TEST time every block is always used. drop_path = 0 gives the original 01 model, weight for weight.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import CNNConfig


class DropPath(nn.Module):
    """Stochastic depth, per sample (identical to the ViT's DropPath in patchvision/models.py)."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep

    def extra_repr(self) -> str:
        return f"p={self.p:.3f}"


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int, drop_path: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)
        self.drop_path = DropPath(drop_path)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(self.drop_path(out) + idt)


class CompactResNet(nn.Module):
    def __init__(self, cfg: CNNConfig):
        super().__init__()
        w = cfg.widths
        self.stem = nn.Sequential(nn.Conv2d(3, w[0], 7, 2, 3, bias=False), nn.BatchNorm2d(w[0]),
                                  nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1))
        n_blocks = sum(cfg.blocks)
        dpr = torch.linspace(0, cfg.drop_path, n_blocks).tolist()     # linear rule, as in the ViT
        stages, cin, k = [], w[0], 0
        for i, (c, n) in enumerate(zip(w, cfg.blocks)):
            blocks = []
            for j in range(n):
                blocks.append(BasicBlock(cin, c, stride=(2 if (i > 0 and j == 0) else 1), drop_path=dpr[k]))
                cin, k = c, k + 1
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(w[-1], cfg.num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        if cfg.zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.zeros_(m.bn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stages(self.stem(x))
        return self.head(torch.flatten(self.pool(x), 1))


def build_model(cfg: CNNConfig) -> nn.Module:
    return CompactResNet(cfg)
