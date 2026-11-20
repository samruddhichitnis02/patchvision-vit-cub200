"""Dry run (sanity checks before a long training) for the CNN baselines."""
from __future__ import annotations

import math
from typing import Optional

import torch

from . import _vit  # noqa: F401
from patchvision.data import build_loaders, load_data_info
from patchvision.losses import SoftTargetCrossEntropy, smooth_one_hot
from patchvision.metrics import measure_efficiency
from patchvision.utils import count_parameters, get_device, seed_everything

from .config import CNNExperiment, PathsConfig
from .model import build_model


def dry_run(exp: CNNExperiment, paths: PathsConfig, device: Optional[torch.device] = None) -> dict:
    """Builds the model and pushes ONE real batch through forward/backward.
    Checks: parameter count, output shape, loss at initialisation (~ln(classes)), finite gradients everywhere."""
    device = device or get_device()
    tc, mc = exp.train, exp.model
    info = load_data_info(paths)
    mc.img_size, mc.num_classes = info.img_size, info.num_classes
    seed_everything(exp.seed)
    _, dl = build_loaders(info, tc, exp.seed, num_workers=0, pin_memory=False)
    model = build_model(mc).to(device)
    total_p = count_parameters(model)[0]
    eff = measure_efficiency(model, device, info.img_size, batch_sizes=(1,), iters=3)
    print(f"{exp.name}: {total_p / 1e6:.2f} M parameters | GFLOPs {eff['GFLOPs_per_image']:.2f} | "
          f"latency bs1 {eff['latency_ms_bs1']:.1f} ms")
    x, _m, y, _ = next(iter(dl["train"]))
    x, y = x.to(device), y.to(device)
    print(f"batch: images {tuple(x.shape)} | x mean/std {x.mean().item():.2f}/{x.std().item():.2f}")
    model.train()
    model.zero_grad(set_to_none=True)
    logits = model(x)
    assert logits.shape == (x.shape[0], info.num_classes), f"unexpected logits shape {tuple(logits.shape)}"
    loss = SoftTargetCrossEntropy()(logits.float(), smooth_one_hot(y, info.num_classes, tc.label_smoothing))
    loss.backward()
    bad = [n for n, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    print(f"logits {tuple(logits.shape)} | loss {loss.item():.4f} (chance ln({info.num_classes}) = "
          f"{math.log(info.num_classes):.4f}) | params without finite grad: {bad if bad else 'none [OK]'}")
    assert not bad, f"parameters without a finite gradient: {bad}"
    return {"params_M": total_p / 1e6, "loss": loss.item(), **eff}
