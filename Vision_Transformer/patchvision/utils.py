"""Small helpers: seeding, logging, json, environment record."""
from __future__ import annotations

import json
import platform
import random
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """DataLoader worker init. Must live in a module (not the notebook) so Windows `spawn` can pickle it."""
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def fmt_time(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class Logger:
    """print() and append every line to a file (the file always receives everything)."""

    def __init__(self, path, quiet: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.quiet = quiet

    def __call__(self, msg: str, show: bool = True) -> None:
        if show and not self.quiet:
            print(msg, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def _json_default(o: Any):
    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def save_json(obj: Any, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       timeout=5).decode().strip()
    except Exception:
        return "unavailable"


def env_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(), "git_commit": git_commit()}
    try:
        import torchvision
        info["torchvision"] = torchvision.__version__
    except Exception:
        pass
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    return info
