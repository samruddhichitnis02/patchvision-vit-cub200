"""Exact hardware / software used for a benchmark, saved next to every result."""
from __future__ import annotations

import platform
import subprocess

import torch


def _smi(q: str) -> str:
    try:
        return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return "unavailable"


def env_info() -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "driver": _smi("driver_version"),
        "gpu_max_sm_clock_mhz": _smi("clocks.max.sm"),
        "gpu_power_limit": _smi("power.limit"),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    for mod in ("torchvision", "onnx", "onnxruntime", "tensorrt", "triton"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = "not installed"
    return info
