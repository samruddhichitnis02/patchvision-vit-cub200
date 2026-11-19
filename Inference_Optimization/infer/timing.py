"""GPU timing done properly.

* CUDA events around every call (measures GPU time, independent of Python timer resolution).
* A long warm-up first (cuDNN autotuning, memory allocator, clocks ramping up from idle).
* Several rounds of many calls; we report the median and p95 over all calls, plus the spread between rounds
  so an unstable measurement (e.g. another user's job on this shared GPU) is visible.
* GPU utilization and SM clock are sampled before each measurement with nvidia-smi and stored with the result.
* Inputs are already on the GPU: this is MODEL latency. Host->device copy and JPEG decoding are excluded
  (they are identical for every model and are measured separately where relevant).
"""
from __future__ import annotations

import statistics
import subprocess
from typing import Callable, Dict, List

import numpy as np
import torch


def gpu_state() -> Dict[str, float]:
    """Utilization %, SM clock MHz, temperature C, power W (nan if nvidia-smi is unavailable)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm,temperature.gpu,power.draw",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        u, c, t, p = [float(v) for v in out.strip().split(",")]
        return {"gpu_util_pct": u, "sm_clock_mhz": c, "temp_c": t, "power_w": p}
    except Exception:
        return {"gpu_util_pct": float("nan"), "sm_clock_mhz": float("nan"), "temp_c": float("nan"),
                "power_w": float("nan")}


@torch.inference_mode()
def time_gpu(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, warmup: int = 50, rounds: int = 3,
             iters: int = 100) -> Dict[str, float]:
    """Times fn(x). Returns latency statistics in ms (per call) and peak memory in MB."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()

    state_before = gpu_state()
    all_ms: List[float] = []
    round_medians: List[float] = []
    for _ in range(rounds):
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            fn(x)
            ends[i].record()
        torch.cuda.synchronize()
        ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
        all_ms += ms
        round_medians.append(statistics.median(ms))
    state_after = gpu_state()

    a = np.asarray(all_ms)
    med = float(np.median(a))
    return {
        "latency_ms_median": med,
        "latency_ms_mean": float(a.mean()),
        "latency_ms_p95": float(np.percentile(a, 95)),
        "latency_ms_p99": float(np.percentile(a, 99)),
        "latency_ms_min": float(a.min()),
        "latency_ms_std": float(a.std()),
        "round_spread_pct": float((max(round_medians) - min(round_medians)) / med * 100),
        "throughput_img_s": x.shape[0] / (med / 1000.0),
        "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20,
        "activation_mem_mb": (torch.cuda.max_memory_allocated() - base_mem) / 2**20,
        "n_calls": len(all_ms),
        "gpu_util_before_pct": state_before["gpu_util_pct"],
        "sm_clock_mhz": state_after["sm_clock_mhz"],
        "temp_c": state_after["temp_c"],
        "power_w": state_after["power_w"],
    }


def count_gflops(model: torch.nn.Module, x: torch.Tensor) -> float:
    """Forward-pass GFLOPs for the given input (PyTorch's FlopCounterMode; 1 multiply-add = 2 FLOPs)."""
    from torch.utils.flop_counter import FlopCounterMode
    with torch.inference_mode(), FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops() / 1e9
