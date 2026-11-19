"""STEP 1 - the baseline: every model measured on the same GPU, same software, same method.

(A) accuracy check  - all 10 trained runs on the 5,794 test photos, in
        * fp32      : plain PyTorch, full precision   (the reference for every later optimization)
        * bf16-amp  : torch.autocast(bfloat16), exactly how the training code evaluated the test set,
                      so these numbers must match the recorded test_metrics.json.
(B) speed           - the 4 distinct architectures, plain PyTorch eager FP32 (PyTorch default settings),
                      batch sizes 1 ... 256: latency (median / p95), throughput, peak GPU memory, params, GFLOPs.

Outputs: results/step1_baseline/{accuracy.csv, speed.csv, env.json} and cache/ref_logits_<run>.pt
Run from the repo root:   python Inference_Optimization/step1_baseline.py [--part acc|speed|all]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer.data import CACHE, Normalizer, accuracy_metrics, evaluate_logits, load_test_set  # noqa: E402
from infer.envinfo import env_info  # noqa: E402
from infer.models import ARCHITECTURES, RUNS, count_params, load_run  # noqa: E402
from infer.timing import count_gflops, time_gpu  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "step1_baseline"
BATCH_SIZES = (1, 8, 32, 64, 128, 256)


def part_accuracy(dev):
    x, y = load_test_set("test")
    print(f"test set: {tuple(x.shape)} uint8, {len(y)} labels", flush=True)
    rows = []
    for name in RUNS:
        run = load_run(name, dev)
        rec = json.loads((run.run_dir / "test_metrics.json").read_text())["row"]
        t0 = time.time()
        lg32 = evaluate_logits(run.model, x, dev)
        t_fp32 = time.time() - t0

        def amp_fn(b, m=run.model):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return m(b)
        lg16 = evaluate_logits(amp_fn, x, dev)
        torch.save(lg32, CACHE / f"ref_logits_{name}.pt")
        m32, m16 = accuracy_metrics(lg32, y), accuracy_metrics(lg16, y)
        rows.append({"run": name, "family": run.family, "best_epoch": run.best_epoch,
                     "recorded_top1": rec["test_acc1"], "recorded_top5": rec["test_acc5"],
                     "recorded_macro_f1": rec["test_macro_f1"],
                     "bf16amp_top1": m16["top1"], "bf16amp_top5": m16["top5"], "bf16amp_macro_f1": m16["macro_f1"],
                     "fp32_top1": m32["top1"], "fp32_top5": m32["top5"], "fp32_macro_f1": m32["macro_f1"],
                     "bf16amp_vs_recorded_top1": m16["top1"] - rec["test_acc1"],
                     "fp32_vs_recorded_top1": m32["top1"] - rec["test_acc1"],
                     "eval_s_fp32": t_fp32})
        r = rows[-1]
        print(f"{name:20s} recorded {r['recorded_top1']:.2f} | bf16-amp {r['bf16amp_top1']:.2f} "
              f"({r['bf16amp_vs_recorded_top1']:+.2f}) | fp32 {r['fp32_top1']:.2f} ({r['fp32_vs_recorded_top1']:+.2f})",
              flush=True)
        del run
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "accuracy.csv", index=False)
    return df


def part_speed(dev):
    rows = []
    for arch, name in ARCHITECTURES.items():
        run = load_run(name, dev)
        params = count_params(run.raw)
        gflops = count_gflops(run.model, torch.randn(1, 3, 224, 224, device=dev))
        for bs in BATCH_SIZES:
            xb = torch.randn(bs, 3, 224, 224, device=dev)
            r = time_gpu(run.model, xb)
            rows.append({"model": arch, "run_used": name, "variant": "pytorch_eager_fp32", "batch_size": bs,
                         "params_M": params / 1e6, "gflops_per_img": gflops, **r})
            print(f"{arch:14s} bs {bs:3d} | median {r['latency_ms_median']:8.3f} ms  p95 {r['latency_ms_p95']:8.3f} | "
                  f"{r['throughput_img_s']:9.1f} img/s | peak {r['peak_mem_mb']:7.0f} MB | "
                  f"spread {r['round_spread_pct']:.1f}% | util-before {r['gpu_util_before_pct']:.0f}%", flush=True)
        del run
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "speed.csv", index=False)
    return df


@torch.inference_mode()
def part_profile(dev, n_iter: int = 30):
    """(C) WHY is it slow? torch.profiler at batch 1 and 64: number of GPU kernels launched per forward pass and
    how much of the wall-clock time the GPU is actually busy (sum of kernel durations / latency).
    A low busy-fraction means the model is launch/overhead-bound (the GPU waits for the CPU), not compute-bound."""
    from torch.profiler import ProfilerActivity, profile
    rows = []
    for arch, name in ARCHITECTURES.items():
        run = load_run(name, dev)
        for bs in (1, 64):
            xb = torch.randn(bs, 3, 224, 224, device=dev)
            for _ in range(30):
                run.model(xb)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                for _ in range(n_iter):
                    run.model(xb)
                torch.cuda.synchronize()
            kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
            kernel_ms = sum(e.device_time for e in kernels) / 1000.0 / n_iter
            lat = time_gpu(run.model, xb, warmup=20, rounds=2, iters=50)["latency_ms_median"]
            rows.append({"model": arch, "batch_size": bs, "kernels_per_forward": len(kernels) / n_iter,
                         "gpu_kernel_ms_per_forward": kernel_ms, "latency_ms_median": lat,
                         "gpu_busy_pct": min(100.0, kernel_ms / lat * 100)})
            r = rows[-1]
            print(f"{arch:14s} bs {bs:3d} | {r['kernels_per_forward']:6.0f} kernels/forward | GPU busy "
                  f"{r['gpu_kernel_ms_per_forward']:.2f} of {lat:.2f} ms = {r['gpu_busy_pct']:.0f}%", flush=True)
        del run
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "profile.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["acc", "speed", "profile", "all"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    (OUT / "env.json").write_text(json.dumps(env_info(), indent=2))
    if args.part in ("acc", "all"):
        part_accuracy(dev)
    if args.part in ("speed", "all"):
        part_speed(dev)
    if args.part in ("profile", "all"):
        part_profile(dev)


if __name__ == "__main__":
    main()
