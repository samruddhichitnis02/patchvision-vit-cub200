# Inference optimization

Making PatchVision-X (and the CNN baselines) fast at inference, measured fairly on one GPU.
The full write-up with figures is **[section 10 of the main README](../README.md#10-inference-optimization-making-patchvision-x-fast)**.

## Status

| Step | What | Status |
|---|---|---|
| 1 | Same-GPU baseline, accuracy check, profiler diagnosis | ✅ done |
| 2 | Remove training-only work, FP16/BF16, `torch.compile`, CUDA graphs | next |
| 3 | ONNX, TensorRT | planned |
| 4 | INT8 quantization | planned |
| 5 | Token merging / pruning, distillation | planned |
| 6 | Fused Triton kernel for the graph attention | planned |
| 7 | Serving under load | planned |

**Step 1 in one line:** on an RTX 5090, PatchVision-X takes 9.70 ms per photo against 1.48 ms for the same-size CNN (6.5×).
The GPU is busy only 14 % of that time, because the model launches 668 small kernels per photo. It is
launch-bound, not compute-bound.

## Layout

```
Inference_Optimization/
├── infer/                    shared helpers
│   ├── models.py             load any of the 10 trained runs (ViT or CNN) -> logits-only model
│   ├── data.py               test set decoded + resized once, cached as uint8; accuracy / agreement metrics
│   ├── timing.py             CUDA-event timing (warm-up, rounds, median / p95, memory, GPU state)
│   └── envinfo.py            hardware / software record saved with every result
├── step1_baseline.py         (A) accuracy check of all 10 runs, (B) speed sweep, (C) profiler
├── step1_repeatability.py    5 interleaved repeats of batch-1 latency
├── step1_timeline.py         real GPU kernel timeline of one photo (CUPTI trace)
├── make_figures.py           every figure / GIF in docs/
├── results/step1_baseline/   accuracy.csv, speed.csv, profile.csv, repeatability_bs1.csv, kernel_timeline_bs1.csv, env.json
├── docs/figures, docs/animations
├── cache/                    (git-ignored) pre-processed test set, FP32 reference logits
└── requirements.txt
```

All scripts run from the repository root, for example `python Inference_Optimization/step1_baseline.py`.
They need `CUB_200_2011/` in the repo root and each run's `best_model.pt`. Neither is stored in Git.
