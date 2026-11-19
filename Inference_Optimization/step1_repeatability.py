"""STEP 1 (D) - run-to-run repeatability of batch-1 latency: 5 independent repeats per architecture, interleaved
(model A, B, C, D, A, B, ...) so slow drifts of the shared machine hit every model equally."""
import sys
from pathlib import Path
import pandas as pd
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer.models import ARCHITECTURES, load_run  # noqa: E402
from infer.timing import time_gpu  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "step1_baseline"
runs = {a: load_run(n) for a, n in ARCHITECTURES.items()}
x = torch.randn(1, 3, 224, 224, device="cuda")
rows = []
for rep in range(5):
    for arch, run in runs.items():
        r = time_gpu(run.model, x, warmup=50, rounds=3, iters=100)
        rows.append({"model": arch, "repeat": rep, "batch_size": 1, **r})
        print(f"rep {rep} {arch:14s} median {r['latency_ms_median']:.3f} ms  p95 {r['latency_ms_p95']:.3f}", flush=True)
df = pd.DataFrame(rows)
df.to_csv(OUT / "repeatability_bs1.csv", index=False)
print(df.groupby("model")["latency_ms_median"].agg(["median", "min", "max", "std"]).round(3))
