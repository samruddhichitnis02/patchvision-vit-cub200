"""STEP 1 (E) - the real GPU timeline of ONE batch-1 forward pass (PatchVision-X and CNN small).

Kernel start/end times come from the CUPTI device trace (torch.profiler -> Chrome trace, category "kernel"),
i.e. when each kernel actually ran ON THE GPU. Times are relative to the start of the forward pass (CPU side).
Used for the 'where does the time go' animation."""
import json
import sys
import tempfile
from pathlib import Path
import pandas as pd
import torch
from torch.profiler import ProfilerActivity, profile, record_function
sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer.models import ARCHITECTURES, load_run  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "step1_baseline"
rows, summary = [], []
x = torch.randn(1, 3, 224, 224, device="cuda")
for arch in ("PatchVision-X", "CNN small"):
    run = load_run(ARCHITECTURES[arch])
    with torch.inference_mode():
        for _ in range(50):
            run.model(x)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for i in range(3):
                with record_function(f"fwd{i}"):
                    run.model(x)
                torch.cuda.synchronize()
    tmp = Path(tempfile.gettempdir()) / f"trace_{arch.replace(' ', '_')}.json"
    prof.export_chrome_trace(str(tmp))
    ev = json.loads(tmp.read_text())["traceEvents"]
    win = [e for e in ev if e.get("name") == "fwd2" and e.get("ph") == "X" and e.get("cat") == "user_annotation"][0]
    t0, t1 = win["ts"], win["ts"] + win["dur"]
    ks = sorted([e for e in ev if e.get("cat") == "kernel" and e.get("ph") == "X" and t0 <= e["ts"] <= t1 + 3000],
                key=lambda e: e["ts"])
    for e in ks:
        rows.append({"model": arch, "kernel": e["name"][:80], "start_us": e["ts"] - t0, "end_us": e["ts"] + e["dur"] - t0})
    busy = sum(e["dur"] for e in ks) / 1000
    summary.append((arch, len(ks), win["dur"] / 1000, busy))
    print(f"{arch}: {len(ks)} kernels, forward window {win['dur'] / 1000:.2f} ms (under profiler), "
          f"GPU busy {busy:.2f} ms, last kernel ends at {(ks[-1]['ts'] + ks[-1]['dur'] - t0) / 1000:.2f} ms")
pd.DataFrame(rows).to_csv(OUT / "kernel_timeline_bs1.csv", index=False)
