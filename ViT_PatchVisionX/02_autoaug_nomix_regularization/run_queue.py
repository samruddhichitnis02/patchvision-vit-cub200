"""Run several round-02 PatchVision-X trainings one after another in ONE terminal, printing every step and epoch.

    python run_queue.py                          -> r0 ... r7
    python run_queue.py r8_combined

Finished runs are skipped and interrupted runs resume, so the same command can simply be started again.
A one-line summary of every finished run is appended to results/patchvision_x/queue_summary.csv.
"""
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train import RESULTS, ROOT, train  # noqa: E402
from pvx.config import MODEL, RUNS  # noqa: E402

if __name__ == "__main__":
    runs = sys.argv[1:] or list(RUNS)
    summary = ROOT / RESULTS / MODEL / "queue_summary.csv"
    print(f"QUEUE for {MODEL}: {runs}", flush=True)
    for i, run in enumerate(runs, 1):
        print(f"\n\n{'#' * 140}\n# QUEUE {i}/{len(runs)}: {MODEL} / {run}   (started {time.strftime('%Y-%m-%d %H:%M:%S')})"
              f"\n{'#' * 140}\n", flush=True)
        try:
            row = train(run)
            summary.parent.mkdir(parents=True, exist_ok=True)
            keep = ["experiment", "best_epoch", "epochs_run", "val_acc1_at_best", "overfit_gap_clean_minus_val",
                    "test_acc1", "test_acc5", "test_macro_f1", "test_ece", "test_attn_in_box", "train_time_s"]
            df = pd.DataFrame([{k: row.get(k) for k in keep}])
            if summary.exists():
                old = pd.read_csv(summary)
                df = pd.concat([old[old.experiment != row["experiment"]], df], ignore_index=True)
            df.to_csv(summary, index=False)
            print(f"\n>>> {MODEL}/{run} finished: best val top-1 {row['val_acc1_at_best']:.2f}% @ epoch "
                  f"{row['best_epoch']} | overfitting gap {row['overfit_gap_clean_minus_val']:.1f} pts | "
                  f"test top-1 {row['test_acc1']:.2f}%", flush=True)
        except Exception:
            print(f"\n!!! {MODEL}/{run} FAILED - continuing with the next run\n{traceback.format_exc()}", flush=True)
    print(f"\nQUEUE DONE ({time.strftime('%Y-%m-%d %H:%M:%S')}). Summary: {summary}", flush=True)
