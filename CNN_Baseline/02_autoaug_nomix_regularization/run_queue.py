"""Run several round-02 trainings one after another in ONE terminal, printing every step and every epoch.

    python run_queue.py cnn_small                    -> r0 ... r7 for CNN small (round 1 + round 2)
    python run_queue.py cnn_small r0_reference r7_sam

Finished runs are skipped and interrupted runs resume, so the same command can simply be started again.
A one-line summary of every finished run is appended to results/<model>/queue_summary.csv.
"""
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# import THIS folder's train.py first: importing `cnn` puts Vision_Transformer on sys.path, and that folder has its
# own train.py, which would otherwise be found instead
from train import RESULTS, ROOT, train  # noqa: E402
from cnn.config import RUNS  # noqa: E402

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "cnn_small"
    runs = sys.argv[2:] or list(RUNS)
    summary = ROOT / RESULTS / model / "queue_summary.csv"
    print(f"QUEUE for {model}: {runs}", flush=True)
    for i, run in enumerate(runs, 1):
        print(f"\n\n{'#' * 140}\n# QUEUE {i}/{len(runs)}: {model} / {run}   (started {time.strftime('%Y-%m-%d %H:%M:%S')})"
              f"\n{'#' * 140}\n", flush=True)
        try:
            row = train(model, run)
            summary.parent.mkdir(parents=True, exist_ok=True)
            keep = ["experiment", "best_epoch", "epochs_run", "val_acc1_at_best", "overfit_gap_clean_minus_val",
                    "test_acc1", "test_acc5", "test_macro_f1", "test_ece", "train_time_s"]
            df = pd.DataFrame([{k: row.get(k) for k in keep}])
            if summary.exists():
                old = pd.read_csv(summary)
                df = pd.concat([old[old.experiment != row["experiment"]], df], ignore_index=True)
            df.to_csv(summary, index=False)
            print(f"\n>>> {model}/{run} finished: best val top-1 {row['val_acc1_at_best']:.2f}% @ epoch "
                  f"{row['best_epoch']} | overfitting gap {row['overfit_gap_clean_minus_val']:.1f} pts | "
                  f"test top-1 {row['test_acc1']:.2f}%", flush=True)
        except Exception:
            print(f"\n!!! {model}/{run} FAILED - continuing with the next run\n{traceback.format_exc()}", flush=True)
    print(f"\nQUEUE DONE ({time.strftime('%Y-%m-%d %H:%M:%S')}). Summary: {summary}", flush=True)
