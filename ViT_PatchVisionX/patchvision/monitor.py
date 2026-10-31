"""Live view of a running (or finished) training run. Read-only: it only reads train_log.txt / history.csv.

    terminal :  python watch_training.py                      (or:  python watch_training.py full 42)
    notebook :  from patchvision.monitor import watch; watch("results/full/seed42")
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

_STEP_RE = re.compile(
    r"\[E(\d+)/(\d+) \| S(\d+)/(\d+) \| g(\d+)\] loss ([\d.]+) \(ce ([\d.]+) \| bbox ([\d.]+) \| cons ([\d.]+)/([\d.]+)\)"
    r" \| acc1\s+([\d.]+)% .*?in-box ([\d.]+).*?lr ([\d.e+-]+) \| gnorm\s+([\d.]+).*?\|\s+(\d+) img/s.*?mem ([\d.]+) GB")


def _bar(frac: float, width: int = 30) -> str:
    frac = min(max(frac, 0.0), 1.0)
    n = int(round(frac * width))
    return "[" + "#" * n + "-" * (width - n) + f"] {100 * frac:5.1f}%"


def _hms(s: float) -> str:
    s = int(max(0, s))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _last_step(log_path: Path):
    if not log_path.exists():
        return None
    with open(log_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - 40000))
        tail = f.read().decode("utf-8", errors="ignore").splitlines()
    for line in reversed(tail):
        m = _STEP_RE.search(line)
        if m:
            return m
    return None


def render(run_dir, rows: int = 12) -> str:
    rd = Path(run_dir)
    out = [f"PatchVision-X live monitor | {rd}  |  {time.strftime('%H:%M:%S')}", "=" * 118]
    log = rd / "train_log.txt"
    if not log.exists():
        return "\n".join(out + ["waiting for the run to start (train_log.txt not found yet) ..."])
    finished = (rd / "test_metrics.json").exists()
    h = pd.read_csv(rd / "history.csv") if (rd / "history.csv").exists() else pd.DataFrame()
    m = _last_step(log)
    total = int(m.group(2)) if m else (int(h["epoch"].max()) if len(h) else 0)

    if m:
        ep, tot, st, nst = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        out.append(f"epoch {ep:3d}/{tot}  {_bar(st / nst)}  step {st}/{nst}")
        out.append(f"  loss {float(m.group(6)):.3f} (ce {float(m.group(7)):.3f}, bbox {float(m.group(8)):.3f}, "
                   f"cons {float(m.group(9)):.4f}/{float(m.group(10)):.4f}) | batch acc {float(m.group(11)):.1f}% | "
                   f"in-box {float(m.group(12)):.3f} | lr {float(m.group(13)):.2e} | gnorm {float(m.group(14)):.2f} | "
                   f"{m.group(15)} img/s | GPU mem {m.group(16)} GB")
    if len(h):
        done = int(h["epoch"].max())
        per_ep = (h["train_time_s"] + h["val_time_s"]).tail(10).mean()
        remaining = max(0, total - done) * per_ep
        out.append(f"overall {_bar(done / total if total else 0)}  {done}/{total} epochs | "
                   f"{per_ep:.1f} s/epoch | " + ("FINISHED" if finished else f"ETA ~{_hms(remaining)} (less if early-stopped)"))
        out.append("")
        out.append(" epoch | train loss | val loss | val top-1 | raw top-1 | clean-train | gap | val top-5 | val F1 | in-box | lr        | time")
        out.append("-" * 118)
        t = h.tail(rows)
        for _, r in t.iterrows():
            clean = f"{r['clean_acc1']:6.2f}%" if pd.notna(r.get("clean_acc1")) else "      -"
            gap = f"{r['clean_acc1'] - r['val_acc1']:+5.1f}" if pd.notna(r.get("clean_acc1")) else "    -"
            raw = f"{r['valraw_acc1']:6.2f}%" if pd.notna(r.get("valraw_acc1")) else "      -"
            out.append(f" {int(r['epoch']):5d} | {r['train_loss']:10.3f} | {r['val_loss']:8.3f} | {r['val_acc1']:8.2f}% | "
                       f"{raw:>9} | {clean:>11} | {gap:>4} | {r['val_acc5']:8.2f}% | {r['val_macro_f1']:6.2f} | "
                       f"{r.get('val_attn_in_box', float('nan')):6.3f} | {r['train_lr_first']:.2e} | "
                       f"{r['train_time_s']:4.0f}s{'  <== best' if r['is_best'] else ''}")
        b = h.loc[h["val_acc1"].idxmax()]
        out.append("-" * 118)
        out.append(f"best so far: val top-1 {b['val_acc1']:.2f}% at epoch {int(b['epoch'])} | "
                   f"(top-1 columns are in %, 'gap' = clean-train minus val, in points)")
    else:
        out.append("first epoch still running ...")
    if finished:
        j = json.loads((rd / "test_metrics.json").read_text())["row"]
        out += ["", f"TEST RESULT (best epoch {j['best_epoch']}): top-1 {j['test_acc1']:.2f}% | top-5 {j['test_acc5']:.2f}% | "
                    f"macro-F1 {j['test_macro_f1']:.2f}% | ECE {j['test_ece']:.2f}% | params {j['params_M']:.2f}M"]
    return "\n".join(out)


def watch(run_dir, interval: float = 2.0, rows: int = 12) -> None:
    """Refreshes in place until the run finishes (or Ctrl+C / the notebook cell is interrupted)."""
    in_notebook = "ipykernel" in sys.modules
    if in_notebook:
        from IPython.display import clear_output
    else:
        os.system("")                      # enables ANSI escape codes in the Windows console
    try:
        while True:
            text = render(run_dir, rows)
            if in_notebook:
                clear_output(wait=True)
                print(text)
            else:
                sys.stdout.write("\033[H\033[J" + text + "\n")
                sys.stdout.flush()
            if (Path(run_dir) / "test_metrics.json").exists():
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n(monitor stopped; training is NOT affected)")
