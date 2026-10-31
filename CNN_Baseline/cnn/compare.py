"""CNN vs ViT comparison: one table, one figure, one per-class scatter. Works on any finished run folder of either
study, because both write the same files (test_metrics.json, history.csv, test_per_class.csv)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Entry = Tuple[str, Path]          # (label, run_dir)  e.g. ("ViT full", ".../ViT_PatchVisionX/results/full/seed42")

_COLS = [("params_M", "params (M)"), ("GFLOPs", "GFLOPs"), ("test_acc1", "top-1 (%)"), ("test_acc5", "top-5 (%)"),
         ("test_macro_f1", "macro-F1 (%)"), ("test_ece", "ECE (%)"), ("overfit_gap_clean_minus_val", "train-val gap (pts)"),
         ("best_epoch", "best epoch"), ("epochs_run", "epochs run"), ("train_min", "train time (min)"),
         ("sec_per_epoch", "s / epoch"), ("latency_ms_bs1", "latency bs1 (ms)")]


def default_entries(repo_root, seed: int = 42, include_trials: bool = False) -> List[Entry]:
    """The runs of both studies (only those that exist). Order = table order."""
    root = Path(repo_root)
    vit, cnn = root / "ViT_PatchVisionX" / "results", root / "CNN_Baseline" / "results"
    cand = [("CNN small (4.2M)", cnn / "cnn_small"), ("CNN large (11.3M)", cnn / "cnn_large"),
            ("ViT base", vit / "base"), ("PatchVision-X (full)", vit / "full")]
    if include_trials:
        cand += [("ViT full + drop-path 0.3", vit / "ovf1_droppath0.3"), ("ViT full + wd 0.1", vit / "ovf2_wd0.1"),
                 ("ViT full + auto-aug", vit / "ovf3_autoaug"), ("ViT full + dropout 0.1", vit / "ovf4_dropout0.1")]
    out = []
    for label, d in cand:
        rd = d / f"seed{seed}"
        if (rd / "test_metrics.json").exists():
            out.append((label, rd))
        else:
            print(f"[compare] skipped (not finished / not found): {label} -> {rd}")
    return out


def comparison_table(entries: Sequence[Entry]) -> pd.DataFrame:
    rows = []
    for label, rd in entries:
        r = json.loads((Path(rd) / "test_metrics.json").read_text())["row"]
        r = {**r, "train_min": r["train_time_s"] / 60.0}
        rows.append({"model": label, **{name: r.get(k, np.nan) for k, name in _COLS}})
    return pd.DataFrame(rows).set_index("model")


def plot_comparison(entries: Sequence[Entry], path=None, show: bool = False):
    """Left: test top-1 / top-5 / macro-F1. Middle: validation top-1 during training. Right: train-val gap."""
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.2))
    tab = comparison_table(entries)
    n, w = len(tab), 0.26
    x = np.arange(n)
    for i, (col, name) in enumerate([("top-1 (%)", "top-1"), ("top-5 (%)", "top-5"), ("macro-F1 (%)", "macro-F1")]):
        b = ax[0].bar(x + (i - 1) * w, tab[col].values, w, label=name)
        for xi, v in zip(x + (i - 1) * w, tab[col].values):
            ax[0].text(xi, v + 0.6, f"{v:.1f}", ha="center", fontsize=7)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(tab.index, rotation=20, ha="right", fontsize=8)
    ax[0].set_ylabel("%")
    ax[0].set_title("Test set (5,794 photos)")
    ax[0].legend()
    for label, rd in entries:
        h = pd.read_csv(Path(rd) / "history.csv")
        ax[1].plot(h["epoch"], h["val_acc1"], label=label, lw=1.6)
        c = h.dropna(subset=["clean_acc1"])
        ax[2].plot(c["epoch"], c["clean_acc1"] - c["val_acc1"], label=label, lw=1.6)
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("validation top-1 (%)")
    ax[1].set_title("Validation accuracy while training (600 photos)")
    ax[1].legend(fontsize=8)
    ax[2].set_xlabel("epoch")
    ax[2].set_ylabel("clean-train minus validation (points)")
    ax[2].set_title("Memorizing gap (lower = less overfitting)")
    for a in ax[1:]:
        a.grid(alpha=0.3)
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_per_class_scatter(entry_a: Entry, entry_b: Entry, path=None, show: bool = False, top_k: int = 6):
    """One dot per species: F1 of model A (x) vs model B (y). Dots above the diagonal = B is better on that species."""
    (la, da), (lb, db) = entry_a, entry_b
    a = pd.read_csv(Path(da) / "test_per_class.csv").set_index("class_name")["f1"]
    b = pd.read_csv(Path(db) / "test_per_class.csv").set_index("class_name")["f1"]
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    df["diff"] = df["b"] - df["a"]
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.scatter(df["a"], df["b"], s=14, alpha=0.7)
    ax.plot([0, 100], [0, 100], "k--", lw=1)
    for name, r in pd.concat([df.nlargest(top_k, "diff"), df.nsmallest(top_k, "diff")]).iterrows():
        ax.annotate(str(name).split(".")[-1][:22], (r["a"], r["b"]), fontsize=6.5, alpha=0.85)
    ax.set_xlabel(f"F1 per species: {la}")
    ax.set_ylabel(f"F1 per species: {lb}")
    ax.set_title(f"{lb} better on {int((df['diff'] > 0).sum())} of {len(df)} species, "
                 f"{la} on {int((df['diff'] < 0).sum())}")
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return df.sort_values("diff")
