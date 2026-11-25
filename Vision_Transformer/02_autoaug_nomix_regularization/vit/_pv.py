"""Makes the `patchvision` package (data, model, metrics, optimiser) importable from this round-02 folder.

Importing this module (any `from . import _pv`) adds <repo>/Vision_Transformer to sys.path. Nothing in
Vision_Transformer/patchvision is modified.
"""
import sys
from pathlib import Path

PV_DIR = Path(__file__).resolve().parents[2]          # <repo>/Vision_Transformer
if not (PV_DIR / "patchvision").is_dir():
    raise ImportError(f"{PV_DIR / 'patchvision'} not found")
if str(PV_DIR) not in sys.path:
    sys.path.insert(0, str(PV_DIR))
