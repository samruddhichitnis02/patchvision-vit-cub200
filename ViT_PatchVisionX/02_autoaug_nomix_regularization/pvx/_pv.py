"""Makes the shared `patchvision` package importable from this round-02 folder.

Importing this module adds <repo>/Vision_Transformer to sys.path. PatchVision-X round 02 uses the SAME model, data
pipeline and trainer as its round 01 (Vision_Transformer/patchvision), so the extras keep working.
"""
import sys
from pathlib import Path

PV_DIR = Path(__file__).resolve().parents[3] / "Vision_Transformer"
if not (PV_DIR / "patchvision").is_dir():
    raise ImportError(f"{PV_DIR / 'patchvision'} not found")
if str(PV_DIR) not in sys.path:
    sys.path.insert(0, str(PV_DIR))
