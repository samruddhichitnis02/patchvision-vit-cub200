"""Live monitor for a running round-02 PatchVision-X training (read-only).

    python watch_training.py r0_reference 42
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pvx import _pv  # noqa: F401,E402
from patchvision.monitor import watch  # noqa: E402

if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else "r0_reference"
    seed = sys.argv[2] if len(sys.argv) > 2 else "42"
    watch(HERE / "results" / "patchvision_x" / run / f"seed{seed}", interval=2.0, rows=15)
