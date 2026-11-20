"""Live monitor for a running round-02 CNN training (read-only; uses the ViT study's monitor).

    python watch_training.py cnn_small r0_reference 42
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cnn import _vit  # noqa: F401,E402
from patchvision.monitor import watch  # noqa: E402

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "cnn_small"
    run = sys.argv[2] if len(sys.argv) > 2 else "r0_reference"
    seed = sys.argv[3] if len(sys.argv) > 3 else "42"
    watch(HERE / "results" / model / run / f"seed{seed}", interval=2.0, rows=15)
