"""Train one round-02 PatchVision-X run from the terminal.

    python train.py r0_reference          -> seed 42
    python train.py r7_sam 42

Run names are the keys of pvx.config.RUNS. Everything is written to results/patchvision_x/<run>/seed<seed>/ ;
re-running resumes an interrupted run from last.pt and skips a finished one. Data is read from the repo root.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent       # ViT_PatchVisionX/02_autoaug_nomix_regularization/
ROOT = HERE.parents[1]                       # repository root
sys.path.insert(0, str(HERE))

from pvx.config import PathsConfig, get_experiment  # noqa: E402
from patchvision.experiment import run_experiment  # noqa: E402

RESULTS = "ViT_PatchVisionX/02_autoaug_nomix_regularization/results"


def train(run: str, seed: int = 42):
    paths = PathsConfig(root=str(ROOT), results_dir=RESULTS)
    return run_experiment(get_experiment(run, seed), paths, num_workers=6, step_print_every=1, resume=True)


if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else "r0_reference"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    train(run, seed)
