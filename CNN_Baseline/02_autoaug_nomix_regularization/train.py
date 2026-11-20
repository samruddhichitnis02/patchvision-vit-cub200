"""Train one round-02 CNN run from the terminal.

    python train.py cnn_small r0_reference          -> seed 42
    python train.py cnn_large r7_sam 42

Run names are the keys of cnn.config.RUNS. Run from inside CNN_Baseline/02_autoaug_nomix_regularization/ .
Everything is written to results/<model>/<run>/seed<seed>/ ; re-running resumes an interrupted run from last.pt
and skips a finished one. The shared data (CUB_200_2011/, DataPreprocessing_Results/) is read from the repo root.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent       # CNN_Baseline/02_autoaug_nomix_regularization/
ROOT = HERE.parents[1]                       # repository root
sys.path.insert(0, str(HERE))

from cnn.config import PathsConfig, get_experiment  # noqa: E402
from cnn.engine import run_experiment  # noqa: E402

RESULTS = "CNN_Baseline/02_autoaug_nomix_regularization/results"


def train(model: str, run: str, seed: int = 42, **overrides):
    paths = PathsConfig(root=str(ROOT), results_dir=RESULTS)
    exp = get_experiment(model, run, seed=seed, **overrides)
    return run_experiment(exp, paths, num_workers=6, step_print_every=1, resume=True)


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "cnn_small"
    run = sys.argv[2] if len(sys.argv) > 2 else "r0_reference"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    train(model, run, seed)
