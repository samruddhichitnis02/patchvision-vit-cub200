"""Train one CNN experiment from the terminal (same code path as training.ipynb).

    python train.py                  -> experiment "cnn_small", seed 42
    python train.py cnn_large 42     -> experiment "cnn_large", seed 42

Experiment names are the keys of cnn.config.CNN_SUITE.  Run it from inside CNN_Baseline/ .
Everything is written to CNN_Baseline/results/<experiment>/seed<seed>/ ; re-running resumes an interrupted run.
The shared data (CUB_200_2011/, DataPreprocessing_Results/) is read from the repository root (one level up).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent       # CNN_Baseline/
ROOT = HERE.parent                           # repository root: shared CUB_200_2011/ and DataPreprocessing_Results/
sys.path.insert(0, str(HERE))

from cnn.config import PathsConfig, get_experiment
from cnn.engine import run_experiment

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "cnn_small"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    paths = PathsConfig(root=str(ROOT), results_dir="CNN_Baseline/results")
    run_experiment(get_experiment(name, seed=seed), paths, num_workers=6, step_print_every=1, resume=True)
