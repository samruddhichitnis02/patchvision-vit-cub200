"""
PatchVision-X: a Vision Transformer trained from scratch on small fine-grained data (CUB-200-2011).

Modules
-------
config          dataclass configs (paths, model, training) + the ablation suite definitions
utils           seeding, logging, json helpers, environment record
data            CUB dataset (box-aware), augmentation tiers from preprocess_config.json, mixup/CutMix, loaders
models          the PatchVision-X architecture (ONE model, components switchable for ablations)
losses          soft-target CE, bounding-box attention loss, flip-consistency loss
regularization  EMA weights, early stopping
optim           AdamW parameter groups + warm-up/cosine schedule
metrics         top-k, macro P/R/F1, confusion matrix, calibration (ECE), efficiency
visualize       curves, confusion matrix, reliability diagram, attention overlays
engine          the Trainer (train / validate / checkpoint / log / test)
experiment      run one experiment, run an ablation suite, build the results table
"""
__version__ = "1.0.0"
