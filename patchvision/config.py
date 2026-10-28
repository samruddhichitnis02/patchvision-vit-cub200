"""Configuration dataclasses and the ablation-suite definitions."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


# ----------------------------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------------------------
@dataclass
class PathsConfig:
    """All locations, relative to `root` (the project folder that holds the notebook)."""
    root: str = "."
    prep_dir: str = "DataPreprocessing_Results"   # train/val/test.csv, norm_stats.json, preprocess_config.json
    cub_dir: str = "CUB_200_2011"                 # images/, bounding_boxes.txt, ...
    results_dir: str = "results"                  # everything we write goes here

    @property
    def root_path(self) -> Path:
        return Path(self.root).resolve()

    @property
    def prep(self) -> Path:
        return self.root_path / self.prep_dir

    @property
    def cub(self) -> Path:
        return self.root_path / self.cub_dir

    @property
    def results(self) -> Path:
        return self.root_path / self.results_dir


# ----------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """PatchVision-X.  Defaults = the full model.  Every novel component has an on/off switch so the
    SAME code produces the ablation rows (switch everything off -> a GvT-style anchor model)."""
    img_size: int = 224
    num_classes: int = 200
    embed_dim: int = 256
    depth: int = 8                    # total transformer blocks
    heads: int = 8
    mlp_ratio: float = 3.0
    drop: float = 0.0
    attn_drop: float = 0.0
    drop_path: float = 0.1            # stochastic depth (max rate, linearly increased)

    # --- inherited from GvT (Sci. Rep. 2025) ---
    tok_groups: int = 4               # group-conv groups in the conv tokenizer (total stride 16 -> 14x14 tokens)
    pool_after: Optional[int] = None  # blocks BEFORE graph pooling (default depth // 2)
    pool_ratio: int = 4               # token reduction at pooling (196 -> 49)
    adj_mode: str = "hadamard"        # "hadamard": A = (A_I + I) * E | "sum": A = (A_I + I) + E
    use_sparse_gate: bool = True      # GvT sparse talking-heads shrinkage gate

    # --- taken from MSCViT (arXiv 2501.06040) and adapted to graph attention ---
    use_lfe: bool = True              # Local Feature Extraction block replaces the absolute position embedding
    keep_abs_pos: bool = False        # also keep the learnable position embedding when use_lfe=True
    use_multiscale: bool = True       # heads are split into groups that attend over different graph scales
    scales: Tuple[int, ...] = (1, 2, 4, 0)   # hop radius per head group; 0 = global (all tokens)

    def effective_scales(self) -> Tuple[int, ...]:
        return tuple(self.scales) if self.use_multiscale else (1,)

    def resolved_pool_after(self) -> int:
        return self.depth // 2 if self.pool_after is None else self.pool_after


# ----------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------
@dataclass
class TrainConfig:
    # --- optimisation (GvT paper recipe) ---
    epochs: int = 300
    batch_size: int = 64
    lr: float = 5e-4
    min_lr: float = 1e-6
    warmup_epochs: int = 10
    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.999)
    grad_clip: float = 1.0
    label_smoothing: float = 0.1
    amp: bool = True

    # --- data (tiers are defined in preprocess_config.json) ---
    aug_tier: str = "strong"          # none | basic | strong | auto
    use_mixup_cutmix: bool = True

    # --- our small-data / anti-overfitting components ---
    use_bbox_loss: bool = True        # R1: bounding-box attention loss (training-time only)
    bbox_weight: float = 0.1
    use_ema: bool = True              # R2: exponential moving average of the weights (used for eval)
    ema_decay: float = 0.999
    use_consistency: bool = True      # R3: flip-equivariance consistency of predictions AND attention maps
    cons_logit_weight: float = 1.0
    cons_attn_weight: float = 1.0
    use_box_crop: bool = True         # R4: bird-box-guided crop augmentation (training-time only)
    box_crop_prob: float = 0.3
    box_crop_margin: Tuple[float, float] = (0.1, 0.4)

    # --- model selection / stopping ---
    best_metric: str = "acc1"         # acc1 | macro_f1 | loss  (selected on VALIDATION only)
    early_stop_patience: int = 60     # epochs without validation improvement (0 = off)
    early_stop_start: int = 30        # never stop before this epoch

    # --- what gets recorded ---
    clean_eval_every: int = 5         # accuracy on un-augmented train images (honest overfitting gauge)
    clean_eval_size: int = 600
    snapshot_every: int = 10          # attention overlays + per-class val report + confusion matrix
    weights_every: int = 50           # keep a weights-only checkpoint every N epochs (0 = off)
    plot_every: int = 10              # refresh curves.png


@dataclass
class ExperimentConfig:
    name: str = "full"
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ExperimentConfig":
        return ExperimentConfig(name=d["name"], seed=d["seed"],
                                model=_build(ModelConfig, d.get("model", {})),
                                train=_build(TrainConfig, d.get("train", {})))


def _build(cls, d: Dict[str, Any]):
    names = {f.name for f in fields(cls)}
    kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in names}
    return cls(**kw)


# ----------------------------------------------------------------------------------------------
# Ablation suite
# ----------------------------------------------------------------------------------------------
_REG_FLAGS = {"bbox": "use_bbox_loss", "ema": "use_ema", "cons": "use_consistency", "boxcrop": "use_box_crop"}
_REGS_OFF = {flag: False for flag in _REG_FLAGS.values()}


def ablation_suite() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """name -> {"model": overrides, "train": overrides}.

    base          GvT-style anchor: conv tokenizer + sparse graph attention + abs. position embedding,
                  no multi-scale heads, none of our regularisers.
    lfe/multiscale/arch_x     architecture additions (arch_x = both = PatchVision-X without regularisers)
    arch_x+<r>    one regulariser at a time on top of arch_x   (r = bbox | ema | cons | boxcrop)
    full          everything on (the proposed PatchVision-X recipe)
    full-<c>      leave-one-out: full model with one component removed
    """
    arch_off = {"use_lfe": False, "use_multiscale": False}
    s: Dict[str, Dict[str, Dict[str, Any]]] = {
        "base": {"model": dict(arch_off), "train": dict(_REGS_OFF)},
        "lfe": {"model": {"use_lfe": True, "use_multiscale": False}, "train": dict(_REGS_OFF)},
        "multiscale": {"model": {"use_lfe": False, "use_multiscale": True}, "train": dict(_REGS_OFF)},
        "arch_x": {"model": {}, "train": dict(_REGS_OFF)},
    }
    for key, flag in _REG_FLAGS.items():
        s[f"arch_x+{key}"] = {"model": {}, "train": {**_REGS_OFF, flag: True}}
    s["full"] = {"model": {}, "train": {}}
    s["full-lfe"] = {"model": {"use_lfe": False}, "train": {}}
    s["full-multiscale"] = {"model": {"use_multiscale": False}, "train": {}}
    s["full-gate"] = {"model": {"use_sparse_gate": False}, "train": {}}
    for key, flag in _REG_FLAGS.items():
        s[f"full-{key}"] = {"model": {}, "train": {flag: False}}
    return s


def get_experiment(name: str, seed: int = 42, model_overrides: Optional[Dict[str, Any]] = None,
                   train_overrides: Optional[Dict[str, Any]] = None) -> ExperimentConfig:
    """Build an ExperimentConfig from a suite name; extra overrides are applied on top."""
    suite = ablation_suite()
    if name not in suite:
        raise KeyError(f"unknown experiment {name!r}. Available: {list(suite)}")
    spec = copy.deepcopy(suite[name])
    m = {**spec["model"], **(model_overrides or {})}
    t = {**spec["train"], **(train_overrides or {})}
    return ExperimentConfig(name=name, seed=seed, model=_build(ModelConfig, m), train=_build(TrainConfig, t))
