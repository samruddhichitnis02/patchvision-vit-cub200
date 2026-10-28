"""
patchvision_gvt.py
==================
Everything for the PatchVision experiments EXCEPT the training loop (that lives in training.ipynb):

  1. MODELS
       - "vit"       : plain ViT control (linear patch embedding + CLS token, global attention)
       - "gvt"       : Graph-based Vision Transformer with sparsity
                       (Sci. Rep. 15, 2025, "Graph-based vision transformer with sparsity for
                       training on small datasets from scratch")
       - "gvt_bbox"  : GvT + OUR ADD-ON  -> same network as "gvt", but trained with an extra
                       bounding-box attention loss (boxes are used at TRAINING time only;
                       validation / test / inference use images only).

  2. LOSSES      : soft-target cross-entropy (label smoothing + mixup/cutmix) and the
                   bounding-box attention loss (our add-on).
  3. OPTIMIZER   : AdamW with parameter groups (no weight decay on norms/biases/etc.)
                   + per-step linear-warmup -> cosine-decay learning-rate schedule.
  4. METRICS     : top-1 / top-5, macro precision / recall / F1, weighted F1, per-class report,
                   confusion matrix, attention-in-box share, gate sparsity, params / FLOPs / latency.
  5. DATA        : CUB dataset that returns (image, bird-box mask, label) and applies the SAME
                   geometric augmentation to the image and the mask, the augmentation tiers from
                   preprocess_config.json, and a mixup/CutMix that also mixes the masks.
  6. UTILITIES   : seeding, checkpoint save/load, synthetic-CUB generator for smoke tests.

Paper details that are ambiguous in the text were interpreted as described in the comments marked
"INTERPRETATION".  Nothing here uses pretrained weights.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2 as T


# RandomErasing leaves the bird-box mask untouched (an erased patch is still part of the bird's box): expected.
warnings.filterwarnings("ignore", message=".*RandomErasing.*passing through.*Mask.*")


# ----------------------------------------------------------------------------------------------
# 0. Utilities
# ----------------------------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def fmt_time(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ----------------------------------------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------------------------------------
@dataclass
class ModelConfig:
    variant: str = "gvt_bbox"        # "vit" | "gvt" | "gvt_bbox"
    img_size: int = 224
    patch_size: int = 16             # effective patch size (GvT conv tokenizer also totals 16)
    num_classes: int = 200
    embed_dim: int = 256
    depth: int = 8                   # total transformer blocks
    heads: int = 8
    mlp_ratio: float = 3.0
    drop: float = 0.0                # dropout in MLP / projection
    attn_drop: float = 0.0
    drop_path: float = 0.1           # stochastic depth (max rate, linearly increased)
    # --- GvT specific ---
    pool_after: Optional[int] = None  # number of blocks BEFORE graph pooling (default depth // 2)
    pool_ratio: int = 4               # token reduction at pooling (196 -> 49), as 64 -> 16 in the paper
    tok_groups: int = 4               # groups of the group-conv in the conv tokenizer
    adj_mode: str = "hadamard"        # "hadamard": A = (A_I + I) * E   |  "sum": A = (A_I + I) + E

    def resolved_pool_after(self) -> int:
        return self.depth // 2 if self.pool_after is None else self.pool_after


@dataclass
class TrainConfig:
    epochs: int = 300                 # paper: 300 epochs for transformers
    batch_size: int = 64
    lr: float = 5e-4                  # paper
    min_lr: float = 1e-6
    warmup_epochs: int = 10
    weight_decay: float = 0.05        # paper
    betas: Tuple[float, float] = (0.9, 0.999)
    grad_clip: float = 1.0
    label_smoothing: float = 0.1
    attn_loss_weight: float = 0.1     # lambda of the bbox attention loss (our add-on)
    use_mixup_cutmix: bool = True
    amp: bool = True


# ----------------------------------------------------------------------------------------------
# 2. Model building blocks
# ----------------------------------------------------------------------------------------------
class DropPath(nn.Module):
    """Stochastic depth."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class Mlp(nn.Module):
    def __init__(self, dim: int, ratio: float, drop: float):
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


def grid_adjacency(g: int) -> torch.Tensor:
    """A_I of the paper: 8-directional neighbour relations on a g x g token grid (no self loops)."""
    idx = torch.arange(g * g).view(g, g)
    A = torch.zeros(g * g, g * g)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ys = torch.arange(max(0, -dy), min(g, g - dy))
            xs = torch.arange(max(0, -dx), min(g, g - dx))
            src = idx[ys][:, xs].reshape(-1)
            dst = idx[ys + dy][:, xs + dx].reshape(-1)
            A[src, dst] = 1.0
    return A


def sym_norm(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """D^{-1/2} A D^{-1/2} with D = diag(row sums of A). Works on (..., n, n)."""
    deg = A.sum(-1).clamp_min(eps)
    dinv = deg.pow(-0.5)
    return dinv.unsqueeze(-1) * A * dinv.unsqueeze(-2)


# ----------------------------- Graph attention with sparsity (GvT) ----------------------------
class GraphSparseAttention(nn.Module):
    """
    One GvT attention layer. In plain words:

      1. Build a graph over the tokens: neighbour edges (8-neighbours on the patch grid) weighted by
         feature similarity E.                      A = (A_I + I) * E     (INTERPRETATION below)
      2. Queries and keys are produced by a graph convolution:  Q = Â (X Wq),  K = Â (X Wk)
         with Â = D^-1/2 A D^-1/2 and a separate (d/h x d/h) weight per head.
      3. Attention S = softmax(Q K^T / sqrt(d_h)).
      4. SPARSE talking-heads: a small gate Z (per head, per token) computed from the heads'
         descriptors shrinks weak token links:  S_hat = diag(Z) S diag(Z);  heads then exchange
         information through a learnable h x h matrix Phi:  R = Phi . S_hat.
      5. Values by a second graph convolution guided by the relations R:
                   out_i = (D_R^-1/2 R_i D_R^-1/2) (X_i Wv_i) + X_i
    INTERPRETATION: the paper writes D_cos = 1 - cosine similarity but describes the edges as
    "feature similarity", and writes both a sum and a Hadamard product for combining A_I and E.
    We use cosine SIMILARITY inside a (learnable-temperature) softmax and the Hadamard form by
    default (adj_mode="sum" is available). Row sums are removed by the symmetric normalisation.
    The attention is computed in fp32 even under mixed precision for numerical safety.
    """

    def __init__(self, dim: int, heads: int, n_tokens: int, attn_drop: float, proj_drop: float,
                 adj_mode: str = "hadamard"):
        super().__init__()
        assert dim % heads == 0
        self.dim, self.heads, self.dh, self.n = dim, heads, dim // heads, n_tokens
        self.adj_mode = adj_mode
        dh = self.dh
        self.wq = nn.Parameter(torch.randn(heads, dh, dh) * dh ** -0.5)
        self.wk = nn.Parameter(torch.randn(heads, dh, dh) * dh ** -0.5)
        self.wv = nn.Parameter(torch.randn(heads, dh, dh) * dh ** -0.5)
        # sparse talking-heads parameters
        self.w_h = nn.Parameter(torch.randn(heads, 2 * dh) * (2 * dh) ** -0.5)  # token descriptor
        self.w_c = nn.Parameter(torch.randn(heads, n_tokens) * 0.5)            # W_c in R^{h x n} (gate scale)
        self.gate_slope = nn.Parameter(torch.ones(1, heads, 1))                # u^-1, trainable, init 1
        self.phi = nn.Parameter(torch.eye(heads))                              # head mixing, init identity
        self.log_tau = nn.Parameter(torch.log(torch.tensor(0.1)))              # similarity temperature
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, base_adj: torch.Tensor):
        """
        x        : (B, n, d)   (already layer-normalised)
        base_adj : (n, n) or (B, n, n)  -- A_I + I before pooling, learned adjacency after pooling
        returns  : out (B, n, d), A (B, n, n) final adjacency, aux dict
        """
        in_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            B, n, d = x.shape
            h, dh = self.heads, self.dh
            xh = x.view(B, n, h, dh).permute(0, 2, 1, 3)                      # (B,h,n,dh)

            # --- 1. adjacency: neighbours weighted by feature similarity
            xn = F.normalize(x, dim=-1)
            cos = xn @ xn.transpose(1, 2)                                      # (B,n,n)
            E = torch.softmax(cos / self.log_tau.exp().clamp_min(1e-3), dim=-1)
            base = base_adj if base_adj.ndim == 3 else base_adj.unsqueeze(0)
            A = base * E if self.adj_mode == "hadamard" else base + E          # (B,n,n)
            A_hat = sym_norm(A).unsqueeze(1)                                    # (B,1,n,n)

            # --- 2. graph-convolution queries / keys
            q = A_hat @ torch.einsum("bhnd,hde->bhne", xh, self.wq)
            k = A_hat @ torch.einsum("bhnd,hde->bhne", xh, self.wk)

            # --- 3. attention
            S = torch.softmax((q @ k.transpose(-1, -2)) * dh ** -0.5, dim=-1)  # (B,h,n,n)
            S = self.attn_drop(S)

            # --- 4. sparse talking-heads
            f = torch.einsum("bhnc,hc->bhn", torch.cat([q, k], dim=-1), self.w_h)  # F: (B,h,n)
            # standardise each head's descriptor over tokens so the pooled head-correlation matrix is O(1);
            # without this the gate input is ~1e-3, the gate stays frozen at 0.5 and no sparsity can emerge.
            f = (f - f.mean(-1, keepdim=True)) / (f.std(-1, keepdim=True) + 1e-5)
            C = (f @ f.transpose(1, 2)) / n                                    # bilinear pooling (B,h,h)
            Y = C @ self.w_c                                                    # (B,h,n)
            Z = torch.sigmoid(Y * self.gate_slope)                              # shrinkage gate (B,h,n)
            S_hat = Z.unsqueeze(-1) * S * Z.unsqueeze(-2)                       # M S M
            R = torch.einsum("gh,bhnm->bgnm", self.phi, S_hat).clamp_min(0.0)   # talking heads

            # --- 5. graph-convolution values guided by R (+ residual of the head slice)
            v = torch.einsum("bhnd,hde->bhne", xh, self.wv)
            out = sym_norm(R) @ v + xh                                          # (B,h,n,dh)
            out = out.permute(0, 2, 1, 3).reshape(B, n, d)

            # --- aux: "attention received by each token" (rows of R normalised), used by bbox loss
            Rn = R / R.sum(-1, keepdim=True).clamp_min(1e-6)
            received = Rn.mean(dim=(1, 2))                                      # (B,n), sums to 1
            aux = {
                "received": received,
                "gate_mean": Z.detach().mean(),
                "gate_sparsity": (Z.detach() < 0.1).float().mean(),
            }
        out = self.proj_drop(self.proj(out.to(in_dtype)))
        return out, A, aux


class GraphBlock(nn.Module):
    def __init__(self, dim, heads, n_tokens, mlp_ratio, drop, attn_drop, drop_path, adj_mode):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = GraphSparseAttention(dim, heads, n_tokens, attn_drop, drop, adj_mode)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio, drop)
        self.drop_path = DropPath(drop_path)

    def forward(self, x, base_adj):
        a, A, aux = self.attn(self.norm1(x), base_adj)
        x = x + self.drop_path(a)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, A, aux


class GraphPool(nn.Module):
    """
    Graph pooling n -> t tokens.
    INTERPRETATION: the paper's dimensions (X' = U X^T X, C = U X^T) do not match, so we implement the
    stated *idea*: t learnable queries U (t x d) attend over the n tokens, C = softmax(U K^T) in R^{t x n};
        X' = C X          A' = C A C^T
    """

    def __init__(self, dim: int, n_out: int):
        super().__init__()
        self.U = nn.Parameter(torch.randn(n_out, dim) * dim ** -0.5)
        self.key = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.dim, self.n_out = dim, n_out

    def forward(self, x: torch.Tensor, A: torch.Tensor):
        in_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
            k = self.key(x32)
            C = torch.softmax((self.U.float() @ k.transpose(1, 2)) * self.dim ** -0.5, dim=-1)  # (B,t,n)
            x_new = C @ x32
            A_rn = A / A.sum(-1, keepdim=True).clamp_min(1e-6)
            A_new = C @ A_rn @ C.transpose(1, 2)                                                  # (B,t,t)
            A_new = A_new / A_new.sum(-1, keepdim=True).clamp_min(1e-6)
            A_new = A_new + torch.eye(self.n_out, device=A_new.device).unsqueeze(0)               # self loops
        return self.norm(x_new.to(in_dtype)), A_new


class ConvTokenizer(nn.Module):
    """Two-layer convolutional token embedding with a group convolution (kernel 5), as in the paper.
    224x224 -> (conv 8x8/8) 28x28 -> (group-conv 5x5/2) 14x14 = 196 tokens (an effective 16x16 patch)."""

    def __init__(self, in_ch: int, dim: int, groups: int):
        super().__init__()
        mid = dim // 2
        assert mid % groups == 0 and dim % groups == 0, "embed_dim/2 and embed_dim must be divisible by tok_groups"
        self.conv1 = nn.Conv2d(in_ch, mid, kernel_size=8, stride=8)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(mid, dim, kernel_size=5, stride=2, padding=2, groups=groups)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return x.flatten(2).transpose(1, 2)                                 # (B, n, d)


class GraphViT(nn.Module):
    """GvT: conv tokenizer -> [graph blocks] -> graph pooling -> [graph blocks] -> GAP -> classifier."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        g = cfg.img_size // cfg.patch_size
        self.grid = g
        n0 = g * g
        pre = cfg.resolved_pool_after()
        self.n_pre = pre if 0 < pre < cfg.depth else cfg.depth      # depth < 2  ->  no pooling
        n_post_tokens = max(1, n0 // cfg.pool_ratio)
        self.tokenizer = ConvTokenizer(3, cfg.embed_dim, cfg.tok_groups)
        self.pos_embed = nn.Parameter(torch.zeros(1, n0, cfg.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.register_buffer("base_adj", grid_adjacency(g) + torch.eye(n0), persistent=False)

        dpr = torch.linspace(0, cfg.drop_path, cfg.depth).tolist()
        self.blocks = nn.ModuleList()
        for i in range(cfg.depth):
            n_tok = n0 if i < self.n_pre else n_post_tokens
            self.blocks.append(GraphBlock(cfg.embed_dim, cfg.heads, n_tok, cfg.mlp_ratio, cfg.drop,
                                          cfg.attn_drop, dpr[i], cfg.adj_mode))
        self.pool = GraphPool(cfg.embed_dim, n_post_tokens) if self.n_pre < cfg.depth else None
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        x = self.tokenizer(x) + self.pos_embed
        base = self.base_adj
        attn_maps: List[torch.Tensor] = []
        gate_mean, gate_sp = [], []
        A = None
        for i, blk in enumerate(self.blocks):
            if self.pool is not None and i == self.n_pre:
                x, base = self.pool(x, A)
            x, A, aux = blk(x, base)
            if i < self.n_pre:                       # tokens still map 1:1 to image patches
                attn_maps.append(aux["received"])
            gate_mean.append(aux["gate_mean"]); gate_sp.append(aux["gate_sparsity"])
        x = self.norm(x).mean(dim=1)
        logits = self.head(x)
        return logits, {"attn_maps": attn_maps,
                        "gate_mean": torch.stack(gate_mean).mean(),
                        "gate_sparsity": torch.stack(gate_sp).mean()}


# ------------------------------------- Plain ViT (control) ------------------------------------
class MHSA(nn.Module):
    def __init__(self, dim, heads, attn_drop, proj_drop):
        super().__init__()
        self.h, self.dh = heads, dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, n, d = x.shape
        q, k, v = self.qkv(x).view(B, n, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        attn = self.attn_drop(torch.softmax((q @ k.transpose(-1, -2)) * self.dh ** -0.5, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, n, d)
        return self.proj_drop(self.proj(out)), attn


class ViTBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, drop, attn_drop, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MHSA(dim, heads, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio, drop)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        a, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(a)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn


class PlainViT(nn.Module):
    """Standard ViT built from scratch: linear patch embedding, CLS token, learnable positions."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        g = cfg.img_size // cfg.patch_size
        n0 = g * g
        self.patch_embed = nn.Conv2d(3, cfg.embed_dim, cfg.patch_size, cfg.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n0 + 1, cfg.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02); nn.init.trunc_normal_(self.cls_token, std=0.02)
        dpr = torch.linspace(0, cfg.drop_path, cfg.depth).tolist()
        self.blocks = nn.ModuleList([ViTBlock(cfg.embed_dim, cfg.heads, cfg.mlp_ratio, cfg.drop,
                                              cfg.attn_drop, dpr[i]) for i in range(cfg.depth)])
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)
        self.apply(GraphViT._init)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1) + self.pos_embed
        attn = None
        for blk in self.blocks:
            x, attn = blk(x)
        logits = self.head(self.norm(x)[:, 0])
        cls_attn = attn[:, :, 0, 1:].mean(1)                                    # (B, n_patches), last block
        cls_attn = cls_attn / cls_attn.sum(-1, keepdim=True).clamp_min(1e-6)
        zero = x.new_zeros(())
        return logits, {"attn_maps": [cls_attn.float()], "gate_mean": zero, "gate_sparsity": zero}


def build_model(cfg: ModelConfig) -> nn.Module:
    if cfg.variant == "vit":
        return PlainViT(cfg)
    if cfg.variant in ("gvt", "gvt_bbox"):
        return GraphViT(cfg)
    raise ValueError(f"unknown variant {cfg.variant!r}")


# ----------------------------------------------------------------------------------------------
# 3. Losses
# ----------------------------------------------------------------------------------------------
def smooth_one_hot(y: torch.Tensor, num_classes: int, smoothing: float) -> torch.Tensor:
    off = smoothing / num_classes
    t = torch.full((y.shape[0], num_classes), off, device=y.device)
    t.scatter_(1, y.unsqueeze(1), 1.0 - smoothing + off)
    return t


class SoftTargetCrossEntropy(nn.Module):
    def forward(self, logits, soft_targets):
        return torch.sum(-soft_targets * F.log_softmax(logits.float(), dim=-1), dim=-1).mean()


class BBoxAttentionLoss(nn.Module):
    """
    OUR ADD-ON. Pushes the attention onto the bird.
        mass_b = sum_j  received_attention_j * box_mask_j       (share of attention inside the box)
        loss   = -log(mass)   averaged over the pre-pooling blocks and over valid samples.
    A sample is 'valid' only if its (augmented) crop still contains at least half a patch of bird.
    The box is used ONLY here (training); the network never receives it as an input.
    """

    def __init__(self, min_mask_patches: float = 0.5, eps: float = 1e-4):
        super().__init__()
        self.min_mask, self.eps = min_mask_patches, eps

    def forward(self, attn_maps: Sequence[torch.Tensor], patch_mask: torch.Tensor):
        valid = (patch_mask.sum(1) >= self.min_mask).float()                    # (B,)
        masses = torch.stack([(a.float() * patch_mask).sum(1) for a in attn_maps], 0).mean(0)  # (B,)
        per_sample = -torch.log(masses.clamp_min(self.eps))
        loss = (per_sample * valid).sum() / valid.sum().clamp_min(1.0)
        return loss, masses.detach(), valid


class TrainingLoss(nn.Module):
    """total = soft-target CE  +  lambda * bbox-attention loss (only for the gvt_bbox variant)."""

    def __init__(self, use_bbox_loss: bool, attn_weight: float):
        super().__init__()
        self.use_bbox, self.w = use_bbox_loss, attn_weight
        self.ce = SoftTargetCrossEntropy()
        self.attn_loss = BBoxAttentionLoss()

    def forward(self, logits, soft_targets, aux, patch_mask):
        ce = self.ce(logits, soft_targets)
        if self.use_bbox:
            attn_l, mass, valid = self.attn_loss(aux["attn_maps"], patch_mask)
            total = ce + self.w * attn_l
        else:                                                    # only measured for logging, no gradient
            with torch.no_grad():
                attn_l, mass, valid = self.attn_loss(aux["attn_maps"], patch_mask)
            total = ce
        return {"total": total, "ce": ce.detach(), "attn": attn_l.detach(), "mass": mass, "valid": valid}


# ----------------------------------------------------------------------------------------------
# 4. Optimizer + adaptive (scheduled) learning rate
# ----------------------------------------------------------------------------------------------
def build_optimizer(model: nn.Module, tc: TrainConfig) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (p.ndim <= 1 or name.endswith(".bias") or "pos_embed" in name or "cls_token" in name
                or "gate_slope" in name or "log_tau" in name or "phi" in name or ".U" in name):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [{"params": decay, "weight_decay": tc.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=tc.lr, betas=tc.betas)


def build_scheduler(optimizer, tc: TrainConfig, steps_per_epoch: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Per-STEP schedule: linear warm-up (warmup_epochs) then cosine decay from lr to min_lr."""
    total = tc.epochs * steps_per_epoch
    warm = max(1, tc.warmup_epochs * steps_per_epoch)
    floor = tc.min_lr / tc.lr

    def factor(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        progress = (step - warm) / max(1, total - warm)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def global_grad_norm(model: nn.Module) -> float:
    sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sq += p.grad.detach().float().pow(2).sum().item()
    return math.sqrt(sq)


# ----------------------------------------------------------------------------------------------
# 5. Metrics
# ----------------------------------------------------------------------------------------------
class MetricTracker:
    """Accumulates loss / top-1 / top-5 / confusion matrix over an epoch (hard labels only)."""

    def __init__(self, num_classes: int):
        self.nc = num_classes
        self.reset()

    def reset(self):
        self.loss_sum = 0.0
        self.n = 0
        self.top1 = 0
        self.top5 = 0
        self.conf = torch.zeros(self.nc, self.nc, dtype=torch.long)
        self.mass_sum, self.mass_n = 0.0, 0
        self.area_sum = 0.0
        self.gate_mean_sum, self.gate_sp_sum, self.gate_n = 0.0, 0.0, 0

    @torch.no_grad()
    def update(self, logits, targets, loss: float, mass=None, valid=None, area=None, gate=None):
        b = targets.shape[0]
        k = min(5, logits.shape[1])
        pred_top = logits.topk(k, dim=1).indices
        self.top1 += (pred_top[:, 0] == targets).sum().item()
        self.top5 += (pred_top == targets.unsqueeze(1)).any(1).sum().item()
        self.loss_sum += float(loss) * b
        self.n += b
        idx = targets.cpu() * self.nc + pred_top[:, 0].cpu()
        self.conf += torch.bincount(idx, minlength=self.nc * self.nc).view(self.nc, self.nc)
        if mass is not None and valid is not None and valid.sum() > 0:
            self.mass_sum += (mass * valid).sum().item()
            self.mass_n += int(valid.sum().item())
            if area is not None:
                self.area_sum += (area * valid).sum().item()
        if gate is not None:
            self.gate_mean_sum += float(gate[0]) * b
            self.gate_sp_sum += float(gate[1]) * b
            self.gate_n += b

    def compute(self) -> Dict[str, float]:
        conf = self.conf.double()
        tp = conf.diag()
        support = conf.sum(1)
        pred_cnt = conf.sum(0)
        prec = torch.where(pred_cnt > 0, tp / pred_cnt.clamp_min(1), torch.zeros_like(tp))
        rec = torch.where(support > 0, tp / support.clamp_min(1), torch.zeros_like(tp))
        f1 = torch.where(prec + rec > 0, 2 * prec * rec / (prec + rec).clamp_min(1e-12), torch.zeros_like(tp))
        present = support > 0
        out = {
            "loss": self.loss_sum / max(1, self.n),
            "acc1": 100.0 * self.top1 / max(1, self.n),
            "acc5": 100.0 * self.top5 / max(1, self.n),
            "macro_precision": 100.0 * prec[present].mean().item(),
            "macro_recall": 100.0 * rec[present].mean().item(),
            "macro_f1": 100.0 * f1[present].mean().item(),
            "weighted_f1": 100.0 * (f1 * support).sum().item() / max(1.0, support.sum().item()),
            "n": self.n,
        }
        if self.mass_n:
            out["attn_in_box"] = self.mass_sum / self.mass_n
            out["box_area_frac"] = self.area_sum / self.mass_n
        if self.gate_n:
            out["gate_mean"] = self.gate_mean_sum / self.gate_n
            out["gate_sparsity"] = self.gate_sp_sum / self.gate_n
        return out

    def per_class_report(self, class_names: Optional[Sequence[str]] = None) -> List[dict]:
        conf = self.conf.double()
        tp, support, pred_cnt = conf.diag(), conf.sum(1), conf.sum(0)
        rows = []
        for c in range(self.nc):
            p = (tp[c] / pred_cnt[c]).item() if pred_cnt[c] > 0 else 0.0
            r = (tp[c] / support[c]).item() if support[c] > 0 else 0.0
            f = 2 * p * r / (p + r) if p + r > 0 else 0.0
            rows.append({"class_id": c, "class_name": class_names[c] if class_names else str(c),
                         "precision": 100 * p, "recall": 100 * r, "f1": 100 * f, "support": int(support[c])})
        return rows


def box_to_patch_mask(mask: torch.Tensor, grid: int) -> torch.Tensor:
    """(B,1,H,W) bird mask in [0,1] -> (B, grid*grid) fraction of each patch covered by the box."""
    return F.adaptive_avg_pool2d(mask.float(), grid).flatten(1)


def mix_weights_report(info: dict) -> str:
    return f"{info.get('type', 'none')}(lam={info.get('lam', 1.0):.2f})"


@torch.no_grad()
def measure_efficiency(model: nn.Module, device: torch.device, img_size: int = 224,
                       batch_sizes: Sequence[int] = (1, 64), iters: int = 20) -> Dict[str, float]:
    """Parameters, FLOPs of one forward pass (batch 1), and inference latency / throughput."""
    model.eval()
    out: Dict[str, float] = {"params_M": count_parameters(model)[0] / 1e6}
    try:
        from torch.utils.flop_counter import FlopCounterMode
        x = torch.randn(1, 3, img_size, img_size, device=device)
        with FlopCounterMode(display=False) as fc:
            model(x)
        out["GFLOPs_per_image"] = fc.get_total_flops() / 1e9
    except Exception as e:  # pragma: no cover
        out["GFLOPs_per_image"] = float("nan")
    for bs in batch_sizes:
        x = torch.randn(bs, 3, img_size, img_size, device=device)
        for _ in range(3):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        out[f"latency_ms_bs{bs}"] = dt * 1000
        out[f"throughput_img_s_bs{bs}"] = bs / dt
    return out


# ----------------------------------------------------------------------------------------------
# 6. Data: dataset, augmentation tiers, mixup / cutmix (mask-aware)
# ----------------------------------------------------------------------------------------------
def load_bboxes(path: Optional[os.PathLike]) -> Optional[Dict[int, Tuple[float, float, float, float]]]:
    """CUB bounding_boxes.txt -> {image_id: (x, y, w, h)}. Returns None if the file does not exist."""
    if path is None or not Path(path).exists():
        return None
    boxes = {}
    with open(path) as f:
        for line in f:
            i, x, y, w, h = line.split()
            boxes[int(i)] = (float(x), float(y), float(w), float(h))
    return boxes


class MaskAwareTrivialAugmentWide(T.TrivialAugmentWide):
    """TrivialAugmentWide that also moves the bird-box mask: the SAME sampled op/magnitude is applied to the
    mask when it is geometric (shear / translate / rotate, nearest interpolation); colour ops touch the
    image only. (torchvision's own TrivialAugmentWide refuses masks.)  Uses a few private torchvision hooks;
    if a future torchvision removes them, use the "strong" tier instead."""

    _GEOMETRIC = {"ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate"}

    def forward(self, img, mask):
        h, w = T.functional.get_size(img)
        transform_id, (magnitudes_fn, signed) = self._get_random_item(self._AUGMENTATION_SPACE)
        magnitudes = magnitudes_fn(self.num_magnitude_bins, h, w)
        if magnitudes is not None:
            magnitude = float(magnitudes[int(torch.randint(self.num_magnitude_bins, ()))])
            if signed and torch.rand(()) <= 0.5:
                magnitude *= -1
        else:
            magnitude = 0.0
        img = self._apply_image_or_video_transform(img, transform_id, magnitude,
                                                   interpolation=self.interpolation, fill=self._fill)
        if transform_id in self._GEOMETRIC:
            mask = self._apply_image_or_video_transform(mask, transform_id, magnitude,
                                                        interpolation=T.InterpolationMode.NEAREST, fill=self._fill)
        return img, mask


def build_transforms(tier: str, img_size: int, mean: Sequence[float], std: Sequence[float], train: bool):
    """Augmentation tiers exactly as described in preprocess_config.json. Every geometric op is applied
    jointly to the image and the bird-box mask (torchvision v2 handles tv_tensors.Mask)."""
    size = (img_size, img_size)
    to_float = T.ToDtype({tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.float32, "others": None},
                         scale=True)
    norm = T.Normalize(mean=list(mean), std=list(std))
    if not train or tier == "none":
        return T.Compose([T.Resize(size, antialias=True), to_float, norm])
    if tier == "basic":
        ops = [T.RandomResizedCrop(size, scale=(0.5, 1.0), antialias=True), T.RandomHorizontalFlip(0.5),
               to_float, norm]
    elif tier == "strong":
        ops = [T.RandomResizedCrop(size, scale=(0.35, 1.0), antialias=True), T.RandomHorizontalFlip(0.5),
               T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
               T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.0),
               to_float, norm, T.RandomErasing(p=0.25)]
    elif tier == "auto":
        ops = [T.RandomResizedCrop(size, scale=(0.35, 1.0), antialias=True), T.RandomHorizontalFlip(0.5),
               MaskAwareTrivialAugmentWide(), to_float, norm, T.RandomErasing(p=0.25)]
    else:
        raise ValueError(f"unknown augmentation tier {tier!r}")
    return T.Compose(ops)


class CUBDataset(Dataset):
    """
    Reads train/val/test.csv (columns: image_id, filepath, label, class_name).
    Returns (image[3,H,W] normalised, bird_mask[1,H,W] in {0,1}, label, image_id).
    `root` is the directory that contains the 'data/...' prefix used in the csv filepaths.
    """

    def __init__(self, csv_path, transform, bboxes=None, root: os.PathLike = "."):
        import pandas as pd
        df = pd.read_csv(csv_path)
        self.records = df.to_dict("records")
        self.transform, self.bboxes, self.root = transform, bboxes, Path(root)
        self.class_names = df.drop_duplicates("label").sort_values("label")["class_name"].tolist()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        img = Image.open(self.root / r["filepath"]).convert("RGB")
        w, h = img.size
        mask = torch.zeros(1, h, w, dtype=torch.uint8)
        if self.bboxes is not None and int(r["image_id"]) in self.bboxes:
            x, y, bw, bh = self.bboxes[int(r["image_id"])]
            x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
            x1, y1 = min(w, int(round(x + bw))), min(h, int(round(y + bh)))
            mask[:, y0:y1, x0:x1] = 1
        img_t = tv_tensors.Image(T.functional.pil_to_tensor(img))
        img_t, mask_t = self.transform(img_t, tv_tensors.Mask(mask))
        return img_t, torch.as_tensor(mask_t).float().clamp_(0, 1), int(r["label"]), int(r["image_id"])


class MixupCutmix:
    """
    Batch-level mixup / CutMix (config: mixup_alpha, cutmix_alpha, p_cutmix = probability of picking
    CutMix over mixup). Also mixes the bird-box masks in exactly the same way, so the bbox attention
    loss stays consistent. Returns soft targets that include label smoothing.
    """

    def __init__(self, num_classes: int, mixup_alpha=0.2, cutmix_alpha=1.0, p_cutmix=0.5,
                 label_smoothing=0.1, prob=1.0):
        self.nc, self.ma, self.ca, self.pc, self.ls, self.prob = (num_classes, mixup_alpha, cutmix_alpha,
                                                                   p_cutmix, label_smoothing, prob)

    def __call__(self, x, y, mask):
        t1 = smooth_one_hot(y, self.nc, self.ls)
        info = {"type": "none", "lam": 1.0}
        if np.random.rand() > self.prob:
            return x, t1, mask, info
        x2, y2, m2 = x.flip(0), y.flip(0), mask.flip(0)
        if np.random.rand() < self.pc:                                   # ---- CutMix
            lam = np.random.beta(self.ca, self.ca)
            H, W = x.shape[-2:]
            cut = math.sqrt(1.0 - lam)
            ch, cw = int(H * cut), int(W * cut)
            cy, cx = np.random.randint(H), np.random.randint(W)
            y0, y1 = max(cy - ch // 2, 0), min(cy + ch // 2, H)
            x0, x1 = max(cx - cw // 2, 0), min(cx + cw // 2, W)
            x, mask = x.clone(), mask.clone()
            x[..., y0:y1, x0:x1] = x2[..., y0:y1, x0:x1]
            mask[..., y0:y1, x0:x1] = m2[..., y0:y1, x0:x1]
            lam = 1.0 - ((y1 - y0) * (x1 - x0) / (H * W))
            info = {"type": "cutmix", "lam": lam}
        else:                                                            # ---- Mixup
            lam = float(np.random.beta(self.ma, self.ma))
            x = lam * x + (1 - lam) * x2
            mask = lam * mask + (1 - lam) * m2
            info = {"type": "mixup", "lam": lam}
        t2 = smooth_one_hot(y2, self.nc, self.ls)
        return x, lam * t1 + (1 - lam) * t2, mask, info


# ----------------------------------------------------------------------------------------------
# 7. Checkpoints
# ----------------------------------------------------------------------------------------------
def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best, history, mcfg, tcfg, extra=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None, "epoch": epoch, "best": best,
        "history": history, "model_cfg": asdict(mcfg), "train_cfg": asdict(tcfg), "extra": extra or {},
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ck["model"])
    if optimizer is not None and ck.get("optimizer"):
        optimizer.load_state_dict(ck["optimizer"])
    if scheduler is not None and ck.get("scheduler"):
        scheduler.load_state_dict(ck["scheduler"])
    if scaler is not None and ck.get("scaler"):
        scaler.load_state_dict(ck["scaler"])
    return ck


# ----------------------------------------------------------------------------------------------
# 8. Synthetic CUB (smoke tests only) -- lets you verify the whole pipeline in ~1 minute
# ----------------------------------------------------------------------------------------------
def make_synthetic_cub(out_root, n_classes=8, n_train=12, n_val=2, n_test=4, size=(320, 280), seed=0):
    """Creates <out_root>/data/CUB_200_2011/{images/,bounding_boxes.txt} and
    <out_root>/data/{train,val,test}.csv with the same columns as your real csv files.
    Each class = a coloured ellipse ('bird') on a noisy background, so the task is learnable."""
    import pandas as pd
    from PIL import ImageDraw
    rng = np.random.RandomState(seed)
    out_root = Path(out_root)
    cub = out_root / "data" / "CUB_200_2011"
    palette = rng.randint(40, 255, size=(n_classes, 3))
    rows, boxes, image_id = {"train": [], "val": [], "test": []}, [], 1
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        for c in range(n_classes):
            cname = f"{c + 1:03d}.Class_{c}"
            (cub / "images" / cname).mkdir(parents=True, exist_ok=True)
            for j in range(n):
                W, H = size
                bg = rng.randint(0, 255, size=(H, W, 3)).astype(np.uint8) // 3
                img = Image.fromarray(bg)
                bw, bh = rng.randint(W // 4, W // 2), rng.randint(H // 4, H // 2)
                bx, by = rng.randint(0, W - bw), rng.randint(0, H - bh)
                ImageDraw.Draw(img).ellipse([bx, by, bx + bw, by + bh], fill=tuple(int(v) for v in palette[c]))
                rel = f"data/CUB_200_2011/images/{cname}/{split}_{j}.jpg"
                img.save(out_root / rel, quality=90)
                rows[split].append({"image_id": image_id, "filepath": rel, "label": c,
                                    "class_name": f"Class {c}"})
                boxes.append(f"{image_id} {bx} {by} {bw} {bh}")
                image_id += 1
    (cub / "bounding_boxes.txt").write_text("\n".join(boxes))
    for split in rows:
        pd.DataFrame(rows[split]).sample(frac=1, random_state=seed).to_csv(out_root / "data" / f"{split}.csv",
                                                                           index=False)
    return out_root
