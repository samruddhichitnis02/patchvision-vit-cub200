"""Losses: soft-target CE, bounding-box attention loss, flip-consistency loss."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TrainConfig


def smooth_one_hot(y: torch.Tensor, num_classes: int, smoothing: float) -> torch.Tensor:
    off = smoothing / num_classes
    t = torch.full((y.shape[0], num_classes), off, device=y.device)
    t.scatter_(1, y.unsqueeze(1), 1.0 - smoothing + off)
    return t


class SoftTargetCrossEntropy(nn.Module):
    def forward(self, logits, soft_targets):
        return torch.sum(-soft_targets * F.log_softmax(logits.float(), dim=-1), dim=-1).mean()


class BBoxAttentionLoss(nn.Module):
    """R1. Pushes attention onto the bird.
        mass = sum_j received_attention_j * box_mask_j   (share of attention inside the box)
        loss = -log(mass), averaged over the pre-pooling blocks and over valid samples.
    A sample is 'valid' if its (augmented) crop still contains at least half a patch of bird.
    The box is used ONLY here (training); the network never receives it as an input."""

    def __init__(self, min_mask_patches: float = 0.5, eps: float = 1e-4):
        super().__init__()
        self.min_mask, self.eps = min_mask_patches, eps

    def forward(self, attn_maps: Sequence[torch.Tensor], patch_mask: torch.Tensor):
        valid = (patch_mask.sum(1) >= self.min_mask).float()
        masses = torch.stack([(a.float() * patch_mask).sum(1) for a in attn_maps], 0).mean(0)
        per_sample = -torch.log(masses.clamp_min(self.eps))
        loss = (per_sample * valid).sum() / valid.sum().clamp_min(1.0)
        return loss, masses.detach(), valid


def flip_tokens(a: torch.Tensor, grid: int) -> torch.Tensor:
    """Horizontally flip a per-token map (B, grid*grid)."""
    return a.view(a.shape[0], grid, grid).flip(-1).reshape(a.shape[0], grid * grid)


def sym_kl_logits(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    la, lb = F.log_softmax(a.float(), -1), F.log_softmax(b.float(), -1)
    return 0.5 * ((la.exp() * (la - lb)).sum(-1) + (lb.exp() * (lb - la)).sum(-1)).mean()


def sym_kl_probs(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.float().clamp_min(eps)
    q = q.float().clamp_min(eps)
    p, q = p / p.sum(-1, keepdim=True), q / q.sum(-1, keepdim=True)
    return 0.5 * ((p * (p / q).log()).sum(-1) + (q * (q / p).log()).sum(-1)).mean()


class TrainingLoss(nn.Module):
    """
    total = CE (+ w_bbox * bbox-attention loss) (+ w_l * KL(pred_A, pred_B) + w_a * KL(attn_A, flip(attn_B)))

    With use_consistency the engine feeds a batch of 2B images = [x, hflip(x)].  Both views share the label,
    so CE is averaged over both; the consistency terms ask the model to be flip-equivariant: the prediction
    must not change and the attention map must flip with the image.  (A free, label-free regulariser:
    it halves the effective hypothesis space the model can overfit to.)
    """

    def __init__(self, tc: TrainConfig, grid: int):
        super().__init__()
        self.tc, self.grid = tc, grid
        self.ce = SoftTargetCrossEntropy()
        self.attn_loss = BBoxAttentionLoss()

    def forward(self, logits, soft_targets, aux, patch_mask, two_views: bool):
        tc = self.tc
        zero = logits.new_zeros(())
        if two_views:
            B = logits.shape[0] // 2
            la, lb = logits[:B], logits[B:]
            ce = 0.5 * (self.ce(la, soft_targets) + self.ce(lb, soft_targets))
            cons_logit = sym_kl_logits(la, lb)
            cons_attn = torch.stack([sym_kl_probs(a[:B], flip_tokens(a[B:], self.grid))
                                     for a in aux["attn_maps"]]).mean()
        else:
            B = logits.shape[0]
            ce = self.ce(logits, soft_targets)
            cons_logit = cons_attn = zero

        if tc.use_bbox_loss:
            attn_l, mass, valid = self.attn_loss(aux["attn_maps"], patch_mask)
        else:                                                    # measured for logging only, no gradient
            with torch.no_grad():
                attn_l, mass, valid = self.attn_loss(aux["attn_maps"], patch_mask)

        total = ce
        if tc.use_bbox_loss:
            total = total + tc.bbox_weight * attn_l
        if two_views and tc.use_consistency:
            total = total + tc.cons_logit_weight * cons_logit + tc.cons_attn_weight * cons_attn
        return {"total": total, "ce": ce.detach(), "attn": attn_l.detach(),
                "cons_logit": cons_logit.detach(), "cons_attn": cons_attn.detach(),
                "mass": mass[:B], "valid": valid[:B]}
