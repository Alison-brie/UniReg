"""
unireg/losses/regularization.py
==============================
Regularisation losses on the predicted deformation field.

IMPORTANT: GradLoss converts normalised flow to voxel displacement before
computing spatial gradients, matching FVM-REG convention.  Without this
conversion the penalty operates on [-1,1] normalised values whose gradients
are ~100x smaller, making regularisation ineffective.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GradLoss(nn.Module):
    """
    Gradient regularisation loss on **voxel-space** displacement.

    The forward method accepts normalised flow (B,3,D,H,W) in [-1,1] and
    internally scales it to voxel displacement before computing the penalty,
    consistent with FVM-REG:
        scale = [(D-1)/2, (H-1)/2, (W-1)/2]
        flow_vox = flow_norm * scale

    Args:
        penalty : 'l1' or 'l2' (default 'l2')
        loss_mult: optional scaling factor
    """

    def __init__(self, penalty: str = "l2", loss_mult: float = 1.0):
        super().__init__()
        assert penalty in ("l1", "l2"), f"penalty must be 'l1' or 'l2', got {penalty!r}"
        self.penalty   = penalty
        self.loss_mult = loss_mult

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        """
        flow: (B, 3, D, H, W) normalised displacement (dx, dy, dz) in [-1, 1].
              Channel order: ch0→x(W), ch1→y(H), ch2→z(D).
        """
        D, H, W = flow.shape[2:]
        scale = torch.tensor(
            [float(W - 1) / 2.0, float(H - 1) / 2.0, float(D - 1) / 2.0],
            device=flow.device, dtype=flow.dtype,
        ).view(1, 3, 1, 1, 1)
        flow_vox = flow * scale

        dy = flow_vox[:, :,  1:, :-1, :-1] - flow_vox[:, :, :-1, :-1, :-1]
        dx = flow_vox[:, :, :-1,  1:, :-1] - flow_vox[:, :, :-1, :-1, :-1]
        dz = flow_vox[:, :, :-1, :-1,  1:] - flow_vox[:, :, :-1, :-1, :-1]

        if self.penalty == "l1":
            d = dy.abs().mean() + dx.abs().mean() + dz.abs().mean()
        else:  # l2
            d = (dy ** 2).mean() + (dx ** 2).mean() + (dz ** 2).mean()

        return d / 3.0 * self.loss_mult


# ── Registration ─────────────────────────────────────────────────────
from unireg.losses.registry import register_reg_loss   # noqa: E402


@register_reg_loss("grad_l2")
def _build_grad_l2(cfg: dict):
    return GradLoss(penalty="l2")


@register_reg_loss("grad_l1")
def _build_grad_l1(cfg: dict):
    return GradLoss(penalty="l1")
