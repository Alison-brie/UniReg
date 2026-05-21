"""
unireg/losses/patch_contrast.py
=============================
Patch-level contrastive loss: anchor from one encoder, positive = same patch
from the other encoder, negatives = other random patches.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchContrastiveLoss(nn.Module):
    """
    InfoNCE-style patch contrastive loss between two bottom-level feature maps.

    Anchor = random patches from feat_anchor (e.g. RPNet bottom).
    Positive = same spatial patch from feat_pos (e.g. SAT Nano bottom).
    Negatives = other random patches from feat_pos.

    Both maps are projected to proj_dim before comparison (cosine similarity).
    """

    def __init__(
        self,
        proj_dim: int = 128,
        num_anchors: int = 64,
        num_negatives: int = 16,
        temperature: float = 0.07,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.num_anchors = num_anchors
        self.num_negatives = num_negatives
        self.temperature = temperature
        self.eps = eps

    def forward(
        self,
        feat_anchor: torch.Tensor,
        feat_pos: torch.Tensor,
        proj_anchor: nn.Module,
        proj_pos: nn.Module,
    ) -> torch.Tensor:
        """
        Args:
            feat_anchor: (B, C_anchor, D, H, W) anchor encoder bottom feature.
            feat_pos: (B, C_pos, D, H, W) positive encoder bottom feature (same spatial grid).
            proj_anchor: 1x1 conv mapping C_anchor -> proj_dim.
            proj_pos: 1x1 conv mapping C_pos -> proj_dim.

        Returns:
            Scalar loss (InfoNCE).
        """
        if feat_anchor.shape[2:] != feat_pos.shape[2:]:
            feat_pos = F.interpolate(
                feat_pos, size=feat_anchor.shape[2:],
                mode="trilinear", align_corners=True,
            )

        B, _, D, H, W = feat_anchor.shape
        num_spatial = D * H * W
        if num_spatial < self.num_anchors + self.num_negatives:
            return feat_anchor.new_zeros(1).squeeze()

        a = proj_anchor(feat_anchor)
        p = proj_pos(feat_pos)
        a_flat = a.flatten(2)
        p_flat = p.flatten(2)
        a_flat = F.normalize(a_flat, dim=1, eps=self.eps)
        p_flat = F.normalize(p_flat, dim=1, eps=self.eps)

        indices = torch.randperm(
            num_spatial, device=feat_anchor.device, dtype=torch.long,
        )
        anchor_idx = indices[: self.num_anchors]
        neg_idx = indices[self.num_anchors : self.num_anchors + self.num_negatives]

        anchor_vecs = a_flat[:, :, anchor_idx]
        pos_vecs = p_flat[:, :, anchor_idx]
        neg_vecs = p_flat[:, :, neg_idx]

        anchor_vecs = anchor_vecs.permute(0, 2, 1)
        pos_vecs = pos_vecs.permute(0, 2, 1)
        neg_vecs = neg_vecs.permute(0, 2, 1)

        pos_logits = (anchor_vecs * pos_vecs).sum(dim=-1) / self.temperature
        neg_logits = torch.einsum("bnc,bmc->bnm", anchor_vecs, neg_vecs) / self.temperature
        logits = torch.cat([pos_logits.unsqueeze(-1), neg_logits], dim=-1)
        labels = torch.zeros(
            B, self.num_anchors, dtype=torch.long, device=feat_anchor.device,
        )
        return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
