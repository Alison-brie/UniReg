"""
unireg/losses/feature.py
======================
Feature-level losses for training with backbone feature supervision.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureLoss(nn.Module):
    """
    Cosine, L2, or local NCC similarity between feature pyramids produced by the backbone.

    Args:
        loss_type: 'cos' (1 - cosine similarity), 'l2' (MSE), or 'ncc' (local NCC loss)
        ncc_win: side length of the cubic NCC window when loss_type='ncc'
    """

    def __init__(self, loss_type: str = "cos", eps: float = 1e-8, ncc_win: int = 9):
        super().__init__()
        assert loss_type in ("cos", "l2", "ncc"), f"loss_type must be 'cos', 'l2', or 'ncc'"
        self.loss_type = loss_type
        self.eps = eps
        self.ncc_win = int(ncc_win)
        self.win_vol = self.ncc_win ** 3
        filt = torch.ones(1, 1, self.ncc_win, self.ncc_win, self.ncc_win)
        self.register_buffer("filt", filt, persistent=False)

    def _local_ncc_per_channel(self, fx: torch.Tensor, fy: torch.Tensor) -> torch.Tensor:
        """Compute local NCC independently for each channel, then average."""
        b, c, d, h, w = fx.shape
        x = fx.reshape(b * c, 1, d, h, w)
        y = fy.reshape(b * c, 1, d, h, w)
        filt = self.filt.to(device=fx.device, dtype=fx.dtype)
        pad = self.ncc_win // 2

        x2 = x * x
        y2 = y * y
        xy = x * y

        def wsum(t: torch.Tensor) -> torch.Tensor:
            return F.conv3d(t, filt, padding=pad)

        x_sum = wsum(x)
        y_sum = wsum(y)
        x2_sum = wsum(x2)
        y2_sum = wsum(y2)
        xy_sum = wsum(xy)

        win_vol = self.win_vol
        u_x = x_sum / win_vol
        u_y = y_sum / win_vol

        cross = xy_sum - u_y * x_sum - u_x * y_sum + u_x * u_y * win_vol
        x_var = x2_sum - 2.0 * u_x * x_sum + u_x * u_x * win_vol
        y_var = y2_sum - 2.0 * u_y * y_sum + u_y * u_y * win_vol

        cc = cross * cross / (x_var * y_var + 1e-5)
        return -cc.mean()

    def forward(self, fx: torch.Tensor, fy: torch.Tensor) -> torch.Tensor:
        """
        fx, fy: (B, C, D, H, W) feature maps
        """
        if self.loss_type == "cos":
            fx_flat = fx.flatten(2)  # [B,C,N]
            fy_flat = fy.flatten(2)
            norm_x = F.normalize(fx_flat, dim=1, eps=self.eps)
            norm_y = F.normalize(fy_flat, dim=1, eps=self.eps)
            cosine_sim = (norm_x * norm_y).sum(dim=1).mean()
            return 1.0 - cosine_sim
        if self.loss_type == "l2":
            return F.mse_loss(fx, fy)
        return self._local_ncc_per_channel(fx, fy)
