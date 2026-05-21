"""
unireg/losses/similarity.py
=========================
Similarity losses for deformable registration.

All losses follow:  loss(warped, fixed) -> scalar
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from typing import Optional, Union, Sequence
import math


class NCC3D(nn.Module):
    """
    Local Normalised Cross-Correlation (window-based).

    Args:
        win_size:
            int: cubic window, e.g., 9 -> (9, 9, 9)
            tuple/list: anisotropic window, e.g., (3, 9, 9) for (D, H, W)
        win_vol:
            optional window volume. If None, automatically computed as D*H*W.
    """

    def __init__(
        self,
        win_size: Union[int, Sequence[int]] = 9,
        win_vol: Optional[int] = None,
    ):
        super().__init__()

        if isinstance(win_size, int):
            self.win_size = (int(win_size), int(win_size), int(win_size))
        else:
            assert len(win_size) == 3, \
                f"win_size must be int or length-3 tuple/list, got {win_size}"
            self.win_size = tuple(int(v) for v in win_size)

        assert all(v > 0 for v in self.win_size), \
            f"all win_size values must be positive, got {self.win_size}"

        # 建议使用奇数窗口，否则 padding 后窗口中心不对称
        assert all(v % 2 == 1 for v in self.win_size), \
            f"all win_size values should be odd, got {self.win_size}"

        self.win_vol = int(win_vol) if win_vol is not None else math.prod(self.win_size)

        self._register_sum_filter()

    def _register_sum_filter(self):
        wz, wy, wx = self.win_size  # D, H, W
        filt = torch.ones(1, 1, wz, wy, wx)
        self.register_buffer("filt", filt, persistent=False)

    def forward(self, warped: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            warped, fixed: (B, 1, D, H, W) float tensors
        Returns:
            scalar loss: negative mean local NCC
        """
        assert warped.shape == fixed.shape, \
            f"warped and fixed must have the same shape, got {warped.shape} vs {fixed.shape}"

        filt = self.filt.to(device=warped.device, dtype=warped.dtype)

        # conv3d padding order is (pad_D, pad_H, pad_W)
        pad = tuple(v // 2 for v in self.win_size)

        I = warped
        J = fixed
        I2 = I * I
        J2 = J * J
        IJ = I * J

        def wsum(x):
            return F.conv3d(x, filt, padding=pad)

        I_sum = wsum(I)
        J_sum = wsum(J)
        I2_sum = wsum(I2)
        J2_sum = wsum(J2)
        IJ_sum = wsum(IJ)

        win_vol = float(self.win_vol)

        u_I = I_sum / win_vol
        u_J = J_sum / win_vol

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_vol
        I_var = I2_sum - 2.0 * u_I * I_sum + u_I * u_I * win_vol
        J_var = J2_sum - 2.0 * u_J * J_sum + u_J * u_J * win_vol

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -cc.mean()


class MaskedNCC3D(nn.Module):
    """
    Masked local Normalised Cross-Correlation.

    Args:
        win_size: int or (D,H,W) local window size.
        min_mask_ratio: ignore local windows with too few valid voxels.

    Forward:
        warped, fixed: [B,1,D,H,W]
        mask: [B,1,D,H,W], binary/soft mask. Only valid regions contribute.
    """

    def __init__(
        self,
        win_size: Union[int, Sequence[int]] = 9,
        min_mask_ratio: float = 0.25,
        eps: float = 1e-5,
    ):
        super().__init__()

        if isinstance(win_size, int):
            self.win_size = (int(win_size), int(win_size), int(win_size))
        else:
            assert len(win_size) == 3, \
                f"win_size must be int or length-3 tuple/list, got {win_size}"
            self.win_size = tuple(int(v) for v in win_size)

        assert all(v > 0 for v in self.win_size), \
            f"all win_size values must be positive, got {self.win_size}"
        assert all(v % 2 == 1 for v in self.win_size), \
            f"all win_size values should be odd, got {self.win_size}"

        self.win_vol = int(math.prod(self.win_size))
        self.min_mask_ratio = float(min_mask_ratio)
        self.eps = float(eps)
        self._register_sum_filter()

    def _register_sum_filter(self):
        wz, wy, wx = self.win_size
        filt = torch.ones(1, 1, wz, wy, wx)
        self.register_buffer("filt", filt, persistent=False)

    def forward(self, warped: torch.Tensor, fixed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert warped.shape == fixed.shape, \
            f"warped and fixed must have same shape, got {warped.shape} vs {fixed.shape}"
        if mask.shape != warped.shape:
            if mask.dim() == 4:
                mask = mask.unsqueeze(1)
            if mask.shape != warped.shape:
                raise ValueError(f"mask shape {mask.shape} does not match image shape {warped.shape}")

        filt = self.filt.to(device=warped.device, dtype=warped.dtype)
        pad = tuple(v // 2 for v in self.win_size)

        I = warped
        J = fixed
        M = (mask > 0.5).to(dtype=warped.dtype, device=warped.device)

        def wsum(x):
            return F.conv3d(x, filt, padding=pad)

        M_sum = wsum(M).clamp_min(self.eps)

        I_sum = wsum(I * M)
        J_sum = wsum(J * M)
        I2_sum = wsum(I * I * M)
        J2_sum = wsum(J * J * M)
        IJ_sum = wsum(I * J * M)

        u_I = I_sum / M_sum
        u_J = J_sum / M_sum

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * M_sum
        I_var = I2_sum - 2.0 * u_I * I_sum + u_I * u_I * M_sum
        J_var = J2_sum - 2.0 * u_J * J_sum + u_J * u_J * M_sum

        cc = cross * cross / (I_var * J_var + self.eps)

        valid = (M_sum >= (self.min_mask_ratio * float(self.win_vol))).to(cc.dtype)
        denom = valid.sum().clamp_min(1.0)
        return -(cc * valid).sum() / denom


class MSELoss(nn.Module):
    """Mean squared error (used for same-modality registration)."""

    def forward(self, warped: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(warped, fixed)


class MINDSSCLoss(nn.Module):
    """
    Modality Independent Neighbourhood Descriptor — SSC variant.
    Works for multi-modal registration (CT↔MRI) without paired supervision.
    """

    def __init__(self, win: int = None):
        super().__init__()
        self.win = win

    def _pdist_sq(self, x: torch.Tensor) -> torch.Tensor:
        xx = (x ** 2).sum(1, keepdim=True).permute(0, 2, 1)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy.permute(0, 2, 1) - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
        dist = torch.clamp(dist.nan_to_num(), 0.0)
        return dist

    def _mindssc(self, img: torch.Tensor, radius: int = 2, dilation: int = 2) -> torch.Tensor:
        kernel_size = radius * 2 + 1
        six_nb = torch.tensor(
            [[0,1,1],[1,1,0],[1,0,1],[1,1,2],[2,1,1],[1,2,1]], dtype=torch.long
        )
        dist   = self._pdist_sq(six_nb.t().unsqueeze(0)).squeeze(0)
        x, y   = torch.meshgrid(torch.arange(6), torch.arange(6), indexing="ij")
        mask   = ((x > y).view(-1) & (dist == 2).view(-1))

        idx1   = six_nb.unsqueeze(1).repeat(1,6,1).view(-1,3)[mask]
        idx2   = six_nb.unsqueeze(0).repeat(6,1,1).view(-1,3)[mask]

        dev    = img.device
        dtype  = img.dtype
        n_pairs = mask.sum().item()

        ms1 = torch.zeros(n_pairs, 1, 3, 3, 3, device=dev, dtype=dtype)
        ms1.view(-1)[torch.arange(n_pairs)*27 + idx1[:,0]*9 + idx1[:,1]*3 + idx1[:,2]] = 1
        ms2 = torch.zeros(n_pairs, 1, 3, 3, 3, device=dev, dtype=dtype)
        ms2.view(-1)[torch.arange(n_pairs)*27 + idx2[:,0]*9 + idx2[:,1]*3 + idx2[:,2]] = 1

        rpad1 = nn.ReplicationPad3d(dilation)
        rpad2 = nn.ReplicationPad3d(radius)

        ssd = F.avg_pool3d(
            rpad2((F.conv3d(rpad1(img), ms1, dilation=dilation)
                 - F.conv3d(rpad1(img), ms2, dilation=dilation)) ** 2),
            kernel_size, stride=1,
        )

        mind = ssd - ssd.min(1, keepdim=True)[0]
        mind_var = mind.mean(1, keepdim=True)
        # Clamp variance to avoid division by near-zero or extreme values
        mind_var = torch.clamp(
            mind_var,
            (mind_var.mean() * 0.001).item(),
            (mind_var.mean() * 1000.0).item(),
        )
        mind = torch.exp(-mind / mind_var)
        return mind

    def forward(self, warped: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
        return ((self._mindssc(warped) - self._mindssc(fixed)) ** 2).mean()


# ── Registration ─────────────────────────────────────────────────────
from unireg.losses.registry import register_sim_loss   # noqa: E402


@register_sim_loss("ncc")
def _build_ncc(cfg: dict):
    return NCC3D(win_size=cfg.get("ncc_win", 9))


@register_sim_loss("masked_ncc")
def _build_masked_ncc(cfg: dict):
    return MaskedNCC3D(
        win_size=cfg.get("ncc_win", 9),
        min_mask_ratio=cfg.get("masked_ncc_min_mask_ratio", 0.25),
    )


@register_sim_loss("mse")
def _build_mse(cfg: dict):
    return MSELoss()


@register_sim_loss("mind")
def _build_mind(cfg: dict):
    return MINDSSCLoss()
