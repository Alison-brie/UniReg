"""
unireg/core/metrics.py
====================
All evaluation metrics in one place.

Key functions:
  dice_per_label(warped_seg, fixed_seg) -> np.ndarray of per-label Dice
  compute_tre_mm(flow, fixed_pts, moving_pts, spacing, shape) -> np.ndarray [N]
  compute_foldings(flow_norm) -> dict {neg, total, ratio}
  metric_bag(...)  -> dict with all metrics in one call
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List


# ---------------------------------------------------------------------------
# Dice
# ---------------------------------------------------------------------------

def dice_per_label(
    warped_np: np.ndarray,
    fixed_np: np.ndarray,
    labels: Optional[List[int]] = None,
    mode: str = "intersection",
    ignore_zero: bool = True,
) -> np.ndarray:
    """
    Compute Dice coefficient per semantic label.

    Args:
        warped_np   : integer segmentation, any shape
        fixed_np    : integer segmentation, same shape
        labels      : list of labels to evaluate; if None, derived from data
        mode        : 'intersection' (only labels in both) or 'union'
        ignore_zero : skip background label 0

    Returns:
        dices : float32 array [n_labels]  (empty → [0.0])
    """
    warped = warped_np.astype(np.int64)
    fixed  = fixed_np.astype(np.int64)

    u_w = set(np.unique(warped).tolist())
    u_f = set(np.unique(fixed).tolist())

    if labels is None:
        if mode == "intersection":
            labels = sorted(u_w & u_f)
        else:
            labels = sorted(u_w | u_f)

    if ignore_zero and 0 in labels:
        labels = [lb for lb in labels if lb != 0]

    if not labels:
        return np.array([0.0], dtype=np.float32)

    dices = []
    for lb in labels:
        A = (warped == lb)
        B = (fixed  == lb)
        denom = A.sum() + B.sum()
        if denom == 0:
            continue
        dices.append(2.0 * float((A & B).sum()) / float(denom))

    return np.array(dices if dices else [0.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# TRE (mm)
# ---------------------------------------------------------------------------

def _pts_to_norm(pts_xyz: torch.Tensor, shape_xyz: Tuple[int, int, int]) -> torch.Tensor:
    """Normalise voxel points (N,3) to [-1,1].  shape_xyz = (X, Y, Z) = (W, H, D)."""
    X, Y, Z = shape_xyz
    nx = 2.0 * pts_xyz[:, 0] / max(1, X - 1) - 1.0
    ny = 2.0 * pts_xyz[:, 1] / max(1, Y - 1) - 1.0
    nz = 2.0 * pts_xyz[:, 2] / max(1, Z - 1) - 1.0
    # grid_sample expects (x, y, z) last dim, where x→W, y→H, z→D
    return torch.stack([nx, ny, nz], dim=-1)  # [N,3]


def _denorm_pts(norm_xyz: torch.Tensor, shape_xyz: Tuple[int, int, int]) -> torch.Tensor:
    X, Y, Z = shape_xyz
    x = (norm_xyz[:, 0] + 1.0) * 0.5 * (X - 1)
    y = (norm_xyz[:, 1] + 1.0) * 0.5 * (Y - 1)
    z = (norm_xyz[:, 2] + 1.0) * 0.5 * (Z - 1)
    return torch.stack([x, y, z], dim=-1)


def compute_tre_mm(
    flow: torch.Tensor,
    fixed_pts_np: np.ndarray,
    moving_pts_np: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    shape_xyz: Tuple[int, int, int],
    device: str = "cuda",
) -> np.ndarray:
    """
    Compute Target Registration Error in millimetres.

    flow          : (1, 3, D, H, W) normalised displacement (dx,dy,dz); on GPU
    fixed_pts_np  : (N, 3) voxel coords (x,y,z) of fixed landmarks
    moving_pts_np : (N, 3) voxel coords (x,y,z) of moving landmarks (ground truth)
    spacing_xyz   : (sx, sy, sz) mm/voxel
    shape_xyz     : (X, Y, Z) = (W, H, D) image size in voxels

    Returns:
        tre : (N,) float32 array of per-landmark TRE in mm
    """
    # 保证为 Python 标量，避免 DataLoader 把 shape/spacing 变成 CPU tensor 导致设备冲突
    shape_xyz = tuple(int(x.item()) if torch.is_tensor(x) else int(x) for x in shape_xyz)
    spacing_xyz = tuple(float(x.item()) if torch.is_tensor(x) else float(x) for x in spacing_xyz)
    # DataLoader batch_size=1 时会把 (N,3) 变成 (1,N,3)，需去掉首维，保证为 (N,3)
    fixed_pts_np = np.asarray(fixed_pts_np, dtype=np.float32)
    moving_pts_np = np.asarray(moving_pts_np, dtype=np.float32)
    if fixed_pts_np.ndim == 3:
        fixed_pts_np = fixed_pts_np.squeeze(0)
    if moving_pts_np.ndim == 3:
        moving_pts_np = moving_pts_np.squeeze(0)
    flow = flow.to(device)
    fix_t = torch.from_numpy(fixed_pts_np).to(device)   # [N,3]
    mov_t = torch.from_numpy(moving_pts_np).to(device)  # [N,3]

    N = fix_t.shape[0]

    # Normalise fixed points for grid_sample
    fix_norm = _pts_to_norm(fix_t, shape_xyz)  # [N,3]
    grid = fix_norm.view(1, N, 1, 1, 3)        # [1,N,1,1,3]

    # Sample flow at fixed landmark positions
    # flow: [1,3,D,H,W] -> disp [1,3,N,1,1]
    disp = F.grid_sample(flow, grid, mode="bilinear", padding_mode="border", align_corners=True)
    disp = disp.view(3, N).permute(1, 0)  # [N,3] channels (dx,dy,dz)

    # Predicted moving position in normalised space
    pred_norm = fix_norm + disp  # [N,3]

    # Convert back to voxel space
    pred_vox = _denorm_pts(pred_norm, shape_xyz)  # [N,3]

    # Difference in voxel space, then mm
    diff_vox = pred_vox - mov_t                  # [N,3]
    sx, sy, sz = spacing_xyz
    diff_mm = diff_vox * torch.tensor([sx, sy, sz], device=device)
    tre = torch.norm(diff_mm, dim=1)             # [N]
    return tre.detach().cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Foldings (Jacobian determinant < 0)
# ---------------------------------------------------------------------------

def compute_foldings(
    flow_norm: torch.Tensor,
) -> dict:
    """
    Compute the number and ratio of folding voxels (Jdet < 0).

    Uses forward-difference Jacobian identical to referma/train_uni.py,
    producing output of shape (D-1, H-1, W-1).

    flow_norm : (B, 3, D, H, W) normalised flow (dx,dy,dz) = (x,y,z);
                only batch index 0 is used.

    Returns dict with keys: neg (count), total, ratio (float).
    """
    flow_cpu = flow_norm.detach().cpu()[0].permute(1, 2, 3, 0).numpy().astype(np.float32)
    D, H, W, _ = flow_cpu.shape

    flow_zyx = np.empty_like(flow_cpu, dtype=np.float32)
    flow_zyx[..., 0] = flow_cpu[..., 2] * (D - 1) / 2.0   # z
    flow_zyx[..., 1] = flow_cpu[..., 1] * (H - 1) / 2.0   # y
    flow_zyx[..., 2] = flow_cpu[..., 0] * (W - 1) / 2.0   # x

    zz, yy, xx = np.meshgrid(
        np.arange(D, dtype=np.float32),
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij",
    )
    grid_zyx = np.stack([zz, yy, xx], axis=-1)
    J = flow_zyx + grid_zyx  # absolute coordinate field [D,H,W,3] (z,y,x)

    dz = J[1:, :-1, :-1, :] - J[:-1, :-1, :-1, :]
    dy = J[:-1, 1:, :-1, :] - J[:-1, :-1, :-1, :]
    dx = J[:-1, :-1, 1:, :] - J[:-1, :-1, :-1, :]

    Jdet = (
        dz[..., 0] * (dy[..., 1] * dx[..., 2] - dy[..., 2] * dx[..., 1])
      - dz[..., 1] * (dy[..., 0] * dx[..., 2] - dy[..., 2] * dx[..., 0])
      + dz[..., 2] * (dy[..., 0] * dx[..., 1] - dy[..., 1] * dx[..., 0])
    )

    neg = int(np.count_nonzero(Jdet < 0))
    total = int(Jdet.size)
    ratio = neg / max(1, total)
    return {"neg": neg, "total": total, "ratio": ratio}


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def metric_bag(
    warped_seg: Optional[np.ndarray],
    fixed_seg: Optional[np.ndarray],
    flow: Optional[torch.Tensor] = None,
    pts_fix: Optional[np.ndarray] = None,
    pts_mov: Optional[np.ndarray] = None,
    spacing_xyz: Optional[Tuple] = None,
    shape_xyz: Optional[Tuple] = None,
    device: str = "cuda",
) -> dict:
    """
    Compute all available metrics in one call.

    Returns dict with keys (only present when inputs provided):
      'dice_mean', 'dice_std', 'tre_mean', 'tre_std', 'neg_jac', 'jac_ratio'
    """
    result = {}

    if warped_seg is not None and fixed_seg is not None:
        d = dice_per_label(warped_seg, fixed_seg)
        result["dice_mean"] = float(d.mean())
        result["dice_std"]  = float(d.std())

    if flow is not None:
        fd = compute_foldings(flow)
        result["neg_jac"]   = fd["neg"]
        result["jac_ratio"] = fd["ratio"]

    if (flow is not None and pts_fix is not None and pts_mov is not None
            and spacing_xyz is not None and shape_xyz is not None):
        tre = compute_tre_mm(flow, pts_fix, pts_mov, spacing_xyz, shape_xyz, device=device)
        result["tre_mean"] = float(tre.mean())
        result["tre_std"]  = float(tre.std())

    return result
