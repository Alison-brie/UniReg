"""
unireg/core/spaces.py
===================
Utilities for converting between the three coordinate spaces used in this framework:

  RAW space    : original NIfTI volume, voxel indices [0, S-1], physical spacing in mm
  COMPUTE space: resized to compute_size for network forward pass, normalised grid [-1,1]
  PHYSICAL space: real-world mm coordinates for metric evaluation (Dice, TRE)

The network predicts flow in COMPUTE space.  Before metric evaluation the flow
must be mapped back to RAW space (or physical mm).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional

Shape3D = Tuple[int, int, int]  # (D, H, W)


def raw_to_compute(
    img_np: np.ndarray,
    target_size: Shape3D,
    mode: str = "trilinear",
) -> torch.Tensor:
    """
    Resize a raw volume (D, H, W) to compute_size (D', H', W').

    Returns float32 tensor (1, 1, D', H', W').
    """
    t = torch.from_numpy(img_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
    out = F.interpolate(t, size=target_size, mode=mode, align_corners=True)
    return out  # [1,1,D',H',W']


def compute_to_raw(
    img_tensor: torch.Tensor,
    original_size: Shape3D,
    mode: str = "trilinear",
) -> np.ndarray:
    """
    Resize a compute-space tensor (1, 1, D', H', W') back to original_size.

    Returns float32 numpy array (D, H, W).
    """
    out = F.interpolate(img_tensor.float(), size=original_size, mode=mode, align_corners=True)
    return out.squeeze().cpu().numpy().astype(np.float32)


def rescale_flow(
    flow: torch.Tensor,
    target_size: Shape3D,
) -> torch.Tensor:
    """
    Rescale a normalised flow field (B, 3, D, H, W) to a different spatial size.

    Normalised flows have values in [-1, 1] regardless of resolution, so only
    the spatial dimensions are scaled (bilinear interpolation); values unchanged.
    """
    return F.interpolate(flow, size=target_size, mode="trilinear", align_corners=True)


def pts_vox_to_norm(
    pts_xyz: np.ndarray,
    shape_xyz: Tuple[int, int, int],
) -> np.ndarray:
    """
    Convert voxel landmark coords (N, 3) in (x, y, z) order
    to normalised [-1, 1] coords (N, 3).

    shape_xyz = (X, Y, Z) = (W, H, D).
    """
    X, Y, Z = shape_xyz
    pts = pts_xyz.astype(np.float32)
    norm = np.zeros_like(pts)
    norm[:, 0] = 2.0 * pts[:, 0] / max(1, X - 1) - 1.0  # x   -> nx
    norm[:, 1] = 2.0 * pts[:, 1] / max(1, Y - 1) - 1.0  # y   -> ny
    norm[:, 2] = 2.0 * pts[:, 2] / max(1, Z - 1) - 1.0  # z   -> nz
    return norm


def pts_norm_to_vox(
    norm_xyz: np.ndarray,
    shape_xyz: Tuple[int, int, int],
) -> np.ndarray:
    """Inverse of pts_vox_to_norm."""
    X, Y, Z = shape_xyz
    n = norm_xyz.astype(np.float32)
    vox = np.zeros_like(n)
    vox[:, 0] = (n[:, 0] + 1.0) * 0.5 * (X - 1)
    vox[:, 1] = (n[:, 1] + 1.0) * 0.5 * (Y - 1)
    vox[:, 2] = (n[:, 2] + 1.0) * 0.5 * (Z - 1)
    return vox


def vox_to_mm(
    pts_xyz: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    origin_xyz: Tuple[float, float, float] = (0., 0., 0.),
) -> np.ndarray:
    """Convert voxel coords (N, 3) to physical mm (N, 3)."""
    s = np.array(spacing_xyz, dtype=np.float32)
    o = np.array(origin_xyz, dtype=np.float32)
    return pts_xyz.astype(np.float32) * s + o


def mm_to_vox(
    pts_mm: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    origin_xyz: Tuple[float, float, float] = (0., 0., 0.),
) -> np.ndarray:
    """Convert physical mm (N, 3) to voxel coords (N, 3)."""
    s = np.array(spacing_xyz, dtype=np.float32)
    o = np.array(origin_xyz, dtype=np.float32)
    return (pts_mm.astype(np.float32) - o) / s
