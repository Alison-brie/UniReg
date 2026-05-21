"""
unireg/core/grid.py
=================
Configurable grid generation. Grid size is fully decoupled from image size,
so you can train at one resolution and evaluate at another.

Coordinate conventions used throughout this framework:
  - Voxel grid   : (D, H, W) indexing, channels (z, y, x), range [0, S-1]
  - Normalised grid: (D, H, W) indexing, channels (x, y, z) for grid_sample,
                    range [-1, 1], align_corners=True
"""

from __future__ import annotations

import torch
from typing import Tuple, Optional

Shape3D = Tuple[int, int, int]  # (D, H, W)


def make_voxel_grid(
    shape_dhw: Shape3D,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Returns voxel-space grid [1, D, H, W, 3] with channels (z, y, x).
    Values span [0, S-1] along each axis.
    Used for Jacobian-determinant / folding calculations.
    """
    D, H, W = shape_dhw
    zz, yy, xx = torch.meshgrid(
        torch.arange(D, device=device, dtype=dtype),
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([zz, yy, xx], dim=-1).unsqueeze(0)  # [1,D,H,W,3]


def make_norm_grid(
    shape_dhw: Shape3D,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Returns normalised grid [1, D, H, W, 3] with channels (x, y, z).
    Values span [-1, 1] with align_corners=True.
    This is the grid consumed by torch.nn.functional.grid_sample.

    Channel order is (x, y, z) because grid_sample expects:
      grid[..., 0] -> sample along W (x)
      grid[..., 1] -> sample along H (y)
      grid[..., 2] -> sample along D (z)
    """
    D, H, W = shape_dhw
    z = torch.linspace(-1.0, 1.0, steps=D, device=device, dtype=dtype)
    y = torch.linspace(-1.0, 1.0, steps=H, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, steps=W, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    # stack as (x, y, z) for grid_sample
    return torch.stack([xx, yy, zz], dim=-1).unsqueeze(0)  # [1,D,H,W,3]


def make_grid(
    shape_dhw: Shape3D,
    normalized: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Unified entry point. Returns [1, D, H, W, 3]."""
    if normalized:
        return make_norm_grid(shape_dhw, device=device, dtype=dtype)
    return make_voxel_grid(shape_dhw, device=device, dtype=dtype)


def norm_flow_to_voxel(
    flow_norm: torch.Tensor,
    shape_dhw: Shape3D,
) -> torch.Tensor:
    """
    Convert normalised flow (B, 3, D, H, W) with channels (dx, dy, dz)
    to voxel-space displacement (B, 3, D, H, W) with channels (dz, dy, dx).

    Normalised flow channel order: (dx, dy, dz)
    Voxel displacement channel order: (dz, dy, dx)
    Scale: norm_delta * (S - 1) / 2  for each axis S.
    """
    D, H, W = shape_dhw
    assert flow_norm.shape[1] == 3, f"Expected 3-channel flow, got {flow_norm.shape}"
    dx_n = flow_norm[:, 0]  # W direction
    dy_n = flow_norm[:, 1]  # H direction
    dz_n = flow_norm[:, 2]  # D direction

    dz = dz_n * (D - 1) / 2.0
    dy = dy_n * (H - 1) / 2.0
    dx = dx_n * (W - 1) / 2.0

    return torch.stack([dz, dy, dx], dim=1)  # (B, 3, D, H, W), channels=(dz,dy,dx)


def voxel_flow_to_norm(
    flow_vox: torch.Tensor,
    shape_dhw: Shape3D,
) -> torch.Tensor:
    """
    Convert voxel-space displacement (B, 3, D, H, W) channels (dz, dy, dx)
    to normalised flow (B, 3, D, H, W) channels (dx, dy, dz).
    """
    D, H, W = shape_dhw
    dz = flow_vox[:, 0]
    dy = flow_vox[:, 1]
    dx = flow_vox[:, 2]

    dz_n = dz / ((D - 1) / 2.0)
    dy_n = dy / ((H - 1) / 2.0)
    dx_n = dx / ((W - 1) / 2.0)

    return torch.stack([dx_n, dy_n, dz_n], dim=1)  # (B, 3, D, H, W), channels=(dx,dy,dz)
