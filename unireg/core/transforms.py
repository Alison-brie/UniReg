"""
unireg/core/transforms.py
=======================
Single source of truth for spatial transforms in this framework.

Flow convention (ALWAYS):
  shape  : (B, 3, D, H, W)
  channel: (dx, dy, dz)  i.e. displacement along (x, y, z) / (W, H, D)
  space  : normalised [-1, 1], align_corners=True

SpatialTransformer.forward(src, flow):
  src  : (B, C, D, H, W)
  flow : (B, 3, D, H, W) normalised displacement
         OR (B, D, H, W, 3) — auto-detected and permuted
  returns warped image same shape as src

compose_flows(flow1, flow2, grid=None):
  Compose: result = flow1(x + flow2(x))
  Both flows are normalised.  Output is also normalised.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from unireg.core.grid import make_norm_grid

Shape3D = Tuple[int, int, int]


class SpatialTransformer(nn.Module):
    """
    Differentiable spatial transformer for 3-D volumes.

    Grid is generated on-the-fly when size changes (no pre-allocated buffer
    with fixed size, so multi-resolution training is fully supported).
    """

    def __init__(self, mode: str = "bilinear"):
        super().__init__()
        self.mode = mode
        self._cached_size: Optional[Tuple] = None
        self._cached_grid: Optional[torch.Tensor] = None

    def _get_grid(self, shape_dhw: Shape3D, device: torch.device) -> torch.Tensor:
        if self._cached_size != shape_dhw or self._cached_grid is None or self._cached_grid.device != device:
            self._cached_grid = make_norm_grid(shape_dhw, device=device)
            self._cached_size = shape_dhw
        return self._cached_grid  # [1,D,H,W,3] (x,y,z)

    def forward(self, src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src  : (B, C, D, H, W)
            flow : (B, 3, D, H, W)  normalised displacement (dx, dy, dz)
                   OR (B, D, H, W, 3) — will be auto-permuted
        Returns:
            warped: same shape as src
        """
        # --- normalise flow to (B, 3, D, H, W) ---
        if flow.dim() == 5 and flow.shape[-1] == 3:
            flow = flow.permute(0, 4, 1, 2, 3).contiguous()  # [B,D,H,W,3]->[B,3,D,H,W]
        assert flow.dim() == 5 and flow.shape[1] == 3, \
            f"[SpatialTransformer] Expected flow (B,3,D,H,W), got {flow.shape}"
        assert src.shape[2:] == flow.shape[2:], \
            f"[SpatialTransformer] src spatial={src.shape[2:]}, flow spatial={flow.shape[2:]}"

        shape_dhw = flow.shape[2:]
        # grid: [1,D,H,W,3] channels (x,y,z); flow: [B,3,D,H,W] channels (dx,dy,dz)
        grid = self._get_grid(shape_dhw, flow.device)  # [1,D,H,W,3]

        # flow permuted to [B,D,H,W,3] for addition with grid
        flow_grid = flow.permute(0, 2, 3, 4, 1)  # [B,D,H,W,3] channels (dx,dy,dz)=(x,y,z) OK

        sample_grid = grid + flow_grid  # [B,D,H,W,3] absolute positions in [-1,1] space
        return F.grid_sample(src, sample_grid, mode=self.mode,
                             align_corners=True, padding_mode="border")


class CompositionTransform(nn.Module):
    """
    Compose two normalised flows:
        composed(x) = flow1(x + flow2(x))

    Both flows are (B, 3, D, H, W) normalised displacement (dx,dy,dz).
    """

    def __init__(self):
        super().__init__()
        self._stn = SpatialTransformer(mode="bilinear")

    def forward(
        self,
        flow1: torch.Tensor,
        flow2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns composed flow (B, 3, D, H, W).
        Semantics: warp by flow2, then by flow1.
        """
        # Warp flow1 by flow2 to get the contribution of flow1 at the displaced position
        flow1_warped = self._stn(flow1, flow2)  # flow1 sampled at (x + flow2(x))
        composed = flow1_warped + flow2
        return composed


def compose_flows(
    flow1: torch.Tensor,
    flow2: torch.Tensor,
    stn: Optional[SpatialTransformer] = None,
) -> torch.Tensor:
    """
    Functional interface for flow composition.
    Returns composed flow = flow1 ∘ flow2.
    """
    if stn is None:
        stn = SpatialTransformer(mode="bilinear")
    flow1_warped = stn(flow1, flow2)
    return flow1_warped + flow2
