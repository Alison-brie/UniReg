from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from unireg.core.grid import make_norm_grid
from unireg.core.transforms import SpatialTransformer, compose_flows


def sam_phi_to_flow_norm(phi: torch.Tensor, target_size=None) -> torch.Tensor:
    """
    Convert same-main SAMCoarse coarse_phi to 3_model normalized displacement.

    Input:
        phi: [B, 3, D, H, W] or [3, D, H, W]
             absolute coordinate field in same-main SAMCoarse convention.

    Output:
        flow_norm: [B, 3, D, H, W]
                   normalized displacement in 3_model convention:
                   channels = dx, dy, dz.
    """
    if phi.dim() == 4:
        phi = phi.unsqueeze(0)

    phi = phi.float()

    if target_size is not None and tuple(phi.shape[2:]) != tuple(target_size):
        phi = F.interpolate(
            phi,
            size=target_size,
            mode="trilinear",
            align_corners=True,
        )

    B, _, D, H, W = phi.shape

    # same-main spatial_transformer:
    # grid = phi.flip(1).permute(0, 2, 3, 4, 1)
    sample_grid = phi.flip(1).permute(0, 2, 3, 4, 1).contiguous()

    identity = make_norm_grid((D, H, W), device=phi.device, dtype=phi.dtype)

    flow_grid = sample_grid - identity
    flow_norm = flow_grid.permute(0, 4, 1, 2, 3).contiguous()

    return flow_norm


class PreAlignWrapper(nn.Module):
    """
    Wrap a registration model.

    Forward:
        moving --pre_flow--> moving_pre
        RPNet(moving_pre, fixed) predicts residual flow
        final_flow = pre_flow composed with residual flow
        warped_final = warp(moving, final_flow)

    Training:
        - similarity loss should use warped_final
        - regularization should use residual flow only
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.stn = SpatialTransformer(mode="bilinear")

    def forward(
        self,
        moving,
        fixed,
        pre_align=None,
        moving_mask=None,
        fixed_mask=None,
        **kwargs,
    ):
        """Forward with optional SAMCoarse pre-alignment.

        Masks are intentionally NOT applied to moving/fixed before the
        pre-align warp or residual registration. They are consumed here so
        they are not forwarded to the base model, and returned in aux for
        masked-similarity loss in Trainer. This preserves full-image context
        while allowing the loss to focus on fixed-mask / valid-overlap regions.
        """
        if pre_align is None:
            return self.base_model(moving, fixed, **kwargs)

        if moving_mask is not None:
            moving_mask = moving_mask.to(device=moving.device, dtype=moving.dtype)
        if fixed_mask is not None:
            fixed_mask = fixed_mask.to(device=fixed.device, dtype=fixed.dtype)

        # Two supported pre-align formats:
        #   - "sam_phi"  : legacy SAMCoarse absolute coordinate field.
        #   - "flow_norm": normalized displacement flow [B,3,D,H,W], used by
        #                  the external pre-align module.
        pre_align_format = str(kwargs.pop("pre_align_format", "sam_phi")).lower()
        if pre_align_format in ("flow_norm", "norm_flow", "normalized_flow"):
            pre_flow = pre_align.to(device=moving.device, dtype=moving.dtype)
            if pre_flow.dim() == 4:
                pre_flow = pre_flow.unsqueeze(0)
            if pre_flow.dim() == 5 and pre_flow.shape[-1] == 3:
                pre_flow = pre_flow.permute(0, 4, 1, 2, 3).contiguous()
            if not (pre_flow.dim() == 5 and pre_flow.shape[1] == 3):
                raise RuntimeError(f"flow_norm pre_align should be [B,3,D,H,W], got {tuple(pre_flow.shape)}")
            if tuple(pre_flow.shape[2:]) != tuple(moving.shape[2:]):
                pre_flow = F.interpolate(
                    pre_flow,
                    size=moving.shape[2:],
                    mode="trilinear",
                    align_corners=True,
                )
        elif pre_align_format in ("sam_phi", "samcoarse", "same"):
            pre_flow = sam_phi_to_flow_norm(
                pre_align.to(moving.device),
                target_size=moving.shape[2:],
            )
        else:
            raise ValueError(f"Unknown pre_align_format: {pre_align_format}")

        # Use full moving image for pre-align warp.
        moving_pre = self.stn(moving, pre_flow)

        # Use full moving_pre and full fixed image for residual registration.
        out = self.base_model(moving_pre, fixed, **kwargs)

        if not isinstance(out, (tuple, list)) or len(out) < 2:
            raise RuntimeError("Base model should return at least (warped, flow).")

        res_flow = out[1]

        final_flow = compose_flows(pre_flow, res_flow, stn=self.stn)
        warped_final = self.stn(moving, final_flow)

        aux = {}
        if len(out) >= 3:
            aux["base_aux"] = out[2:]
        aux["pre_flow"] = pre_flow
        aux["res_flow"] = res_flow
        aux["reg_flow"] = res_flow
        if moving_mask is not None:
            aux["moving_mask"] = moving_mask
        if fixed_mask is not None:
            aux["fixed_mask"] = fixed_mask

        return warped_final, final_flow, aux