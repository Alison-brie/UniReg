
"""Model wrappers used by UniReg.

Primary model aliases:
  - unireg_rpn   : task-conditioned dynamic RPNet/C2F backbone
  - unireg_iirpn : task-conditioned dynamic IIRPNet backbone
  - unireg_mlp   : task-conditioned dynamic CorrMLP backbone

Baselines:
  - rpnet
  - iirpnet
  - corrmlp
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

_THIS_FILE = Path(__file__).resolve()
_PROJ_ROOT = str(_THIS_FILE.parents[2])
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from unireg.core.transforms import SpatialTransformer
from model.dyn_iirpnet import DynIIRPNetBaseline, DynRPNetBaseline, DynCorrMLPBaseline


def _vox_to_norm(flow_vox: torch.Tensor) -> torch.Tensor:
    """Convert voxel-space displacement (dz, dy, dx) to normalized (dx, dy, dz)."""
    D, H, W = flow_vox.shape[2:]
    flow = flow_vox[:, [2, 1, 0]]
    scale = torch.tensor(
        [2.0 / (W - 1), 2.0 / (H - 1), 2.0 / (D - 1)],
        device=flow_vox.device,
        dtype=flow_vox.dtype,
    ).view(1, 3, 1, 1, 1)
    return flow * scale


class RPNetBaseline(nn.Module):
    def __init__(self, compute_size, first_channel=8, shared_encoder=True, **_):
        super().__init__()
        from model.iirp import RPNet
        self.net = RPNet(size=tuple(compute_size), in_channel=1, first_channel=first_channel,
                         shared_encoder=shared_encoder, norm=None)
        self._stn = SpatialTransformer()

    def forward(self, moving, fixed):
        _, flow_vox = self.net(moving, fixed)
        flow_norm = _vox_to_norm(flow_vox)
        return self._stn(moving, flow_norm), flow_norm


class IIRPNetBaseline(nn.Module):
    def __init__(self, compute_size, first_channel=8, shared_encoder=True, **_):
        super().__init__()
        from model.iirp import IIRPNet
        self.net = IIRPNet(size=tuple(compute_size), in_channel=1, first_channel=first_channel,
                           shared_encoder=shared_encoder, norm=None)
        self._stn = SpatialTransformer()

    def forward(self, moving, fixed):
        _, flow_vox = self.net(moving, fixed)
        flow_norm = _vox_to_norm(flow_vox)
        return self._stn(moving, flow_norm), flow_norm


class CorrMLPBaseline(nn.Module):
    def __init__(self, compute_size, shared_encoder=True, **_):
        super().__init__()
        from model.CorrMLP import CorrMLP
        self._arch = CorrMLP(in_channels=1)
        self._stn = SpatialTransformer()

    def forward(self, moving, fixed):
        flow_vox = self._arch(fixed, moving)[1]
        flow_norm = _vox_to_norm(flow_vox)
        return self._stn(moving, flow_norm), flow_norm


_REGISTRY = {
    "unireg_rpn": DynRPNetBaseline,
    "unireg_iirpn": DynIIRPNetBaseline,
    "unireg_mlp": DynCorrMLPBaseline,
    "rpnet": RPNetBaseline,
    "iirpnet": IIRPNetBaseline,
    "corrmlp": CorrMLPBaseline,
}


def build_baseline(arch: str, compute_size: Tuple[int, int, int], device: str = "cuda", **kwargs) -> nn.Module:
    arch = arch.lower()
    if arch not in _REGISTRY:
        raise ValueError(f"Unknown model '{arch}'. Available: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[arch](compute_size, **kwargs).to(device)
