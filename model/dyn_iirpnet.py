"""
dyn_iirpnet.py
================
Task-type and task-id conditioned Dynamic IIRPNet.

Recommended location in your project:
    model/dyn_iirpnet.py  (matches current registry import)
    or unireg/models/dyn_iirpnet.py if registry imports from unireg.models

Minimal registry change:
    from unireg.models.dyn_iirpnet import DynIIRPNetBaseline
    _REGISTRY["dyn_iirpnet"] = DynIIRPNetBaseline

Condition definition:
    task_type: inter-subject / intra-subject
    task_id:   chest / abdomen / head / liver / cardiac / brain

The raw DynIIRPNet returns voxel-space flow, following model.iirp.IIRPNet.
DynIIRPNetBaseline wraps it and returns normalized flow, following current unireg baseline convention.
"""

import sys
from pathlib import Path
from typing import Optional, Union

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make project root importable when this file is placed under unireg/models/.
_THIS_FILE = Path(__file__).resolve()
_PROJ_ROOT = str(_THIS_FILE.parents[2]) if len(_THIS_FILE.parents) >= 3 else str(Path.cwd())
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from model.iirp import (  # type: ignore
    Encoder,
    SpatialTransformer as VoxelSpatialTransformer,
    ConvBlock,
    ConvResBlock,
    LoadableModel,
    store_config_args,
)

try:
    # Current framework unified normalized-grid transformer.
    from unireg.core.transforms import SpatialTransformer as NormalizedSpatialTransformer
except Exception:  # pragma: no cover
    NormalizedSpatialTransformer = None


class DynamicVoxelSpatialTransformer(nn.Module):
    """Voxel-space transformer with an on-the-fly grid.

    This keeps the original IIRP/RPNet voxel-flow convention: flow channels
    are (dz, dy, dx). Unlike model.iirp.SpatialTransformer, the grid is not
    fixed at construction time, so one DynRPNet can process Brain/ACDC/Liver
    volumes with different compute_size during joint training.
    """

    def __init__(self, mode: str = "bilinear"):
        super().__init__()
        self.mode = mode
        self._cached_shape = None
        self._cached_grid = None

    def _grid(self, shape, device, dtype):
        key = (tuple(int(v) for v in shape), device, dtype)
        if self._cached_shape != key or self._cached_grid is None:
            vectors = [torch.arange(0, int(s), device=device, dtype=dtype) for s in shape]
            grids = torch.meshgrid(vectors, indexing="ij")
            self._cached_grid = torch.stack(grids, dim=0).unsqueeze(0)
            self._cached_shape = key
        return self._cached_grid

    def forward(self, src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        assert src.shape[2:] == flow.shape[2:], \
            f"[DynamicVoxelSpatialTransformer] src spatial={src.shape[2:]}, flow spatial={flow.shape[2:]}"
        grid = self._grid(flow.shape[2:], flow.device, flow.dtype)
        new_locs = grid + flow
        D, H, W = flow.shape[2:]
        new_locs[:, 0] = 2 * (new_locs[:, 0] / max(D - 1, 1) - 0.5)
        new_locs[:, 1] = 2 * (new_locs[:, 1] / max(H - 1, 1) - 0.5)
        new_locs[:, 2] = 2 * (new_locs[:, 2] / max(W - 1, 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]
        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode, padding_mode="border")


TASK_TYPE_TO_ID = {
    "inter": 0,
    "inter-subject": 0,
    "intersubject": 0,
    "intra": 1,
    "intra-subject": 1,
    "intrasubject": 1,
}

TASK_NAME_TO_ID = {
    "chest": 0,
    "chest_unified": 0,
    "abdomen": 1,
    "abdomen_unified": 1,
    "head": 2,
    "head_unified": 2,
    "segrap": 2,
    "liver": 3,
    "liver_unified": 3,
    "cardiac": 4,
    "acdc": 4,
    "brain": 5,
    "oasis": 5,
    "lumir": 5,
}


def get_task_type_id(cfg: dict) -> int:
    if "task_type_id" in cfg and cfg["task_type_id"] is not None:
        return int(cfg["task_type_id"])
    name = str(cfg.get("task_type", "inter-subject")).lower().replace("_", "-")
    if name not in TASK_TYPE_TO_ID:
        raise ValueError(f"Unknown task_type={cfg.get('task_type')}. Please set task_type_id explicitly.")
    return TASK_TYPE_TO_ID[name]


def get_task_id_id(cfg: dict) -> int:
    if "task_id_id" in cfg and cfg["task_id_id"] is not None:
        return int(cfg["task_id_id"])
    name = str(cfg.get("task_id", cfg.get("dataset", ""))).lower()
    if name not in TASK_NAME_TO_ID:
        raise ValueError(f"Unknown task_id={name}. Please set task_id_id explicitly.")
    return TASK_NAME_TO_ID[name]


def _read_runtime_cfg_from_argv():
    """
    Best-effort config reader for the current training script.

    The current unireg model factory only forwards size/first_channel/n_steps/shared_encoder
    to baseline classes. Therefore DynIIRPNetBaseline may not directly receive
    task_type_id/task_id_id from YAML. This helper reads the YAML path from
    `--config` in sys.argv so this file can work without changing train.py/test.py.
    """
    cfg_path = None
    argv = list(sys.argv)
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            cfg_path = argv[i + 1]
            break
        if arg.startswith("--config="):
            cfg_path = arg.split("=", 1)[1]
            break

    out = {}
    if cfg_path:
        try:
            import yaml

            with open(cfg_path, "r") as f:
                loaded = yaml.safe_load(f) or {}
            if isinstance(loaded, dict):
                out.update(loaded)
        except Exception as e:
            print(f"[DynIIRPNet] Warning: failed to read config from {cfg_path}: {e}")

        # Filename fallback, useful when YAML cannot be parsed or task fields are absent.
        name = Path(cfg_path).name.lower()
        if "abdomen" in name:
            out.setdefault("task_id", "abdomen")
            out.setdefault("task_id_id", 1)
            out.setdefault("task_type", "inter-subject")
            out.setdefault("task_type_id", 0)
        elif "chest" in name:
            out.setdefault("task_id", "chest")
            out.setdefault("task_id_id", 0)
            out.setdefault("task_type", "inter-subject")
            out.setdefault("task_type_id", 0)
        elif "head" in name or "segrap" in name:
            out.setdefault("task_id", "head")
            out.setdefault("task_id_id", 2)
            out.setdefault("task_type", "inter-subject")
            out.setdefault("task_type_id", 0)
        elif "liver" in name:
            out.setdefault("task_id", "liver")
            out.setdefault("task_id_id", 3)
            out.setdefault("task_type", "intra-subject")
            out.setdefault("task_type_id", 1)
        elif "acdc" in name or "cardiac" in name:
            out.setdefault("task_id", "cardiac")
            out.setdefault("task_id_id", 4)
            out.setdefault("task_type", "intra-subject")
            out.setdefault("task_type_id", 1)
        elif "oasis" in name or "lumir" in name or "brain" in name:
            out.setdefault("task_id", "brain")
            out.setdefault("task_id_id", 5)
            out.setdefault("task_type", "inter-subject")
            out.setdefault("task_type_id", 0)

    return out


def _merge_runtime_cfg(explicit_cfg):
    runtime_cfg = _read_runtime_cfg_from_argv()
    merged = dict(runtime_cfg)
    for k, v in explicit_cfg.items():
        if v is not None and v != "":
            merged[k] = v
    return merged


def _vox_to_norm(flow_vox: torch.Tensor) -> torch.Tensor:
    """Convert voxel displacement (dz,dy,dx) to normalized grid displacement (dx,dy,dz)."""
    D, H, W = flow_vox.shape[2:]
    flow = flow_vox[:, [2, 1, 0]]
    scale = torch.tensor(
        [2.0 / max(W - 1, 1), 2.0 / max(H - 1, 1), 2.0 / max(D - 1, 1)],
        device=flow_vox.device,
        dtype=flow_vox.dtype,
    ).view(1, 3, 1, 1, 1)
    return flow * scale


def _as_batched_id(
    value: Optional[Union[int, torch.Tensor]],
    default_value: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if value is None:
        out = torch.full((batch_size,), int(default_value), dtype=torch.long, device=device)
    elif torch.is_tensor(value):
        out = value.to(device=device).long().view(-1)
    else:
        out = torch.as_tensor(value, dtype=torch.long, device=device).view(-1)

    if out.numel() == 1 and batch_size > 1:
        out = out.repeat(batch_size)
    if out.numel() != batch_size:
        raise RuntimeError(f"Condition id batch mismatch: got {out.numel()}, expected {batch_size}.")
    return out


class DynamicFlowHead3D(nn.Module):
    """
    Dynamic 1x1x1 flow head conditioned on task_type and task_id.

    feat:      B x C x D x H x W
    cond_feat: B x C_cond x d x h x w
    output:    B x 3 x D x H x W, voxel-space displacement
    """

    def __init__(
        self,
        in_channels: int,
        cond_channels: int,
        task_type_num: int = 2,
        task_id_num: int = 6,
        hidden_channels: int = 8,
        zero_init_controller: bool = False,
        final_std: float = 1e-5,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.cond_channels = int(cond_channels)
        self.task_type_num = int(task_type_num)
        self.task_id_num = int(task_id_num)
        self.hidden_channels = int(hidden_channels)
        self.final_std = float(final_std)

        self.weight_nums = [
            self.hidden_channels * self.in_channels,
            self.hidden_channels * self.hidden_channels,
            3 * self.hidden_channels,
        ]
        self.bias_nums = [self.hidden_channels, self.hidden_channels, 3]
        num_params = sum(self.weight_nums) + sum(self.bias_nums)

        # Choose a valid GroupNorm group count for small channel numbers such as 8.
        gn_groups = 1
        for g in (8, 4, 2, 1):
            if self.cond_channels % g == 0:
                gn_groups = g
                break

        self.gap = nn.Sequential(
            nn.GroupNorm(gn_groups, self.cond_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.controller = nn.Conv3d(
            self.cond_channels + self.task_type_num + self.task_id_num,
            num_params,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # IMPORTANT:
        # Do not initialize the whole controller to zero. If every generated
        # dynamic weight and bias is zero, the 3-layer dynamic head can only
        # update the last output bias at the beginning, so the network is
        # almost locked to an identity/constant-flow solution.
        #
        # We instead initialize the controller as a task-conditioned residual
        # around a small but trainable base head:
        #   - controller.weight = 0: all tasks start from the same safe base head;
        #   - controller.bias encodes non-zero first/second-layer weights and a
        #     very small final layer, following the small-flow initialization
        if zero_init_controller:
            # Kept only for ablation/debugging. Not recommended for training.
            nn.init.zeros_(self.controller.weight)
            if self.controller.bias is not None:
                nn.init.zeros_(self.controller.bias)
        else:
            self._init_controller_as_base_dynamic_head()

    def _init_controller_as_base_dynamic_head(self):
        if self.controller.bias is None:
            raise RuntimeError("DynamicFlowHead3D requires controller bias for stable base-head initialization.")

        nn.init.zeros_(self.controller.weight)

        device = self.controller.bias.device
        dtype = self.controller.bias.dtype
        parts = []

        # Layer 1: hidden x in_channels
        w1 = torch.empty(self.hidden_channels, self.in_channels, 1, 1, 1, device=device, dtype=dtype)
        nn.init.kaiming_uniform_(w1, a=0.2)
        b1 = torch.zeros(self.hidden_channels, device=device, dtype=dtype)

        # Layer 2: hidden x hidden
        w2 = torch.empty(self.hidden_channels, self.hidden_channels, 1, 1, 1, device=device, dtype=dtype)
        nn.init.kaiming_uniform_(w2, a=0.2)
        b2 = torch.zeros(self.hidden_channels, device=device, dtype=dtype)

        # Layer 3: 3 x hidden. Small non-zero init keeps initial deformation tiny
        # while allowing gradients to propagate through all generated layers.
        w3 = torch.empty(3, self.hidden_channels, 1, 1, 1, device=device, dtype=dtype)
        nn.init.normal_(w3, mean=0.0, std=self.final_std)
        b3 = torch.zeros(3, device=device, dtype=dtype)

        with torch.no_grad():
            base = torch.cat([
                w1.flatten(), w2.flatten(), w3.flatten(),
                b1.flatten(), b2.flatten(), b3.flatten(),
            ], dim=0)
            if base.numel() != self.controller.bias.numel():
                raise RuntimeError(
                    f"Base dynamic-head parameter count mismatch: {base.numel()} vs {self.controller.bias.numel()}"
                )
            self.controller.bias.copy_(base)

    @staticmethod
    def _one_hot(ids: torch.Tensor, num_classes: int) -> torch.Tensor:
        ids = ids.long().view(-1)
        out = torch.zeros(ids.numel(), num_classes, device=ids.device, dtype=torch.float32)
        out.scatter_(1, ids[:, None], 1.0)
        return out[:, :, None, None, None]

    def parse_dynamic_params(self, params: torch.Tensor):
        B = params.shape[0]
        n_layers = len(self.weight_nums)
        splits = torch.split(params, self.weight_nums + self.bias_nums, dim=1)
        weight_splits = list(splits[:n_layers])
        bias_splits = list(splits[n_layers:])

        weight_splits[0] = weight_splits[0].reshape(B * self.hidden_channels, self.in_channels, 1, 1, 1)
        bias_splits[0] = bias_splits[0].reshape(B * self.hidden_channels)

        weight_splits[1] = weight_splits[1].reshape(B * self.hidden_channels, self.hidden_channels, 1, 1, 1)
        bias_splits[1] = bias_splits[1].reshape(B * self.hidden_channels)

        weight_splits[2] = weight_splits[2].reshape(B * 3, self.hidden_channels, 1, 1, 1)
        bias_splits[2] = bias_splits[2].reshape(B * 3)
        return weight_splits, bias_splits

    @staticmethod
    def heads_forward(x: torch.Tensor, weights, biases, batch_size: int) -> torch.Tensor:
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv3d(x, w, bias=b, stride=1, padding=0, groups=batch_size)
            if i < len(weights) - 1:
                x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        return x

    def forward(
        self,
        feat: torch.Tensor,
        cond_feat: torch.Tensor,
        task_type: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        B, C, D, H, W = feat.shape
        if C != self.in_channels:
            raise RuntimeError(f"DynamicFlowHead3D expected feat C={self.in_channels}, got {C}.")

        type_one_hot = self._one_hot(task_type, self.task_type_num)
        id_one_hot = self._one_hot(task_id, self.task_id_num)
        cond = self.gap(cond_feat)
        cond = torch.cat([cond, type_one_hot, id_one_hot], dim=1)

        params = self.controller(cond).flatten(1)
        weights, biases = self.parse_dynamic_params(params)

        feat_grouped = feat.reshape(1, B * C, D, H, W)
        flow = self.heads_forward(feat_grouped, weights, biases, B)
        return flow.reshape(B, 3, D, H, W)


class DynamicDecoderBlock(nn.Module):
    """
    IIRP decoder block with a task-conditioned dynamic flow head.

    It keeps the original Conv1/Conv2/Conv3/Conv4 feature extractor and replaces
    the fixed Conv5 flow layer with DynamicFlowHead3D.
    """

    def __init__(
        self,
        x_channel: int,
        y_channel: int,
        out_channel: int,
        norm=None,
        task_type_num: int = 2,
        task_id_num: int = 6,
        dyn_hidden_channels: int = 8,
    ):
        super().__init__()
        self.Conv1 = ConvBlock(x_channel + y_channel, out_channel, norm=norm)
        self.Conv2 = ConvResBlock(out_channel, out_channel, norm=norm)
        self.Conv3 = ConvResBlock(out_channel, out_channel, norm=norm)
        self.Conv4 = nn.Conv3d(out_channel, out_channel // 2, 3, padding=1)
        self.dynamic_flow = DynamicFlowHead3D(
            in_channels=out_channel // 2,
            cond_channels=out_channel // 2,
            task_type_num=task_type_num,
            task_id_num=task_id_num,
            hidden_channels=dyn_hidden_channels,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, task_type: torch.Tensor, task_id: torch.Tensor):
        try:
            concat = torch.cat([x, y], dim=1)
        except RuntimeError as e:
            print(f"[ERROR] DynamicDecoderBlock concat failed: x={x.shape}, y={y.shape}.")
            raise e

        feat = self.Conv1(concat)
        feat = self.Conv2(feat)
        feat = self.Conv3(feat)
        feat = self.Conv4(feat)
        flow = self.dynamic_flow(feat=feat, cond_feat=feat, task_type=task_type, task_id=task_id)
        return flow


class DynIIRPNet(LoadableModel):
    """
    Dynamic IIRPNet.

    Compared with the original IIRPNet, every pyramid decoder predicts flow through
    a dynamic head conditioned on:
      - task_type: inter-subject / intra-subject
      - task_id: chest / abdomen / head / liver / cardiac / brain

    Raw forward returns:
        warped_x, flow_vox
    where flow_vox follows the original IIRP voxel-space convention.
    """

    @store_config_args
    def __init__(
        self,
        size,
        in_channel=1,
        first_channel=8,
        shared_encoder=True,
        norm=None,
        task_type_num=2,
        task_id_num=6,
        default_task_type=0,
        default_task_id=0,
        dyn_hidden_channels=8,
    ):
        super().__init__()
        self.shared_encoder = shared_encoder
        self.size = tuple(size)
        self.task_type_num = int(task_type_num)
        self.task_id_num = int(task_id_num)
        self.default_task_type = int(default_task_type)
        self.default_task_id = int(default_task_id)

        c = first_channel
        self.encoder = Encoder(in_channel, c, norm=norm)
        if not self.shared_encoder:
            print("🚀 [DynIIRPNet] Using Independent Encoders for Moving/Fixed images")
            self.encoder_fixed = Encoder(in_channel, c, norm=norm)
        else:
            self.encoder_fixed = None

        common = dict(
            norm=norm,
            task_type_num=self.task_type_num,
            task_id_num=self.task_id_num,
            dyn_hidden_channels=int(dyn_hidden_channels),
        )
        # self.decoder4 = DynamicDecoderBlock(x_channel=32, y_channel=32, out_channel=32, **common)
        # self.decoder3 = DynamicDecoderBlock(x_channel=32, y_channel=32, out_channel=32, **common)
        # self.decoder2 = DynamicDecoderBlock(x_channel=16, y_channel=16, out_channel=32, **common)
        # self.decoder1 = DynamicDecoderBlock(x_channel=8, y_channel=8, out_channel=16, **common)
        
        self.decoder4 = DynamicDecoderBlock(x_channel=4*c, y_channel=4*c, out_channel=4*c, **common)
        self.decoder3 = DynamicDecoderBlock(x_channel=4*c, y_channel=4*c, out_channel=4*c, **common)
        self.decoder2 = DynamicDecoderBlock(x_channel=2*c, y_channel=2*c, out_channel=4*c, **common)
        self.decoder1 = DynamicDecoderBlock(x_channel=c, y_channel=c, out_channel=2*c, **common)
        

        self.transformer = nn.ModuleList([DynamicVoxelSpatialTransformer() for _ in range(4)])
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

    def normalized_cross_correlation(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        mean_img1 = torch.mean(img1)
        mean_img2 = torch.mean(img2)
        std_img1 = torch.std(img1).clamp_min(1e-6)
        std_img2 = torch.std(img2).clamp_min(1e-6)
        return torch.mean((img1 - mean_img1) * (img2 - mean_img2) / (std_img1 * std_img2))

    def _resolve_conditions(self, x: torch.Tensor, task_type=None, task_id=None):
        B = x.shape[0]
        device = x.device
        task_type = _as_batched_id(task_type, self.default_task_type, B, device)
        task_id = _as_batched_id(task_id, self.default_task_id, B, device)
        return task_type, task_id

    def forward(self, x: torch.Tensor, y: torch.Tensor, task_type=None, task_id=None):
        task_type, task_id = self._resolve_conditions(x, task_type, task_id)

        fx1, fx2, fx3, fx4 = self.encoder(x)
        if self.shared_encoder:
            fy1, fy2, fy3, fy4 = self.encoder(y)
        else:
            fy1, fy2, fy3, fy4 = self.encoder_fixed(y)

        ar = br = cr = dr = 10
        pa = pb = pc = pd = 0
        delta1 = 0.005

        wx4 = fx4
        ncc_a = 0
        flowall = None

        for aa in range(ar):
            flow = self.decoder4(wx4, fy4, task_type, task_id)
            previous_flow = flowall
            if aa == 0:
                flowall = flow
            else:
                flowall = self.transformer[3](flowall, flow) + flow
            wx4 = self.transformer[3](fx4, flowall)
            flowx4 = nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True)(8 * flowall)
            mx4 = self.transformer[0](x, flowx4)
            ncc = self.normalized_cross_correlation(mx4, y)
            if ncc < (ncc_a + delta1) and previous_flow is not None:
                flowall = previous_flow
                break
            ncc_a = ncc
            pa += 1

        flowall = self.up(2 * flowall)
        ncc_a = 0
        for bb in range(br):
            previous_flow = flowall
            wx3 = self.transformer[2](fx3, flowall)
            flowx3 = nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True)(4 * flowall)
            mx3 = self.transformer[0](x, flowx3)
            ncc = self.normalized_cross_correlation(mx3, y)
            if ncc < (ncc_a + delta1) and previous_flow is not None:
                flowall = previous_flow
                break
            ncc_a = ncc
            pb += 1
            flow = self.decoder3(wx3, fy3, task_type, task_id)
            flowall = self.transformer[2](flowall, flow) + flow

        flowall = self.up(2 * flowall)
        ncc_a = 0
        for cc in range(cr):
            previous_flow = flowall
            wx2 = self.transformer[1](fx2, flowall)
            flowx2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)(2 * flowall)
            mx2 = self.transformer[0](x, flowx2)
            ncc = self.normalized_cross_correlation(mx2, y)
            if ncc < (ncc_a + delta1) and previous_flow is not None:
                flowall = previous_flow
                break
            ncc_a = ncc
            pc += 1
            flow = self.decoder2(wx2, fy2, task_type, task_id)
            flowall = self.transformer[1](flowall, flow) + flow

        flowall = self.up(2 * flowall)
        ncc_a = 0
        for dd in range(dr):
            previous_flow = flowall
            wx1 = self.transformer[0](fx1, flowall)
            mx = self.transformer[0](x, flowall)
            ncc = self.normalized_cross_correlation(mx, y)
            if ncc < (ncc_a + delta1) and previous_flow is not None:
                flowall = previous_flow
                break
            ncc_a = ncc
            pd += 1
            flow = self.decoder1(wx1, fy1, task_type, task_id)
            flowall = self.transformer[0](flowall, flow) + flow

        warped_x = self.transformer[0](x, flowall)
        return warped_x, flowall


class DynIIRPNetBaseline(nn.Module):
    """
    Wrapper compatible with current unireg baseline registry.

    forward(moving, fixed) -> (warped, flow_norm)
    """

    def __init__(self, compute_size, first_channel=8, shared_encoder=True, **cfg):
        super().__init__()
        size = tuple(compute_size)

        # The outer unireg model factory may not pass task fields to this baseline.
        # Merge YAML config from --config with explicitly forwarded kwargs.
        cfg = _merge_runtime_cfg(cfg)

        # Do NOT use cfg.get(key, get_xxx(cfg)) here: Python evaluates the default
        # argument eagerly, which may call get_task_id_id(cfg) even when task_id_id
        # is already present in cfg.
        if "default_task_type" in cfg:
            self.default_task_type = int(cfg["default_task_type"])
        elif "task_type_id" in cfg:
            self.default_task_type = int(cfg["task_type_id"])
        else:
            self.default_task_type = int(get_task_type_id(cfg))

        if "default_task_id" in cfg:
            self.default_task_id = int(cfg["default_task_id"])
        elif "task_id_id" in cfg:
            self.default_task_id = int(cfg["task_id_id"])
        else:
            self.default_task_id = int(get_task_id_id(cfg))

        print(
            f"[DynIIRPNet] task_type_id={self.default_task_type}, "
            f"task_id_id={self.default_task_id}, "
            f"task_id={cfg.get('task_id', cfg.get('dataset', ''))}"
        )

        self.net = DynIIRPNet(
            size=size,
            in_channel=1,
            first_channel=first_channel,
            shared_encoder=shared_encoder,
            norm=cfg.get("norm", None),
            task_type_num=int(cfg.get("task_type_num", 2)),
            task_id_num=int(cfg.get("task_id_num", 6)),
            default_task_type=self.default_task_type,
            default_task_id=self.default_task_id,
            dyn_hidden_channels=int(cfg.get("dyn_hidden_channels", 8)),
        )

        if NormalizedSpatialTransformer is None:
            raise ImportError("Cannot import unireg.core.transforms.SpatialTransformer.")
        self._stn = NormalizedSpatialTransformer()

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor, task_type=None, task_id=None):
        _, flow_vox = self.net(moving, fixed, task_type=task_type, task_id=task_id)
        flow_norm = _vox_to_norm(flow_vox)
        warped = self._stn(moving, flow_norm)
        return warped, flow_norm


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic RPNet: RPNet-style one-step coarse-to-fine refinement with the same
# task-conditioned DynamicDecoderBlock used by DynIIRPNet.
# Raw network returns voxel-space flow; Baseline wrapper returns normalized flow.
# ─────────────────────────────────────────────────────────────────────────────

class DynRPNet(LoadableModel):
    @store_config_args
    def __init__(
        self,
        size,
        in_channel=1,
        first_channel=8,
        shared_encoder=True,
        norm=None,
        task_type_num=2,
        task_id_num=6,
        default_task_type=0,
        default_task_id=0,
        dyn_hidden_channels=8,
    ):
        super(DynRPNet, self).__init__()
        self.size = tuple(size)
        self.shared_encoder = bool(shared_encoder)
        self.task_type_num = int(task_type_num)
        self.task_id_num = int(task_id_num)
        self.default_task_type = int(default_task_type)
        self.default_task_id = int(default_task_id)

        c = first_channel
        self.encoder = Encoder(in_channel, c, norm=norm)
        if not self.shared_encoder:
            print("🚀 [DynRPNet] Using Independent Encoders for Moving/Fixed images")
            self.encoder_fixed = Encoder(in_channel, c, norm=norm)
        else:
            self.encoder_fixed = None

        common = dict(
            norm=norm,
            task_type_num=self.task_type_num,
            task_id_num=self.task_id_num,
            dyn_hidden_channels=int(dyn_hidden_channels),
        )
        # self.decoder4 = DynamicDecoderBlock(x_channel=32, y_channel=32, out_channel=32, **common)
        # self.decoder3 = DynamicDecoderBlock(x_channel=32, y_channel=32, out_channel=32, **common)
        # self.decoder2 = DynamicDecoderBlock(x_channel=16, y_channel=16, out_channel=32, **common)
        # self.decoder1 = DynamicDecoderBlock(x_channel=8, y_channel=8, out_channel=16, **common)
        
        self.decoder4 = DynamicDecoderBlock(x_channel=4*c, y_channel=4*c, out_channel=4*c, **common)
        self.decoder3 = DynamicDecoderBlock(x_channel=4*c, y_channel=4*c, out_channel=4*c, **common)
        self.decoder2 = DynamicDecoderBlock(x_channel=2*c, y_channel=2*c, out_channel=4*c, **common)
        self.decoder1 = DynamicDecoderBlock(x_channel=c, y_channel=c, out_channel=2*c, **common)
        


        self.transformer = nn.ModuleList([DynamicVoxelSpatialTransformer() for _ in range(4)])
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

    def _resolve_conditions(self, x: torch.Tensor, task_type=None, task_id=None):
        B = x.shape[0]
        device = x.device
        task_type = _as_batched_id(task_type, self.default_task_type, B, device)
        task_id = _as_batched_id(task_id, self.default_task_id, B, device)
        return task_type, task_id

    def _warp_feat(self, feat, flow, level):
        return self.transformer[level](feat, flow)

    def forward(self, x: torch.Tensor, y: torch.Tensor, task_type=None, task_id=None):
        task_type, task_id = self._resolve_conditions(x, task_type, task_id)

        fx1, fx2, fx3, fx4 = self.encoder(x)
        if self.shared_encoder:
            fy1, fy2, fy3, fy4 = self.encoder(y)
        else:
            assert self.encoder_fixed is not None
            fy1, fy2, fy3, fy4 = self.encoder_fixed(y)

        # RPNet logic: one update at each pyramid level, no NCC-based iterative loop.
        wx4 = fx4
        flowall = self.decoder4(wx4, fy4, task_type, task_id)

        flowall = self.up(2 * flowall)
        wx3 = self.transformer[2](fx3, flowall)
        flow = self.decoder3(wx3, fy3, task_type, task_id)
        flowall = self.transformer[2](flowall, flow) + flow

        flowall = self.up(2 * flowall)
        wx2 = self.transformer[1](fx2, flowall)
        flow = self.decoder2(wx2, fy2, task_type, task_id)
        flowall = self.transformer[1](flowall, flow) + flow

        flowall = self.up(2 * flowall)
        wx1 = self.transformer[0](fx1, flowall)
        flow = self.decoder1(wx1, fy1, task_type, task_id)
        flowall = self.transformer[0](flowall, flow) + flow

        warped_x = self.transformer[0](x, flowall)
        return warped_x, flowall


class DynRPNetBaseline(nn.Module):
    """
    Wrapper compatible with current unireg baseline registry.

    forward(moving, fixed) -> (warped, flow_norm)
    """

    def __init__(self, compute_size, first_channel=8, shared_encoder=True, **cfg):
        super().__init__()
        size = tuple(compute_size)
        cfg = _merge_runtime_cfg(cfg)

        if "default_task_type" in cfg:
            self.default_task_type = int(cfg["default_task_type"])
        elif "task_type_id" in cfg:
            self.default_task_type = int(cfg["task_type_id"])
        else:
            self.default_task_type = int(get_task_type_id(cfg))

        if "default_task_id" in cfg:
            self.default_task_id = int(cfg["default_task_id"])
        elif "task_id_id" in cfg:
            self.default_task_id = int(cfg["task_id_id"])
        else:
            self.default_task_id = int(get_task_id_id(cfg))

        print(
            f"[DynRPNet] task_type_id={self.default_task_type}, "
            f"task_id_id={self.default_task_id}, "
            f"task_id={cfg.get('task_id', cfg.get('dataset', ''))}"
        )

        self.net = DynRPNet(
            size=size,
            in_channel=1,
            first_channel=first_channel,
            shared_encoder=shared_encoder,
            norm=cfg.get("norm", None),
            task_type_num=int(cfg.get("task_type_num", 2)),
            task_id_num=int(cfg.get("task_id_num", 6)),
            default_task_type=self.default_task_type,
            default_task_id=self.default_task_id,
            dyn_hidden_channels=int(cfg.get("dyn_hidden_channels", 8)),
        )

        if NormalizedSpatialTransformer is None:
            raise ImportError("Cannot import unireg.core.transforms.SpatialTransformer.")
        self._stn = NormalizedSpatialTransformer()

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor, task_type=None, task_id=None):
        _, flow_vox = self.net(moving, fixed, task_type=task_type, task_id=task_id)
        flow_norm = _vox_to_norm(flow_vox)
        warped = self._stn(moving, flow_norm)
        return warped, flow_norm


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic CorrMLP: keep the original CorrMLP encoder/decoder structure and
# replace the four static RegHead_block modules with task-conditioned
# DynamicFlowHead3D modules.
#
# As long as each dimension is compatible with the 3-level pooling/upsampling
# path, one DynCorrMLP can be used with different task compute_size values.
# ─────────────────────────────────────────────────────────────────────────────

class DynCorrMLPDecoder(nn.Module):
    """CorrMLP decoder with task-conditioned dynamic flow heads."""

    def __init__(
        self,
        in_channels: int = 8,
        channel_num: int = 16,
        use_checkpoint: bool = True,
        task_type_num: int = 2,
        task_id_num: int = 6,
        dyn_hidden_channels: int = 8,
    ):
        super().__init__()
        from model.CorrMLP import (  # type: ignore
            CMWMLP_block,
            PatchExpanding_block,
            ResizeTransformer_block,
            SpatialTransformer_block,
        )

        self.mlp_11 = CMWMLP_block(in_channels, channel_num, use_corr=True, use_checkpoint=use_checkpoint)
        self.mlp_12 = CMWMLP_block(in_channels * 2, channel_num * 2, use_corr=True, use_checkpoint=use_checkpoint)
        self.mlp_13 = CMWMLP_block(in_channels * 4, channel_num * 4, use_corr=True, use_checkpoint=use_checkpoint)
        self.mlp_14 = CMWMLP_block(in_channels * 8, channel_num * 8, use_corr=True, use_checkpoint=use_checkpoint)

        self.mlp_21 = CMWMLP_block(channel_num, channel_num, use_corr=True, use_checkpoint=use_checkpoint)
        self.mlp_22 = CMWMLP_block(channel_num * 2, channel_num * 2, use_corr=True, use_checkpoint=use_checkpoint)
        self.mlp_23 = CMWMLP_block(channel_num * 4, channel_num * 4, use_corr=True, use_checkpoint=use_checkpoint)

        self.upsample_1 = PatchExpanding_block(embed_dim=channel_num * 2)
        self.upsample_2 = PatchExpanding_block(embed_dim=channel_num * 4)
        self.upsample_3 = PatchExpanding_block(embed_dim=channel_num * 8)

        self.ResizeTransformer = ResizeTransformer_block(resize_factor=2, mode="trilinear")
        self.SpatialTransformer = SpatialTransformer_block(mode="bilinear")

        # Original CorrMLP has static RegHead_block at four pyramid levels.
        # Here each static head is replaced by a task-conditioned dynamic head.
        common = dict(
            task_type_num=int(task_type_num),
            task_id_num=int(task_id_num),
            hidden_channels=int(dyn_hidden_channels),
        )
        self.reghead_4 = DynamicFlowHead3D(
            in_channels=channel_num * 8,
            cond_channels=channel_num * 8,
            **common,
        )
        self.reghead_3 = DynamicFlowHead3D(
            in_channels=channel_num * 4,
            cond_channels=channel_num * 4,
            **common,
        )
        self.reghead_2 = DynamicFlowHead3D(
            in_channels=channel_num * 2,
            cond_channels=channel_num * 2,
            **common,
        )
        self.reghead_1 = DynamicFlowHead3D(
            in_channels=channel_num,
            cond_channels=channel_num,
            **common,
        )

    def forward(self, x_fix, x_mov, task_type: torch.Tensor, task_id: torch.Tensor):
        x_fix_1, x_fix_2, x_fix_3, x_fix_4 = x_fix
        x_mov_1, x_mov_2, x_mov_3, x_mov_4 = x_mov

        # Step 1: coarsest level.
        x_4 = self.mlp_14(x_fix_4, x_mov_4)
        flow_4 = self.reghead_4(x_4, x_4, task_type, task_id)

        # Step 2.
        flow_4_up = self.ResizeTransformer(flow_4)
        x_mov_3 = self.SpatialTransformer(x_mov_3, flow_4_up)

        x = self.mlp_13(x_fix_3, x_mov_3)
        x_3 = self.mlp_23(x, self.upsample_3(x_4))

        delta_3 = self.reghead_3(x_3, x_3, task_type, task_id)
        flow_3 = delta_3 + flow_4_up

        # Step 3.
        flow_3_up = self.ResizeTransformer(flow_3)
        x_mov_2 = self.SpatialTransformer(x_mov_2, flow_3_up)

        x = self.mlp_12(x_fix_2, x_mov_2)
        x_2 = self.mlp_22(x, self.upsample_2(x_3))

        delta_2 = self.reghead_2(x_2, x_2, task_type, task_id)
        flow_2 = delta_2 + flow_3_up

        # Step 4: full resolution.
        flow_2_up = self.ResizeTransformer(flow_2)
        x_mov_1 = self.SpatialTransformer(x_mov_1, flow_2_up)

        x = self.mlp_11(x_fix_1, x_mov_1)
        x_1 = self.mlp_21(x, self.upsample_1(x_2))

        delta_1 = self.reghead_1(x_1, x_1, task_type, task_id)
        flow_1 = delta_1 + flow_2_up

        return flow_1


class DynCorrMLPBaseline(nn.Module):
    """
    Dynamic CorrMLP wrapper compatible with the current unireg baseline registry.

    Interface:
        forward(moving, fixed, task_type=None, task_id=None) -> (warped, flow_norm)

    Notes:
      - The CorrMLP encoder/decoder is fully convolutional and can process
        different compute_size values across tasks, provided the dimensions are
        compatible with the 3-level down/up-sampling path.
      - The raw CorrMLP flow is voxel-space (dz, dy, dx), so it is converted to
        normalized-grid flow before using unireg.core.transforms.SpatialTransformer.
    """

    def __init__(self, compute_size, shared_encoder=True, **cfg):
        super().__init__()
        from model.CorrMLP import Conv_encoder  # type: ignore

        cfg = _merge_runtime_cfg(cfg)

        if "default_task_type" in cfg:
            self.default_task_type = int(cfg["default_task_type"])
        elif "task_type_id" in cfg:
            self.default_task_type = int(cfg["task_type_id"])
        else:
            self.default_task_type = int(get_task_type_id(cfg))

        if "default_task_id" in cfg:
            self.default_task_id = int(cfg["default_task_id"])
        elif "task_id_id" in cfg:
            self.default_task_id = int(cfg["task_id_id"])
        else:
            self.default_task_id = int(get_task_id_id(cfg))

        self.enc_channels = int(cfg.get("corr_enc_channels", cfg.get("enc_channels", 8)))
        self.dec_channels = int(cfg.get("corr_dec_channels", cfg.get("dec_channels", 16)))
        self.use_checkpoint = bool(cfg.get("corr_use_checkpoint", cfg.get("use_checkpoint", True)))

        print(
            f"[DynCorrMLP] task_type_id={self.default_task_type}, "
            f"task_id_id={self.default_task_id}, "
            f"task_id={cfg.get('task_id', cfg.get('dataset', ''))}"
        )

        # Keep CorrMLP module names close to the original implementation to make
        # partial checkpoint loading from CorrMLP as straightforward as possible.
        self.Encoder = Conv_encoder(
            in_channels=1,
            channel_num=self.enc_channels,
            use_checkpoint=self.use_checkpoint,
        )
        self.Decoder = DynCorrMLPDecoder(
            in_channels=self.enc_channels,
            channel_num=self.dec_channels,
            use_checkpoint=self.use_checkpoint,
            task_type_num=int(cfg.get("task_type_num", 2)),
            task_id_num=int(cfg.get("task_id_num", 6)),
            dyn_hidden_channels=int(cfg.get("dyn_hidden_channels", 8)),
        )

        if NormalizedSpatialTransformer is None:
            raise ImportError("Cannot import unireg.core.transforms.SpatialTransformer.")
        self._stn = NormalizedSpatialTransformer()

    def _resolve_conditions(self, x: torch.Tensor, task_type=None, task_id=None):
        B = x.shape[0]
        device = x.device
        task_type = _as_batched_id(task_type, self.default_task_type, B, device)
        task_id = _as_batched_id(task_id, self.default_task_id, B, device)
        return task_type, task_id

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor, task_type=None, task_id=None):
        task_type, task_id = self._resolve_conditions(moving, task_type, task_id)

        x_fix = self.Encoder(fixed)
        x_mov = self.Encoder(moving)
        flow_vox = self.Decoder(x_fix, x_mov, task_type, task_id)

        flow_norm = _vox_to_norm(flow_vox)
        warped = self._stn(moving, flow_norm)
        return warped, flow_norm
