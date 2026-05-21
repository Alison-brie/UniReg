"""
unireg/engine/trainer.py
======================
Unified Trainer for training + validation in one place.

Design:
  - P3: training and validation are performed in ONE loop (no separate scripts)
  - validate() maps compute-space predictions back to seg space for Dice
  - Configurable val_every, save_every
  - AMP (Automatic Mixed Precision) optional
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List
from collections import defaultdict

from unireg.core.metrics import dice_per_label, compute_foldings, compute_tre_mm
from unireg.core.transforms import SpatialTransformer
from unireg.engine.checkpoint import save_checkpoint, load_checkpoint
from unireg.engine.logger import MetricLogger
from unireg.losses.feature import FeatureLoss
from unireg.losses.similarity import MaskedNCC3D


class Trainer:
    """
    Unified training + validation runner.

    Args:
        model        : registration network (own model or baseline)
        optimizer    : torch optimizer
        sim_loss     : similarity loss function(warped, fixed) -> scalar
        reg_loss     : regularisation loss(flow) -> scalar
        sim_weight   : weight for similarity loss (default 1.0)
        reg_weight   : weight for regularisation loss (default 0.01)
        device       : 'cuda' | 'cpu'
        log_dir      : directory for logs and checkpoints
        amp          : use torch AMP (default True if CUDA available)
        val_every    : run validation every N steps (default 500)
        save_every   : save checkpoint every N steps (default 500)
        max_steps    : total training steps (default 10000)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        sim_loss: nn.Module,
        reg_loss: nn.Module,
        sim_weight: float = 1.0,
        reg_weight: float  = 0.01,
        task_sim_losses: Optional[Dict[str, nn.Module]] = None,
        task_reg_losses: Optional[Dict[str, nn.Module]] = None,
        task_sim_weights: Optional[Dict[str, float]] = None,
        task_reg_weights: Optional[Dict[str, float]] = None,
        device: str = "cuda",
        log_dir: str = "./logs",
        amp: bool = True,
        val_every: int = 500,
        save_every: int = 500,
        max_steps: int = 10_000,
        lr_scheduler: Optional[Any] = None,
        # ---- optional feature loss ----
        use_feature_loss: bool = False,
        feat_weight: float = 0.0,
        feat_loss_type: str = "cos",
        feat_ncc_win: int = 9,
        feat_level_weights: Optional[List[float]] = None,
        # ---- model selection metric ----
        select_metric: str = "dice",  # "dice" | "tre"
    ):
        self.model        = model.to(device)
        self.optimizer    = optimizer
        self.sim_loss     = sim_loss.to(device) if hasattr(sim_loss, "to") else sim_loss
        self.reg_loss     = reg_loss
        self.sim_weight   = sim_weight
        self.reg_weight   = reg_weight
        self.task_sim_losses = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in (task_sim_losses or {}).items()}
        self.task_reg_losses = dict(task_reg_losses or {})
        self.task_sim_weights = dict(task_sim_weights or {})
        self.task_reg_weights = dict(task_reg_weights or {})
        # MaskedNCC is enabled only for batches with pre_align and fixed_mask.
        # It mirrors each task's NCC window when available. Other tasks are unaffected.
        self.masked_sim_loss = MaskedNCC3D(win_size=getattr(self.sim_loss, "win_size", 9)).to(device)
        self.task_masked_sim_losses = {
            k: MaskedNCC3D(win_size=getattr(v, "win_size", 9)).to(device)
            for k, v in self.task_sim_losses.items()
        }
        self.device       = device
        self.log_dir      = log_dir
        self.val_every    = val_every
        self.save_every   = save_every
        self.max_steps    = max_steps
        self.lr_scheduler = lr_scheduler

        self.use_feature_loss = bool(use_feature_loss)
        self.feat_weight = float(feat_weight)
        self.feat_level_weights = feat_level_weights or [1.0, 1.0, 1.0, 1.0]
        feat_loss_key = str(feat_loss_type).lower().strip()
        if feat_loss_key == "mse":
            feat_loss_key = "l2"
        self.feat_loss = FeatureLoss(loss_type=feat_loss_key, ncc_win=int(feat_ncc_win))

        self.select_metric = str(select_metric).lower().strip() or "dice"

        self.amp           = amp and device.startswith("cuda") and torch.cuda.is_available()
        self.scaler        = torch.cuda.amp.GradScaler() if self.amp else None

        os.makedirs(log_dir, exist_ok=True)
        self.logger  = MetricLogger(log_dir, "train")
        self.val_logger = MetricLogger(log_dir, "val")
        self.stn_nn  = SpatialTransformer(mode="nearest")

        self._step   = 0
        self._epoch  = 0
        self._best_dice = -1.0
        self._best_tre  = float("inf")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def run(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        self.model.train()
        step_since_log = 0

        while self._step < self.max_steps:
            self._epoch += 1
            for batch in train_loader:
                if self._step >= self.max_steps:
                    break

                loss_dict = self._train_step(batch)
                self._step += 1

                # Log every 100 steps (~1 epoch when steps_per_epoch=100)
                if self._step % 100 == 0:
                    self.logger.log(self._step, loss_dict)

                if val_loader and self._step % self.val_every == 0:
                    val_metrics = self.validate(val_loader)
                    self.val_logger.log(self._step, val_metrics, phase="val")

                    if self.select_metric == "tre":
                        tre_mean = val_metrics.get("tre_mean", float("inf"))
                        if tre_mean < self._best_tre:
                            self._best_tre = tre_mean
                            self._save("best.pth")
                    else:
                        dice_mean = val_metrics.get("dice_mean", -1.0)
                        if dice_mean > self._best_dice:
                            self._best_dice = dice_mean
                            self._save("best.pth")

                if self._step % self.save_every == 0:
                    self._save(f"step_{self._step:06d}.pth")

        self._save("last.pth")
        self.logger.close()
        self.val_logger.close()
        if self.select_metric == "tre":
            print(f"[Trainer] Training complete. Best val TRE = {self._best_tre:.4f} mm", flush=True)
        else:
            print(f"[Trainer] Training complete. Best val Dice = {self._best_dice:.4f}", flush=True)

    # ------------------------------------------------------------------
    # Task-aware helpers
    # ------------------------------------------------------------------

    def _batch_task_name(self, batch: Dict) -> Optional[str]:
        task_name = batch.get("task_name", None)
        if task_name is None:
            return None
        if isinstance(task_name, (list, tuple)):
            task_name = task_name[0]
        return str(task_name)

    def _get_task_loss_cfg(self, batch: Dict):
        task_name = self._batch_task_name(batch)
        if task_name is None:
            return self.sim_loss, self.reg_loss, self.sim_weight, self.reg_weight
        return (
            self.task_sim_losses.get(task_name, self.sim_loss),
            self.task_reg_losses.get(task_name, self.reg_loss),
            float(self.task_sim_weights.get(task_name, self.sim_weight)),
            float(self.task_reg_weights.get(task_name, self.reg_weight)),
        )

    def _get_masked_sim_loss(self, batch: Dict):
        task_name = self._batch_task_name(batch)
        if task_name is None:
            return self.masked_sim_loss
        return self.task_masked_sim_losses.get(task_name, self.masked_sim_loss)

    def _build_valid_similarity_mask(
        self,
        flow: torch.Tensor,
        fixed_mask: Optional[torch.Tensor],
        moving_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Build fixed-mask / valid-overlap mask for MaskedNCC.

        valid_mask = fixed_mask * warp(moving_mask, flow) when moving_mask exists;
        otherwise valid_mask = fixed_mask. The mask is thresholded to binary and
        detached so the loss does not backprop through the mask warp.
        """
        if fixed_mask is None:
            return None

        valid_mask = (fixed_mask > 0.5).to(device=flow.device, dtype=flow.dtype)
        if moving_mask is not None:
            stn_nn = self.stn_nn.to(self.device)
            warped_moving_mask = stn_nn(
                moving_mask.to(device=flow.device, dtype=flow.dtype),
                flow,
            )
            warped_moving_mask = (warped_moving_mask > 0.5).to(dtype=flow.dtype)
            valid_mask = valid_mask * warped_moving_mask

        return valid_mask.detach()

    def _get_task_ids(self, batch: Dict):
        task_type = batch.get("task_type_id", None)
        task_id = batch.get("task_id_id", None)
        if task_type is not None:
            task_type = task_type.to(self.device).view(-1).long()
        if task_id is not None:
            task_id = task_id.to(self.device).view(-1).long()
        return task_type, task_id

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    def _train_step(self, batch: Dict) -> Dict[str, float]:
        moving = batch["moving"].to(self.device, dtype=torch.float32)
        fixed  = batch["fixed"].to(self.device, dtype=torch.float32)
        task_type, task_id = self._get_task_ids(batch)
        sim_loss_fn, reg_loss_fn, sim_weight, reg_weight = self._get_task_loss_cfg(batch)

        # Optional SAMCoarse pre-alignment field.
        # UnifiedPairDataset should return batch["pre_align"] when available.
        pre_align = batch.get("pre_align", None)
        if pre_align is not None:
            pre_align = pre_align.to(self.device, dtype=torch.float32)

        moving_mask = batch.get("moving_mask", None)
        fixed_mask = batch.get("fixed_mask", None)
        if moving_mask is not None:
            moving_mask = moving_mask.to(self.device, dtype=torch.float32)
        if fixed_mask is not None:
            fixed_mask = fixed_mask.to(self.device, dtype=torch.float32)

        self.optimizer.zero_grad()

        want_feat = (
            self.use_feature_loss
            and self.feat_weight > 0
            and self._model_supports_features()
        )

        if self.amp:
            with torch.cuda.amp.autocast():
                out = self._forward_model(
                    moving,
                    fixed,
                    pre_align=pre_align,
                    moving_mask=moving_mask,
                    fixed_mask=fixed_mask,
                    return_features=want_feat,
                    task_type=task_type,
                    task_id=task_id,
                )
                warped, flow, aux = self._parse_model_output(out)

                # Important for preAlign + residual registration:
                #   - flow is final_flow = pre_flow composed with residual_flow
                #   - aux["reg_flow"] should be residual_flow from PreAlignWrapper
                # Regularising final_flow would incorrectly penalise the fixed SAMCoarse
                # global pre-alignment field.
                reg_flow = aux.get("reg_flow", flow)
                feat_pairs = aux.get("feat_pairs", None)
                loss_reg = reg_loss_fn(reg_flow)

            with torch.cuda.amp.autocast(enabled=False):
                # NCC in float32 for stability; warped keeps grad to RPNet.
                valid_mask = None
                if pre_align is not None and fixed_mask is not None:
                    valid_mask = self._build_valid_similarity_mask(
                        flow.float(), fixed_mask.float(), moving_mask.float() if moving_mask is not None else None
                    )

                if valid_mask is not None and valid_mask.sum() > 0:
                    masked_sim_loss_fn = self._get_masked_sim_loss(batch)
                    loss_sim = masked_sim_loss_fn(warped.float(), fixed.float(), valid_mask.float())
                    used_masked_sim = True
                else:
                    loss_sim = sim_loss_fn(warped.float(), fixed.float())
                    used_masked_sim = False

                loss = sim_weight * loss_sim + reg_weight * loss_reg
                loss_feat = None
                if feat_pairs is not None and self.use_feature_loss and self.feat_weight > 0:
                    loss_feat = self._feature_loss_from_pairs(feat_pairs)
                    loss = loss + self.feat_weight * loss_feat

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)  

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            out = self._forward_model(
                moving,
                fixed,
                pre_align=pre_align,
                moving_mask=moving_mask,
                fixed_mask=fixed_mask,
                return_features=want_feat,
                task_type=task_type,
                task_id=task_id,
            )
            warped, flow, aux = self._parse_model_output(out)
            reg_flow = aux.get("reg_flow", flow)
            feat_pairs = aux.get("feat_pairs", None)

            valid_mask = None
            if pre_align is not None and fixed_mask is not None:
                valid_mask = self._build_valid_similarity_mask(
                    flow, fixed_mask, moving_mask if moving_mask is not None else None
                )

            if valid_mask is not None and valid_mask.sum() > 0:
                masked_sim_loss_fn = self._get_masked_sim_loss(batch)
                loss_sim = masked_sim_loss_fn(warped, fixed, valid_mask)
                used_masked_sim = True
            else:
                loss_sim = sim_loss_fn(warped, fixed)
                used_masked_sim = False

            loss_reg = reg_loss_fn(reg_flow)
            loss_feat = None
            loss = sim_weight * loss_sim + reg_weight * loss_reg
            if feat_pairs is not None and self.use_feature_loss and self.feat_weight > 0:
                loss_feat = self._feature_loss_from_pairs(feat_pairs)
                loss = loss + self.feat_weight * loss_feat
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        out = {
            "loss":     loss.item(),
            "loss_sim": loss_sim.item(),
            "loss_reg": loss_reg.item(),
        }
        task_name = self._batch_task_name(batch)
        if task_name is not None:
            out[f"task_{task_name}"] = 1.0
            out["sim_weight"] = float(sim_weight)
            out["reg_weight"] = float(reg_weight)
        if pre_align is not None:
            out["use_pre_align"] = 1.0
        if locals().get("used_masked_sim", False):
            out["use_masked_ncc"] = 1.0
        if want_feat:
            out["loss_feat"] = float(loss_feat.item()) if loss_feat is not None else 0.0
        return out

    def _forward_model(
        self,
        moving: torch.Tensor,
        fixed: torch.Tensor,
        pre_align: Optional[torch.Tensor] = None,
        moving_mask: Optional[torch.Tensor] = None,
        fixed_mask: Optional[torch.Tensor] = None,
        return_features: bool = False,
        task_type: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
    ):
        """
        Compatible model forward for mixed model wrappers.

        Different architectures in this project expose slightly different
        forward signatures:
          - PreAlignWrapper-style models may accept ``pre_align``.
          - Dynamic task-conditioned models accept ``task_type`` / ``task_id``.
          - Some RPNet/DynRPNet variants do not accept ``pre_align``.
          - Some older models do not accept task-conditioning arguments.

        We first try the richest call and then remove only the unsupported
        optional keyword(s). This preserves the existing pre-align logic for
        models that support it, while avoiding crashes such as
        ``forward() got an unexpected keyword argument 'pre_align'``.
        """
        kwargs = {}
        if pre_align is not None:
            kwargs["pre_align"] = pre_align
        if moving_mask is not None:
            kwargs["moving_mask"] = moving_mask
        if fixed_mask is not None:
            kwargs["fixed_mask"] = fixed_mask
        if return_features:
            kwargs["return_features"] = True
        if task_type is not None:
            kwargs["task_type"] = task_type
        if task_id is not None:
            kwargs["task_id"] = task_id

        # Try progressively simpler signatures. Keep the order conservative so
        # existing functionality is unchanged whenever the model supports it.
        tried = []
        mask_keys = ("moving_mask", "fixed_mask")
        candidates = [
            dict(kwargs),
            {k: v for k, v in kwargs.items() if k not in mask_keys},
            {k: v for k, v in kwargs.items() if k != "pre_align"},
            {k: v for k, v in kwargs.items() if k not in ("pre_align", *mask_keys)},
            {k: v for k, v in kwargs.items() if k != "return_features"},
            {k: v for k, v in kwargs.items() if k not in ("return_features", *mask_keys)},
            {k: v for k, v in kwargs.items() if k not in ("pre_align", "return_features")},
            {k: v for k, v in kwargs.items() if k not in ("pre_align", "return_features", *mask_keys)},
            {k: v for k, v in kwargs.items() if k not in ("task_type", "task_id")},
            {k: v for k, v in kwargs.items() if k not in ("task_type", "task_id", *mask_keys)},
            {k: v for k, v in kwargs.items() if k not in ("pre_align", "task_type", "task_id")},
            {k: v for k, v in kwargs.items() if k not in ("pre_align", "task_type", "task_id", *mask_keys)},
            {k: v for k, v in kwargs.items() if k not in ("return_features", "task_type", "task_id")},
            {},
        ]

        last_error = None
        for cand in candidates:
            key_tuple = tuple(sorted(cand.keys()))
            if key_tuple in tried:
                continue
            tried.append(key_tuple)
            try:
                return self.model(moving, fixed, **cand)
            except TypeError as e:
                last_error = e
                msg = str(e)
                # Only retry for unsupported optional kwargs. Other TypeErrors
                # inside the model should still be raised.
                if not any(
                    token in msg
                    for token in (
                        "unexpected keyword argument",
                        "got an unexpected keyword",
                        "pre_align",
                        "return_features",
                        "task_type",
                        "task_id",
                        "moving_mask",
                        "fixed_mask",
                    )
                ):
                    raise
        raise last_error

    def _parse_model_output(self, out):
        """
        Normalise model outputs to: warped, flow, aux_dict.

        Supported outputs:
          - (warped, flow)
          - (warped, flow, feat_pairs)
          - (warped, flow, aux_dict)

        PreAlignWrapper should return:
          (warped_final, final_flow, {
              "pre_flow": pre_flow,
              "res_flow": residual_flow,
              "reg_flow": residual_flow,
              ...
          })
        """
        if not isinstance(out, (tuple, list)) or len(out) < 2:
            raise RuntimeError(
                "Model forward should return at least (warped, flow). "
                f"Got type={type(out)}."
            )

        warped, flow = out[0], out[1]
        aux: Dict[str, Any] = {}

        if len(out) >= 3:
            third = out[2]
            if isinstance(third, dict):
                aux.update(third)

                # Some wrappers may store original model extras under base_aux.
                # If base_aux contains feature pairs, expose them as feat_pairs.
                if "feat_pairs" not in aux and "base_aux" in aux:
                    base_aux = aux.get("base_aux")
                    if isinstance(base_aux, (tuple, list)) and len(base_aux) > 0:
                        if isinstance(base_aux[0], dict):
                            aux["feat_pairs"] = base_aux[0].get("feat_pairs")
                        else:
                            aux["feat_pairs"] = base_aux[0]
            else:
                aux["feat_pairs"] = third

        return warped, flow, aux

    def _unpack_aux(self, out):
        """Backward-compatible helper: return feat_pairs only."""
        if len(out) != 3:
            return None
        third = out[2]
        if isinstance(third, dict):
            return third.get("feat_pairs")
        return third

    def _model_supports_features(self) -> bool:
        """Check if model.forward() accepts return_features or arbitrary **kwargs."""
        import inspect
        sig = inspect.signature(self.model.forward)
        if "return_features" in sig.parameters:
            return True
        return any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )

    def _feature_loss_from_pairs(
        self, feat_pairs: List,
    ) -> torch.Tensor:
        """
        Compute feature loss from (warped_fx, fy) pairs returned by model.forward().
        Each pair: (warped_moving_feat_at_level_i, fixed_feat_at_level_i).
        """
        weights = self.feat_level_weights
        lf = 0.0
        for i, (wfx, fy) in enumerate(feat_pairs):
            w = float(weights[i]) if i < len(weights) else 1.0
            lf = lf + w * self.feat_loss(wfx, fy)
        return lf

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Run full validation.

        Multi-task validation metrics:
          - dice_mean_pooled:
              Mean Dice over all validation pairs directly. This metric is
              affected by the number of validation pairs in each task.

          - dice_<task_name>:
              Mean Dice of each task.

          - dice_mean:
              Task-balanced mean Dice. Dice is first averaged within each
              task, and then averaged across tasks. This key is intentionally
              kept as ``dice_mean`` so the existing best.pth selection logic
              automatically uses task-balanced validation Dice.

        Dice preference:
          - If batch provides moving_seg_raw/fixed_seg_raw/raw_shape, evaluate
            Dice in raw space by upsampling flow to raw_shape and warping raw
            segmentation.
          - Otherwise, fall back to compute-space segmentations.
        """
        self.model.eval()
        stn_nn = self.stn_nn.to(self.device)

        dice_all = []
        fold_all = []
        tre_all = []

        dice_by_task = defaultdict(list)
        fold_by_task = defaultdict(list)
        tre_by_task = defaultdict(list)

        for batch in val_loader:
            task_name = self._batch_task_name(batch)
            if task_name is None:
                task_name = "unknown"

            moving = batch["moving"].to(self.device, dtype=torch.float32)
            fixed = batch["fixed"].to(self.device, dtype=torch.float32)
            task_type, task_id = self._get_task_ids(batch)

            # Optional SAMCoarse pre-alignment. Validation should evaluate the
            # final composed flow, not the residual flow alone.
            pre_align = batch.get("pre_align", None)
            if pre_align is not None:
                pre_align = pre_align.to(self.device, dtype=torch.float32)

            moving_mask = batch.get("moving_mask", None)
            fixed_mask = batch.get("fixed_mask", None)
            if moving_mask is not None:
                moving_mask = moving_mask.to(self.device, dtype=torch.float32)
            if fixed_mask is not None:
                fixed_mask = fixed_mask.to(self.device, dtype=torch.float32)

            out = self._forward_model(
                moving,
                fixed,
                pre_align=pre_align,
                moving_mask=moving_mask,
                fixed_mask=fixed_mask,
                return_features=False,
                task_type=task_type,
                task_id=task_id,
            )
            warped, flow, aux = self._parse_model_output(out)

            # ------------------------------------------------------------
            # Dice
            # ------------------------------------------------------------
            moving_seg = batch.get("moving_seg")
            fixed_seg = batch.get("fixed_seg")
            dice_value = None

            # Prefer raw-space evaluation when raw fields are provided.
            if (
                "moving_seg_raw" in batch
                and "fixed_seg_raw" in batch
                and "raw_shape" in batch
            ):
                ms_raw = batch["moving_seg_raw"].to(
                    self.device, dtype=torch.float32
                ).unsqueeze(1)
                fs_raw = batch["fixed_seg_raw"].squeeze().numpy()

                raw_shape_item = batch["raw_shape"]
                if torch.is_tensor(raw_shape_item):
                    raw_shape_item = raw_shape_item.view(-1).cpu().numpy().tolist()
                elif isinstance(raw_shape_item, (list, tuple)) and len(raw_shape_item) > 0:
                    # DataLoader may collate tuples as list/tuple of tensors.
                    if all(torch.is_tensor(x) for x in raw_shape_item):
                        raw_shape_item = [int(x.view(-1)[0].item()) for x in raw_shape_item]
                raw_shape = tuple(int(x) for x in raw_shape_item[:3])

                from unireg.core.spaces import rescale_flow

                flow_up = rescale_flow(flow, target_size=raw_shape)
                ws = stn_nn(ms_raw, flow_up).long().squeeze().cpu().numpy()
                d = dice_per_label(ws, fs_raw)
                dice_value = float(np.mean(d))

            elif moving_seg is not None and fixed_seg is not None:
                ms = moving_seg.to(self.device, dtype=torch.float32)
                ws = stn_nn(ms, flow).long().squeeze().cpu().numpy()
                fs = fixed_seg.squeeze().numpy()
                d = dice_per_label(ws, fs)
                dice_value = float(np.mean(d))

            if dice_value is not None:
                dice_all.append(dice_value)
                dice_by_task[task_name].append(dice_value)

            # ------------------------------------------------------------
            # Folding ratio
            # ------------------------------------------------------------
            fd = compute_foldings(flow.cpu())
            fold_value = float(fd["ratio"])
            fold_all.append(fold_value)
            fold_by_task[task_name].append(fold_value)

            # ------------------------------------------------------------
            # TRE, only when keypoints and spacing/shape are provided.
            # ------------------------------------------------------------
            if (
                "pts_fix" in batch
                and "pts_mov" in batch
                and "spacing_xyz" in batch
                and "shape_xyz" in batch
            ):
                pts_fix = batch["pts_fix"]
                pts_mov = batch["pts_mov"]
                tre = compute_tre_mm(
                    flow,
                    pts_fix.numpy() if hasattr(pts_fix, "numpy") else pts_fix,
                    pts_mov.numpy() if hasattr(pts_mov, "numpy") else pts_mov,
                    tuple(batch["spacing_xyz"]),
                    tuple(batch["shape_xyz"]),
                    device=self.device,
                )
                tre_value = float(np.mean(tre))
                tre_all.append(tre_value)
                tre_by_task[task_name].append(tre_value)

        self.model.train()

        result: Dict[str, float] = {}

        # ------------------------------------------------------------
        # Dice summary
        # ------------------------------------------------------------
        if dice_all:
            # Pooled mean over all validation pairs.
            result["dice_mean_pooled"] = float(np.mean(dice_all))
            result["dice_std_pooled"] = float(np.std(dice_all))

            # Per-task Dice and task-balanced mean Dice.
            task_dice_means = []
            for task_name in sorted(dice_by_task.keys()):
                values = dice_by_task[task_name]
                if len(values) == 0:
                    continue
                task_mean = float(np.mean(values))
                task_std = float(np.std(values))
                result[f"dice_{task_name}"] = task_mean
                result[f"dice_{task_name}_std"] = task_std
                result[f"num_{task_name}"] = float(len(values))
                task_dice_means.append(task_mean)

            if task_dice_means:
                # This is used by run() for best.pth selection.
                result["dice_mean"] = float(np.mean(task_dice_means))
                result["dice_std"] = float(np.std(task_dice_means))

        # ------------------------------------------------------------
        # TRE summary
        # ------------------------------------------------------------
        if tre_all:
            result["tre_mean_pooled"] = float(np.mean(tre_all))
            result["tre_std_pooled"] = float(np.std(tre_all))

            task_tre_means = []
            for task_name in sorted(tre_by_task.keys()):
                values = tre_by_task[task_name]
                if len(values) == 0:
                    continue
                task_mean = float(np.mean(values))
                task_std = float(np.std(values))
                result[f"tre_{task_name}"] = task_mean
                result[f"tre_{task_name}_std"] = task_std
                task_tre_means.append(task_mean)

            if task_tre_means:
                result["tre_mean"] = float(np.mean(task_tre_means))
                result["tre_std"] = float(np.std(task_tre_means))

        # ------------------------------------------------------------
        # Folding summary
        # ------------------------------------------------------------
        if fold_all:
            result["jac_ratio_pooled"] = float(np.mean(fold_all))

            task_fold_means = []
            for task_name in sorted(fold_by_task.keys()):
                values = fold_by_task[task_name]
                if len(values) == 0:
                    continue
                task_mean = float(np.mean(values))
                result[f"jac_ratio_{task_name}"] = task_mean
                task_fold_means.append(task_mean)

            if task_fold_means:
                result["jac_ratio"] = float(np.mean(task_fold_means))

        print(
            f"[val step={self._step}] "
            + "  ".join(f"{k}={v:.4f}" for k, v in result.items()),
            flush=True,
        )
        return result

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def _save(self, name: str):
        path = os.path.join(self.log_dir, name)
        save_checkpoint(
            path,
            self.model,
            self.optimizer,
            step=self._step,
            epoch=self._epoch,
            metrics={
                "best_dice": self._best_dice,
                "best_tre":  self._best_tre,
                "select_metric": self.select_metric,
            },
        )

    def resume(self, path: str):
        info = load_checkpoint(path, self.model, self.optimizer, device=self.device)
        self._step  = info.get("step", 0)
        self._epoch = info.get("epoch", 0)
        metrics = info.get("metrics", {}) or {}
        self._best_dice = metrics.get("best_dice", -1.0)
        self._best_tre  = metrics.get("best_tre", float("inf"))
        sel = metrics.get("select_metric")
        if sel:
            self.select_metric = str(sel).lower().strip()
        print(f"[Trainer] Resumed from {path}  step={self._step}", flush=True)


