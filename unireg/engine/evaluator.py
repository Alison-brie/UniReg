"""
unireg/engine/evaluator.py
========================
Evaluator: pure inference/benchmark mode.
No training, no optimizer. Just load model, iterate test set, compute metrics.
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List

from unireg.core.metrics import metric_bag, compute_tre_mm
from unireg.engine.logger import MetricLogger


def compute_jacobian_det_3d(flow: torch.Tensor) -> torch.Tensor:
    """
    Compute Jacobian determinant for a 3D displacement field.

    Args:
        flow: Tensor with shape [B, 3, D, H, W]. This function assumes
              phi(x) = x + flow(x), i.e., flow is a displacement field.

    Returns:
        jac_det: Tensor with shape [B, D-2, H-2, W-2].
    """
    if flow.ndim != 5:
        raise ValueError(f"Expected 5D flow tensor, got shape {tuple(flow.shape)}")

    # Accept either [B, 3, D, H, W] or [B, D, H, W, 3].
    if flow.shape[1] == 3:
        f = flow
    elif flow.shape[-1] == 3:
        f = flow.permute(0, 4, 1, 2, 3).contiguous()
    else:
        raise ValueError(f"Expected flow channel dimension of size 3, got shape {tuple(flow.shape)}")

    u = f[:, 0]
    v = f[:, 1]
    w = f[:, 2]

    du_dx = (u[:, 1:-1, 1:-1, 2:] - u[:, 1:-1, 1:-1, :-2]) / 2.0
    du_dy = (u[:, 1:-1, 2:, 1:-1] - u[:, 1:-1, :-2, 1:-1]) / 2.0
    du_dz = (u[:, 2:, 1:-1, 1:-1] - u[:, :-2, 1:-1, 1:-1]) / 2.0

    dv_dx = (v[:, 1:-1, 1:-1, 2:] - v[:, 1:-1, 1:-1, :-2]) / 2.0
    dv_dy = (v[:, 1:-1, 2:, 1:-1] - v[:, 1:-1, :-2, 1:-1]) / 2.0
    dv_dz = (v[:, 2:, 1:-1, 1:-1] - v[:, :-2, 1:-1, 1:-1]) / 2.0

    dw_dx = (w[:, 1:-1, 1:-1, 2:] - w[:, 1:-1, 1:-1, :-2]) / 2.0
    dw_dy = (w[:, 1:-1, 2:, 1:-1] - w[:, 1:-1, :-2, 1:-1]) / 2.0
    dw_dz = (w[:, 2:, 1:-1, 1:-1] - w[:, :-2, 1:-1, 1:-1]) / 2.0

    j11 = 1.0 + du_dx
    j12 = du_dy
    j13 = du_dz

    j21 = dv_dx
    j22 = 1.0 + dv_dy
    j23 = dv_dz

    j31 = dw_dx
    j32 = dw_dy
    j33 = 1.0 + dw_dz

    jac_det = (
        j11 * (j22 * j33 - j23 * j32)
        - j12 * (j21 * j33 - j23 * j31)
        + j13 * (j21 * j32 - j22 * j31)
    )
    return jac_det


def compute_sdlogj_from_flow(flow: torch.Tensor, eps: float = 1e-6) -> Dict[str, float]:
    """
    Compute SDlogJ from a predicted 3D displacement field.

    SDlogJ is the standard deviation of log(J), where
    J = det(I + grad(flow)). Non-positive Jacobian values are excluded
    from the log operation and are reported separately as jac_ratio_sdlogj.
    """
    with torch.no_grad():
        jac_det = compute_jacobian_det_3d(flow.detach())
        jac_ratio = (jac_det <= 0).float().mean().item()

        valid_jac = jac_det[jac_det > eps]
        if valid_jac.numel() == 0:
            sdlogj = float("nan")
            logj_mean = float("nan")
        else:
            logj = torch.log(valid_jac)
            sdlogj = torch.std(logj).item()
            logj_mean = torch.mean(logj).item()

    return {
        "sdlogj": float(sdlogj),
        "logj_mean": float(logj_mean),
        "jac_ratio_sdlogj": float(jac_ratio),
    }


class Evaluator:
    """
    Pure inference + metric evaluation.

    Usage:
        evaluator = Evaluator(model, compute_size=(80,192,192), device='cuda')
        results = evaluator.run(test_loader)
        print(results)  # list of per-pair metric dicts
    """

    def __init__(
        self,
        model: nn.Module,
        compute_size: tuple,
        device: str = "cuda",
        log_dir: Optional[str] = None,
        iter_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.model        = model.to(device)
        self.compute_size = compute_size
        self.device       = device
        self.log_dir = log_dir or "."
        self.logger = MetricLogger(log_dir or ".", "eval") if log_dir else None

        iter_cfg = iter_cfg or {}
        self.iter_cfg = iter_cfg
        self._iter_enabled = bool(iter_cfg.get("iter_stats", False) or iter_cfg.get("log_iter_stats", False))
        self._iter_verbose_n = int(iter_cfg.get("iter_verbose_n", 5))
        self._iter_max_iter = int(iter_cfg.get("max_iter", 10))
        self._iter_guide_mode = str(iter_cfg.get("guide_mode", "ncc")).lower()
        self.iter_summary: Optional[Dict[str, Any]] = None
        self._pre_align_format = str(iter_cfg.get("pre_align_format", "same")).lower()

    def _forward_model(self, moving, fixed, pre_align=None, moving_mask=None, fixed_mask=None, return_debug: bool = False, task_type=None, task_id=None):
        """
        Compatible model forward.

        Some models accept pre_align / return_debug, while others only accept
        (moving, fixed). This wrapper keeps evaluator.py usable for both cases.
        """
        kwargs = {}
        if pre_align is not None:
            kwargs["pre_align"] = pre_align
        if moving_mask is not None:
            kwargs["moving_mask"] = moving_mask
        if fixed_mask is not None:
            kwargs["fixed_mask"] = fixed_mask
        if pre_align is not None and self._pre_align_format:
            kwargs["pre_align_format"] = self._pre_align_format
        if return_debug:
            kwargs["return_debug"] = True
        if task_type is not None:
            kwargs["task_type"] = task_type
        if task_id is not None:
            kwargs["task_id"] = task_id

        try:
            return self.model(moving, fixed, **kwargs)
        except TypeError as e:
            msg = str(e)

            # Model does not support pre_align. Retry without it.
            if "pre_align" in kwargs and (
                "unexpected keyword argument 'pre_align'" in msg
                or "got an unexpected keyword argument 'pre_align'" in msg
                or "pre_align" in msg
            ):
                kwargs.pop("pre_align", None)
                try:
                    return self.model(moving, fixed, **kwargs)
                except TypeError as e2:
                    msg2 = str(e2)
                    if "return_debug" in kwargs and (
                        "unexpected keyword argument 'return_debug'" in msg2
                        or "got an unexpected keyword argument 'return_debug'" in msg2
                        or "return_debug" in msg2
                    ):
                        kwargs.pop("return_debug", None)
                        return self.model(moving, fixed, **kwargs)
                    raise

            # Model does not support return_debug. Retry without it.
            if "return_debug" in kwargs and (
                "unexpected keyword argument 'return_debug'" in msg
                or "got an unexpected keyword argument 'return_debug'" in msg
                or "return_debug" in msg
            ):
                kwargs.pop("return_debug", None)
                try:
                    return self.model(moving, fixed, **kwargs)
                except TypeError as e2:
                    msg2 = str(e2)
                    if "pre_align" in kwargs and (
                        "unexpected keyword argument 'pre_align'" in msg2
                        or "got an unexpected keyword argument 'pre_align'" in msg2
                        or "pre_align" in msg2
                    ):
                        kwargs.pop("pre_align", None)
                        return self.model(moving, fixed, **kwargs)
                    raise

            # Models/wrappers that do not consume masks should still run.
            if "moving_mask" in kwargs or "fixed_mask" in kwargs:
                kwargs.pop("moving_mask", None)
                kwargs.pop("fixed_mask", None)
                return self.model(moving, fixed, **kwargs)

            # Task-conditioned models accept task_type/task_id; old models do not.
            if "task_type" in kwargs or "task_id" in kwargs:
                kwargs.pop("task_type", None)
                kwargs.pop("task_id", None)
                return self.model(moving, fixed, **kwargs)

            raise


    @staticmethod
    def _collated_to_scalar(x):
        """Make DataLoader-collated metadata readable."""
        if isinstance(x, (list, tuple)):
            return Evaluator._collated_to_scalar(x[0]) if len(x) else ""
        if torch.is_tensor(x):
            if x.numel() == 1:
                return x.item()
            return x.detach().cpu().numpy().tolist()
        return x

    def _compute_tre_from_flow(self, flow: torch.Tensor, batch: Dict[str, Any], key_prefix: str) -> Dict[str, float]:
        """Compute TRE using the same convention as metric_bag and return named metrics."""
        if batch.get("pts_fix") is None or batch.get("pts_mov") is None:
            return {}
        if batch.get("spacing_xyz") is None or batch.get("shape_xyz") is None:
            return {}

        pts_fix = batch["pts_fix"]
        pts_mov = batch["pts_mov"]
        pts_fix_np = pts_fix.numpy() if hasattr(pts_fix, "numpy") else pts_fix
        pts_mov_np = pts_mov.numpy() if hasattr(pts_mov, "numpy") else pts_mov
        spacing_xyz = tuple(batch["spacing_xyz"])
        shape_xyz = tuple(batch["shape_xyz"])
        tre = compute_tre_mm(
            flow=flow,
            fixed_pts_np=pts_fix_np,
            moving_pts_np=pts_mov_np,
            spacing_xyz=spacing_xyz,
            shape_xyz=shape_xyz,
            device=self.device,
        )
        return {
            f"{key_prefix}_tre_mean": float(tre.mean()),
            f"{key_prefix}_tre_std": float(tre.std()),
        }

    @torch.no_grad()
    def run(self, test_loader: DataLoader) -> List[Dict[str, Any]]:
        """
        Evaluate on all pairs in test_loader.

        Returns list of per-pair metric dicts.
        """
        self.model.eval()
        results = []

        # Iteration stats accumulators (UnifiedIIRPNet debug)
        iter_acc = None
        if self._iter_enabled:
            iter_acc = {
                "iters": {k: [] for k in ("l4", "l3", "l2", "l1")},
                "early_stop": {k: [] for k in ("l4", "l3", "l2", "l1")},
                "final_score": {k: [] for k in ("l4", "l3", "l2", "l1")},
                "final_comp": {k: {"cos": [], "mse": [], "ncc": []} for k in ("l4", "l3", "l2", "l1")},
            }

        for idx, batch in enumerate(test_loader):
            moving = batch["moving"].to(self.device, dtype=torch.float32)
            fixed  = batch["fixed"].to(self.device, dtype=torch.float32)
            task_type = batch.get("task_type_id", None)
            task_id = batch.get("task_id_id", None)
            if task_type is not None:
                task_type = task_type.to(self.device).view(-1).long()
            if task_id is not None:
                task_id = task_id.to(self.device).view(-1).long()
            
            pre_align = batch.get("pre_align", None)
            if pre_align is not None:
                pre_align = pre_align.to(self.device, dtype=torch.float32)

            pre_align = batch.get("pre_align", None)
            if pre_align is not None:
                pre_align = pre_align.to(self.device, dtype=torch.float32)

            moving_mask = batch.get("moving_mask", None)
            fixed_mask = batch.get("fixed_mask", None)
            if moving_mask is not None:
                moving_mask = moving_mask.to(self.device, dtype=torch.float32)
            if fixed_mask is not None:
                fixed_mask = fixed_mask.to(self.device, dtype=torch.float32)

            dbg = None
            if self._iter_enabled:
                out = self._forward_model(moving, fixed, pre_align=pre_align, moving_mask=moving_mask, fixed_mask=fixed_mask, return_debug=True, task_type=task_type, task_id=task_id)
                # possible returns:
                #   (warped, flow, dbg)
                #   (warped, flow, feat_pairs, dbg)
                #   (warped, flow) if return_debug is not supported
                if isinstance(out, (tuple, list)) and len(out) >= 3 and isinstance(out[-1], dict):
                    dbg = out[-1]
                    warped, flow_norm = out[0], out[1]
                else:
                    warped, flow_norm = out[0], out[1]
            else:
                out = self._forward_model(moving, fixed, pre_align=pre_align, moving_mask=moving_mask, fixed_mask=fixed_mask, return_debug=False, task_type=task_type, task_id=task_id)
                warped, flow_norm = out[0], out[1]

            # Move to CPU for metric computation
            warped_np = warped.squeeze().cpu().numpy()
            flow_cpu  = flow_norm.cpu()

            # Compute SDlogJ on the predicted displacement field.
            # This assumes flow_norm is used as displacement in SpatialTransformer.
            sdlogj_metrics = compute_sdlogj_from_flow(flow_norm)

            moving_seg = batch.get("moving_seg")
            fixed_seg  = batch.get("fixed_seg")
            warped_seg_np = None
            fixed_seg_np = None

            if "moving_seg_raw" in batch and "fixed_seg_raw" in batch and "raw_shape" in batch:
                ms = batch["moving_seg_raw"].to(self.device, dtype=torch.float32).unsqueeze(1)  # [B,1,D,H,W]
                fs = batch["fixed_seg_raw"].squeeze().numpy()
                raw_shape = tuple(batch["raw_shape"][i].item() for i in range(3))
                
                # upscale flow to native size
                from unireg.core.spaces import rescale_flow
                flow_up = rescale_flow(flow_norm, target_size=raw_shape)
                
                from unireg.core.transforms import SpatialTransformer
                stn = SpatialTransformer(mode="nearest").to(self.device)
                w_seg = stn(ms, flow_up).long().squeeze().cpu().numpy()
                warped_seg_np = w_seg
                fixed_seg_np  = fs
                
            elif moving_seg is not None:
                # Warp moving segmentation with predicted flow at compute size
                from unireg.core.transforms import SpatialTransformer
                stn = SpatialTransformer(mode="nearest")
                stn = stn.to(self.device)
                ms = moving_seg.to(self.device, dtype=torch.float32)
                w_seg = stn(ms, flow_norm).long().squeeze().cpu().numpy()
                warped_seg_np = w_seg

                if fixed_seg is not None:
                    fixed_seg_np = fixed_seg.squeeze().numpy()

            kwargs = dict(warped_seg=warped_seg_np, fixed_seg=fixed_seg_np, flow=flow_cpu, device=self.device)
            if batch.get("pts_fix") is not None and batch.get("pts_mov") is not None:
                pts_fix = batch["pts_fix"]
                pts_mov = batch["pts_mov"]
                kwargs["pts_fix"] = pts_fix.numpy() if hasattr(pts_fix, "numpy") else pts_fix
                kwargs["pts_mov"] = pts_mov.numpy() if hasattr(pts_mov, "numpy") else pts_mov
                if batch.get("spacing_xyz") is not None and batch.get("shape_xyz") is not None:
                    kwargs["spacing_xyz"] = tuple(batch["spacing_xyz"])
                    kwargs["shape_xyz"] = tuple(batch["shape_xyz"])
            m = metric_bag(**kwargs)

            # DIR-QA convenience metrics:

            m.update(sdlogj_metrics)
            m["pair_idx"] = idx
            task_name = batch.get("task_name", None)
            if isinstance(task_name, (list, tuple)):
                task_name = task_name[0]
            if task_name is not None:
                m["task_name"] = str(task_name)
            m.update(batch.get("meta", {}))

            results.append(m)
            if self.logger:
                self.logger.log(idx, m, phase="eval")

            # ---- Iteration stats aggregation / concise debug logs ----
            if iter_acc is not None and isinstance(dbg, dict):
                # l4/l3/l2/l1
                for lk in ("l4", "l3", "l2", "l1"):
                    iters = int((dbg.get("level_iters") or {}).get(lk, 0))
                    iter_acc["iters"][lk].append(iters)

                    hist = (dbg.get("score_hist") or {}).get(lk, []) or []
                    iter_acc["early_stop"][lk].append(int(len(hist) < self._iter_max_iter))
                    if len(hist) > 0:
                        iter_acc["final_score"][lk].append(float(hist[-1]))
                        comp = (dbg.get("comp_hist") or {}).get(lk, []) or []
                        if len(comp) > 0 and isinstance(comp[-1], dict):
                            c = comp[-1]
                            for ck in ("cos", "mse", "ncc"):
                                v = c.get(ck)
                                if v is not None:
                                    iter_acc["final_comp"][lk][ck].append(float(v))
                    # else: leave empty

                # detailed history for first few samples only
                if idx < self._iter_verbose_n:
                    def _fmt_hist(h):
                        if not h:
                            return "[]"
                        if len(h) <= 6:
                            return "[" + ", ".join(f"{x:.4f}" for x in h) + "]"
                        head = ", ".join(f"{x:.4f}" for x in h[:3])
                        tail = ", ".join(f"{x:.4f}" for x in h[-2:])
                        return "[" + head + ", ..., " + tail + f"]({len(h)})"

                    print(f"[iter] pair_idx={idx} guide_mode={self._iter_guide_mode}", flush=True)
                    for lk in ("l4", "l3", "l2", "l1"):
                        h = (dbg.get("score_hist") or {}).get(lk, []) or []
                        it = int((dbg.get("level_iters") or {}).get(lk, 0))
                        print(f"  {lk}: accept_iters={it:2d} early_stop={int(len(h) < self._iter_max_iter)} score_hist={_fmt_hist(h)}", flush=True)

                        # print per-step alpha for iterative guide modes that produce it
                        if self._iter_guide_mode in ("iiv1", "iiv_refmag"):
                            comp = (dbg.get("comp_hist") or {}).get(lk, []) or []
                            # collect chosen alphas per step
                            alphas = []
                            meta = []
                            for c in comp:
                                if isinstance(c, dict) and c.get("alpha") is not None:
                                    a = float(c["alpha"])
                                    alphas.append(a)
                                    meta.append({
                                        "alpha": a,
                                        "best_delta": c.get("best_delta_raw", c.get("best_delta")),
                                        "best_rel": c.get("best_rel_gain"),
                                        "kept": c.get("kept"),
                                        "ref_mag": c.get("ref_mag"),
                                        "flow_mag": c.get("flow_mag"),
                                    })
                            if alphas:
                                # print first few steps only, plus a distribution summary
                                show = meta[:min(5, len(meta))]
                                if self._iter_guide_mode == "iiv1":
                                    show_str = ", ".join(
                                        f"(a={m['alpha']:.3f}, d={float(m['best_delta']):+.4g})"
                                        for m in show
                                        if m.get("best_delta") is not None
                                    )
                                else:
                                    show_str = ", ".join(
                                        f"(a={m['alpha']:.3f}, m_ref={float(m['ref_mag']):.4g}, m={float(m['flow_mag']):.4g})"
                                        for m in show
                                        if m.get("ref_mag") is not None and m.get("flow_mag") is not None
                                    )
                                # distribution
                                dist = {}
                                for a in alphas:
                                    dist[a] = dist.get(a, 0) + 1
                                dist_str = ", ".join(f"{k:g}×{v}" for k, v in sorted(dist.items(), key=lambda kv: kv[0]))
                                print(f"       {self._iter_guide_mode}: alpha(steps)={show_str} | alpha_dist={dist_str}", flush=True)

        self._print_summary(results)
        if iter_acc is not None:
            self.iter_summary = self._iter_summary(iter_acc)
            self._print_iter_summary(self.iter_summary)
        return results

    @staticmethod
    def _print_summary(results: List[Dict]):
        dice_vals = [r["dice_mean"] for r in results if "dice_mean" in r]
        tre_vals  = [r["tre_mean"]  for r in results if "tre_mean"  in r]
        initial_tre_vals = [r["initial_tre_mean"] for r in results if "initial_tre_mean" in r]
        jac_vals  = [r["jac_ratio"] for r in results if "jac_ratio" in r]
        sdlogj_vals = [r["sdlogj"] for r in results if "sdlogj" in r and not np.isnan(r["sdlogj"])]

        print("=" * 50, flush=True)
        print("EVALUATION SUMMARY", flush=True)
        if dice_vals:
            print(f"  Dice    : {np.mean(dice_vals):.4f} ± {np.std(dice_vals):.4f}", flush=True)
        if initial_tre_vals:
            print(f"  Initial TRE(mm): {np.mean(initial_tre_vals):.3f} ± {np.std(initial_tre_vals):.3f}", flush=True)
        if tre_vals:
            print(f"  Final TRE(mm)  : {np.mean(tre_vals):.3f} ± {np.std(tre_vals):.3f}", flush=True)
        if jac_vals:
            print(f"  Jac<0   : {np.mean(jac_vals)*100:.2f}%", flush=True)
        if sdlogj_vals:
            print(f"  SDlogJ  : {np.mean(sdlogj_vals):.4f} ± {np.std(sdlogj_vals):.4f}", flush=True)
        print("=" * 50, flush=True)

    @staticmethod
    def _iter_summary(iter_acc: Dict[str, Any]) -> Dict[str, Any]:
        def _mean(x):
            return float(np.mean(x)) if len(x) else float("nan")

        out = {"levels": {}, "overall": {}}
        for lk in ("l4", "l3", "l2", "l1"):
            it = iter_acc["iters"][lk]
            es = iter_acc["early_stop"][lk]
            fs = iter_acc["final_score"][lk]
            out["levels"][lk] = {
                "accept_iters_mean": _mean(it),
                "early_stop_ratio": _mean(es),
                "final_score_mean": _mean(fs),
                "final_comp_mean": {
                    ck: _mean(iter_acc["final_comp"][lk][ck])
                    for ck in ("ncc", "cos", "mse")
                },
            }
        return out

    @staticmethod
    def _print_iter_summary(summary: Dict[str, Any]):
        print("=" * 50, flush=True)
        print("IIRP ITERATION SUMMARY", flush=True)
        for lk in ("l4", "l3", "l2", "l1"):
            s = summary["levels"][lk]
            print(
                f"  {lk}: accept_iters={s['accept_iters_mean']:.3f} | "
                f"early_stop={s['early_stop_ratio']*100:.1f}% | "
                f"final_score={s['final_score_mean']:.4f}",
                flush=True,
            )
            cm = s["final_comp_mean"]
            comps = []
            for ck in ("ncc", "cos", "mse"):
                v = cm.get(ck)
                if v is not None and not np.isnan(v):
                    comps.append(f"{ck}={v:.4f}")
            if comps:
                print("       comps: " + ", ".join(comps), flush=True)
        print("=" * 50, flush=True)
