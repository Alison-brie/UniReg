"""
unireg/engine/checkpoint.py
=========================
统一权重的保存与加载。

- 保存：model.state_dict()，若为 DataParallel 则用 model.module 避免 module. 前缀
- 加载：支持常见 wrapper 前缀的兼容匹配，例如：
    * module.xxx        <-> xxx
    * xxx               <-> base_model.xxx
    * base_model.xxx    <-> xxx
  这用于兼容 PreAlignWrapper(model) 后模型参数名前缀变为 base_model.xxx 的情况。
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: int = 0,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
):
    """保存统一权重：backbone + decoder 等完整模型。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    payload = {
        "model_state_dict": state,
        "step":             step,
        "epoch":            epoch,
        "metrics":          metrics or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)
    print(f"[checkpoint] Saved → {path} (step={step}, epoch={epoch})", flush=True)


def _strip_prefix_once(k: str, prefix: str) -> str:
    return k[len(prefix):] if k.startswith(prefix) else k


def _candidate_keys(k: str) -> List[str]:
    """
    Generate conservative candidate names for fuzzy checkpoint loading.

    Main target case:
      checkpoint: net.encoder.xxx
      model:      base_model.net.encoder.xxx   # PreAlignWrapper

    Also supports the reverse direction:
      checkpoint: base_model.net.encoder.xxx
      model:      net.encoder.xxx
    """
    cands: List[str] = []

    def add(x: str):
        if x not in cands:
            cands.append(x)

    # original and DataParallel-cleaned variants
    add(k)
    k_no_module = _strip_prefix_once(k, "module.")
    add(k_no_module)

    # Add base_model. for loading unwrapped checkpoints into PreAlignWrapper.
    add("base_model." + k_no_module)

    # Remove base_model. for loading wrapped checkpoints into unwrapped models.
    if k_no_module.startswith("base_model."):
        add(k_no_module[len("base_model."):])

    # Handle nested DataParallel + PreAlignWrapper variants.
    if k.startswith("module.base_model."):
        add(k[len("module.base_model."):])
    if k.startswith("module."):
        add("base_model." + k[len("module."):])

    return cands


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Load checkpoint with fuzzy key matching.

    This loader intentionally supports wrapper-prefix compatibility:
      - normal model <-> PreAlignWrapper(model), where keys gain/lose "base_model."
      - DataParallel/DDP "module." prefix

    It still requires tensor shapes to match, so dynamic heads that do not exist in
    the source checkpoint remain randomly initialized.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    raw = torch.load(path, map_location=device)

    # Detect state dict location.
    sd: Dict = (
        raw.get("model_state_dict") or
        raw.get("state_dict") or
        raw.get("model") or
        raw
    )

    model_sd = model.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    source_to_target: Dict[str, str] = {}
    not_in_model = []
    shape_mismatch = []
    duplicated_target = []

    for k, v in sd.items():
        matched_key = None
        candidate_shape_mismatches = []

        for kk in _candidate_keys(str(k)):
            if kk not in model_sd:
                continue
            if v.shape != model_sd[kk].shape:
                candidate_shape_mismatches.append(f"{k} -> {kk}: {v.shape} vs {model_sd[kk].shape}")
                continue
            matched_key = kk
            break

        if matched_key is None:
            if candidate_shape_mismatches:
                shape_mismatch.extend(candidate_shape_mismatches[:1])
            else:
                not_in_model.append(k)
            continue

        if matched_key in filtered:
            duplicated_target.append(f"{k} -> {matched_key}")
            continue

        filtered[matched_key] = v
        source_to_target[k] = matched_key

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    total = len(model_sd)
    loaded = len(filtered)

    print(f"[checkpoint] Loaded {loaded}/{total} params matched", flush=True)

    # Show prefix remapping examples when useful.
    remapped = [(s, t) for s, t in source_to_target.items() if s != t]
    if remapped:
        print(f"[checkpoint] Remapped {len(remapped)} keys (first 5): {remapped[:5]}", flush=True)

    if not_in_model:
        print(f"[checkpoint] Skipped {len(not_in_model)} keys (not in model, first 5): {not_in_model[:5]}", flush=True)
    if shape_mismatch:
        print(f"[checkpoint] Skipped {len(shape_mismatch)} keys (shape mismatch, first 3): {shape_mismatch[:3]}", flush=True)
    if duplicated_target:
        print(f"[checkpoint] Skipped {len(duplicated_target)} duplicated target keys (first 3): {duplicated_target[:3]}", flush=True)
    if missing:
        print(f"[checkpoint] {len(missing)} keys in model but not in checkpoint (first 5): {missing[:5]}", flush=True)

    if strict and missing:
        raise RuntimeError(f"Strict checkpoint loading failed: {len(missing)} missing keys.")

    if optimizer is not None and "optimizer_state_dict" in raw:
        try:
            optimizer.load_state_dict(raw["optimizer_state_dict"])
        except Exception as e:
            print(f"[checkpoint] Could not restore optimizer: {e}", flush=True)

    return {
        "step":    raw.get("step", 0) if isinstance(raw, dict) else 0,
        "epoch":   raw.get("epoch", 0) if isinstance(raw, dict) else 0,
        "metrics": raw.get("metrics", {}) if isinstance(raw, dict) else {},
        "num_loaded": loaded,
        "num_model_keys": total,
    }
