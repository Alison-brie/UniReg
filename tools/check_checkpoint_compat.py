#!/usr/bin/env python3
"""Check whether a released/legacy checkpoint can be loaded by a UniReg config."""
from __future__ import annotations

import argparse
from pathlib import Path

import sys
# from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from train import _expand_env
from unireg.models.registry import build_model
from unireg.engine.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Check UniReg checkpoint compatibility.")
    parser.add_argument("--config", required=True, help="YAML config used to build the model.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path, e.g. best.pth.")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0.")
    parser.add_argument("--min_ratio", type=float, default=0.98,
                        help="Minimum acceptable ratio of matched parameters.")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    ckpt_path = Path(args.checkpoint)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    with cfg_path.open("r") as f:
        cfg = _expand_env(yaml.safe_load(f))

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.")
        device = "cpu"

    model = build_model(cfg, device=device)
    info = load_checkpoint(str(ckpt_path), model, device=device, strict=False)
    ratio = info["num_loaded"] / max(1, info["num_model_keys"])
    print(f"Matched parameter tensors: {info['num_loaded']}/{info['num_model_keys']} ({ratio:.2%})")

    if ratio < args.min_ratio:
        raise SystemExit(
            f"Compatibility check failed: matched ratio {ratio:.2%} < {args.min_ratio:.2%}."
        )
    print("Checkpoint compatibility check passed.")


if __name__ == "__main__":
    main()
