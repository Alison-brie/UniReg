#!/usr/bin/env python3
"""Prepare a compact checkpoint for public release.

The script keeps model weights and lightweight metadata, and removes optimizer
states to reduce file size. It does not change tensor names, so legacy
checkpoints trained with arch=dyn_rpnet remain loadable with arch=unireg_rpn
as long as the underlying model implementation is unchanged.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a UniReg checkpoint for release.")
    parser.add_argument("--input", required=True, help="Input training checkpoint, e.g. logs/.../best.pth")
    parser.add_argument("--output", required=True, help="Output checkpoint, e.g. model_zoo/unireg_rpn_6task.pth")
    parser.add_argument("--arch", default="unireg_rpn", help="Released architecture name stored as metadata.")
    parser.add_argument("--name", default="UniReg-RPN 6-task", help="Model name stored as metadata.")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        raise FileNotFoundError(f"Input checkpoint not found: {in_path}")

    raw = torch.load(str(in_path), map_location="cpu")
    state_dict = (
        raw.get("model_state_dict")
        or raw.get("state_dict")
        or raw.get("model")
        or raw
    )

    payload = {
        "model_state_dict": state_dict,
        "arch": args.arch,
        "name": args.name,
        "step": raw.get("step", 0) if isinstance(raw, dict) else 0,
        "epoch": raw.get("epoch", 0) if isinstance(raw, dict) else 0,
        "metrics": raw.get("metrics", {}) if isinstance(raw, dict) else {},
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(out_path))
    print(f"Saved release checkpoint to: {out_path}")
    print(f"Number of tensors: {len(state_dict)}")
    print("Optimizer state was not included.")


if __name__ == "__main__":
    main()
