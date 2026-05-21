#!/usr/bin/env python3
"""Load key YAML configs, dataset profiles, and build configured models on CPU."""
from pathlib import Path
import sys
import yaml

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from unireg.models.registry import build_model
from unireg.data.dataset import get_dataset_profile

required_profiles = [
    "chest_unified",
    "abdomen_unified",
    "head_unified",
    "liver_unified",
    "oasis",
    "acdc",
    "dirqa",
]
for name in required_profiles:
    profile = get_dataset_profile(name)
    if profile is None:
        raise SystemExit(f"Missing dataset profile: {name}")
print(f"Loaded {len(required_profiles)} dataset profiles.")

config_paths = [
    root / "configs/train/single_task/lumir_rpnet.yaml",
    root / "configs/train/multi_task/unireg_rpn_6task_from_brain.yaml",
    root / "configs/eval/eval_brain_unireg_rpn.yaml",
    root / "configs/eval/eval_liver_unireg_rpn.yaml",
]

for p in config_paths:
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = dict(cfg)
    # Use a small but valid spatial size for model-construction checks only.
    cfg["compute_size"] = [32, 32, 32]
    cfg["device"] = "cpu"
    model = build_model(cfg, device="cpu")
    print(f"built {cfg['arch']} from {p.relative_to(root)}")

print("Config/model smoke checks passed.")
