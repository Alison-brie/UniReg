#!/usr/bin/env python3
"""Smoke checks for the clean UniReg repository."""
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))


for forbidden_dir in ["__MACOSX", "eval_output"]:
    hits = [p for p in root.rglob(forbidden_dir) if p.is_dir()]
    assert not hits, f"Forbidden directory still exists: {hits}"

for forbidden_file in [".DS_Store"]:
    hits = list(root.rglob(forbidden_file))
    assert not hits, f"Forbidden file still exists: {hits}"

assert not (root / "configs" / "datasets.yaml").exists(), "Use only configs/datasets/datasets.yaml"

for forbidden in ["ran" + "_pytorch.py", "trans" + "morph.py", "voxel" + "morph_unet.py", "affine" + "_prealign.py"]:
    hits = list(root.rglob(forbidden))
    assert not hits, f"Forbidden file still exists: {hits}"

from unireg.models.registry import registered_models
models = set(registered_models())
expected = {"unireg_rpn", "unireg_iirpn", "unireg_mlp", "rpnet", "iirpnet", "corrmlp"}
assert models == expected, f"Unexpected models: {sorted(models)}"

dataset_text = (root / "unireg/data/dataset.py").read_text()
assert ("amos" + "_ct") not in dataset_text
assert "ct_abdomen" in dataset_text
assert "mri_percentile" in dataset_text

print("Clean-repo smoke checks passed.")
