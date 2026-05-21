"""Unified dataset entry point for UniReg.

Supported datasets: chest_unified, abdomen_unified, head_unified, liver_unified, oasis, acdc, and dirqa.
"""

from __future__ import annotations

import os
import glob
import random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from torch.utils.data import Dataset

import nibabel as nib
import yaml
import json
import csv
from unireg.data.loaders import load_nifti as _load_nifti_dhw
from unireg.data.loaders import load_seg as _load_seg_dhw

Shape3D = Tuple[int, int, int]


def _expand_path(p):
    if p is None:
        return None
    return os.path.expanduser(os.path.expandvars(str(p)))


def _load_prealign_phi(prealign_dir: str, moving_id: str, fixed_id: str) -> Optional[torch.Tensor]:
    if prealign_dir is None:
        return None

    path = os.path.join(prealign_dir, f"{moving_id}_{fixed_id}_coarse_phi.npy")
    if not os.path.exists(path):
        # Missing direction; do not infer the reverse direction automatically.
        return None

    phi = np.load(path).astype(np.float32)  # [3, D, H, W]
    return torch.from_numpy(phi)


def _load_keypoints_csv(path: str) -> np.ndarray:
    """Load keypoints from CSV (N, 3) in voxel (x,y,z) = (W, H, D)."""
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    return data[:, :3]


def _load_landmarks_txt(path: str) -> np.ndarray:
    """
    Load landmarks from DIR-QA text file.
    Expected format: whitespace-separated columns, first 3 columns are (x, y, z) in voxel coords (W, H, D).
    """
    # DIR-QA landmarks are comma-separated x,y,z coordinates
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]
    return data[:, :3]


# ════════════════════════════════════════════════════════════════════════════════
# 0. DATASET PROFILES (configs/datasets/datasets.yaml)
# ════════════════════════════════════════════════════════════════════════════════

_DATASET_PROFILES: Dict[str, Any] = {}
_DATASET_PROFILES_LOADED: bool = False


def _load_dataset_profiles() -> Dict[str, Any]:
    """Load dataset profiles from the public config directory.

    Dataset profiles are stored in ``configs/datasets/datasets.yaml``.
    YAML parsing errors are raised explicitly instead of being silently
    swallowed; otherwise users may see a misleading "No profile" error even
    when the profile file exists.
    """
    global _DATASET_PROFILES_LOADED, _DATASET_PROFILES
    if _DATASET_PROFILES_LOADED:
        return _DATASET_PROFILES

    root = Path(__file__).resolve().parents[2]
    cfg_path = root / "configs" / "datasets" / "datasets.yaml"

    if not cfg_path.exists():
        _DATASET_PROFILES = {}
    else:
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                raw = f.read()
            data = yaml.safe_load(raw) or {}
        except Exception as e:
            raise RuntimeError(
                f"Failed to parse dataset profile file: {cfg_path}. "
                "Please check YAML syntax and quote environment-variable paths "
                "such as '${DATA_ROOT}/Chest'."
            ) from e

        if not isinstance(data, dict):
            raise RuntimeError(f"Dataset profile file must contain a mapping: {cfg_path}")
        _DATASET_PROFILES = {str(k).lower(): v for k, v in data.items()}

    _DATASET_PROFILES_LOADED = True
    return _DATASET_PROFILES


def _get_dataset_profile(dataset: str) -> Optional[Dict[str, Any]]:
    profiles = _load_dataset_profiles()
    return profiles.get(dataset.lower())


def get_dataset_profile(dataset: str) -> Optional[Dict[str, Any]]:
    """Public accessor for dataset profile (used by train.py etc.)."""
    return _get_dataset_profile(dataset)


# ════════════════════════════════════════════════════════════════════════════════
# 1. NORMALIZATION
# ════════════════════════════════════════════════════════════════════════════════

def _ct_abdomen(x: np.ndarray) -> np.ndarray:
    lo, hi = -150.0, 250.0
    return (np.clip(x, lo, hi) - lo) / (hi - lo)



def _dirqa_ct(x: np.ndarray) -> np.ndarray:
    return _ct_abdomen(x - 1000.0)


def _mri_minmax(x: np.ndarray, use_mask: bool = True) -> np.ndarray:
    """Percentile [0.5, 99.5] → [0,1]. use_mask=True: estimate percentiles on foreground voxels"""
    x = x.astype(np.float32)
    if use_mask:
        m = x > 0
        vals = x[m]
        if vals.size < 16:
            return np.zeros_like(x, dtype=np.float32)
        p_lo = float(np.percentile(vals, 0.5))
        p_hi = float(np.percentile(vals, 99.5))
        y = np.clip(x, p_lo, p_hi)
        minv = float(y[m].min())
        maxv = float(y[m].max())
    else:
        p_lo = float(np.percentile(x, 0.5))
        p_hi = float(np.percentile(x, 99.5))
        y = np.clip(x, p_lo, p_hi)
        minv = float(y.min())
        maxv = float(y.max())
    scale = maxv - minv
    out = (y - minv) / (scale + 1e-8)
    if scale < 1e-8:
        out = np.zeros_like(x, dtype=np.float32)
    return out.astype(np.float32)


def _zscore_local(x: np.ndarray) -> np.ndarray:
    p_lo = float(np.percentile(x, 0.5))
    p_hi = float(np.percentile(x, 99.5))
    y = np.clip(x, p_lo, p_hi)
    mean = float(y.mean())
    std = float(y.std())
    return (y - mean) / (std + 1e-8)


NORM_REGISTRY: Dict[str, Any] = {
    "ct_abdomen":       _ct_abdomen,
    "dirqa_ct":       _dirqa_ct,
    "mri_percentile": _mri_minmax,
    "sat_zscore":     _zscore_local,
    "stu_zscore":     _zscore_local,
    "none":           lambda x: x,
}

_DEFAULT_NORM: Dict[str, str] = {
    "chest_unified": "ct_abdomen",
    "abdomen_unified": "ct_abdomen",
    "head_unified": "ct_abdomen",
    "liver_unified": "ct_abdomen",
    "oasis": "mri_percentile",
    "acdc": "mri_percentile",
    "dirqa": "dirqa_ct",
}


def apply_norm(vol: np.ndarray, strategy: str) -> np.ndarray:
    if strategy not in NORM_REGISTRY:
        raise ValueError(f"Unknown norm '{strategy}'. Available: {sorted(NORM_REGISTRY)}")
    return NORM_REGISTRY[strategy](vol.astype(np.float32, copy=False)).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════════
# 2. PATH RESOLUTION (fully profile-driven)
# ════════════════════════════════════════════════════════════════════════════════

def _resolve_paths(
    dataset: str,
    split: str,
    img_dir: Optional[str] = None,
    seg_dir: Optional[str] = None,
    native_img_dir: Optional[str] = None,
    native_seg_dir: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """
    Resolve all data paths for (dataset, split) from datasets.yaml profile.

    Priority:
        explicit arguments > profile.splits > profile templates > ValueError.

    Returns dict with keys:
        img_dir, seg_dir, kpt_dir, pair_csv, native_img_dir, native_seg_dir

    For unified pair-based datasets:
        img_dir  = dataset root
        pair_csv = root/pairs/{split}.csv or explicitly defined pair_csv
    """
    # Manual overrides always win
    if img_dir:
        return {
            "img_dir": _expand_path(img_dir),
            "seg_dir": _expand_path(seg_dir),
            "kpt_dir": None,
            "pair_csv": None,
            "native_img_dir": _expand_path(native_img_dir),
            "native_seg_dir": _expand_path(native_seg_dir),
        }

    profile = _get_dataset_profile(dataset)
    if profile is None:
        raise ValueError(
            f"No profile in configs/datasets/datasets.yaml for '{dataset}'. "
            f"Either add a profile or pass explicit img_dir."
        )

    # ── Mode 1: explicit per-split paths, including unified pair-based datasets ──
    splits_cfg = profile.get("splits")
    if splits_cfg and split in splits_cfg:
        sp = splits_cfg[split]

        root = _expand_path(sp.get("root"))
        pair_csv = _expand_path(sp.get("pair_csv"))

        # Unified dataset shortcut:
        # if root is given but pair_csv is omitted, infer root/pairs/{split}.csv
        if root is not None and pair_csv is None:
            pair_csv = os.path.join(root, "pairs", f"{split}.csv")

        return {
            "img_dir":         _expand_path(sp.get("img")) or root,
            "seg_dir":         _expand_path(sp.get("seg")) or _expand_path(native_seg_dir),
            "kpt_dir":         _expand_path(sp.get("kpt")),
            "pair_csv":        pair_csv,
            "native_img_dir":  _expand_path(native_img_dir) or _expand_path(sp.get("raw_img")),
            "native_seg_dir":  _expand_path(native_seg_dir) or _expand_path(sp.get("raw_seg")),
        }

    # Fallback: template-based paths
    split_names = profile.get("split_names", {})
    actual_split = split_names.get(split, split)

    def _t(template: Optional[str]) -> Optional[str]:
        if template is None:
            return None
        return _expand_path(template.replace("{split}", actual_split))

    raw = profile.get("raw") or {}
    processed = profile.get("processed")

    if processed and isinstance(processed, dict):
        # Processed data exists → use for compute, raw for native eval
        result = {
            "img_dir":         _t(processed.get("img")),
            "seg_dir":         _t(processed.get("seg")),
            "kpt_dir":         _t(raw.get("kpt")),
            "pair_csv":        _t(processed.get("pair_csv")) or _t(raw.get("pair_csv")),
            "native_img_dir":  native_img_dir or _t(raw.get("img")),
            "native_seg_dir":  native_seg_dir or _t(raw.get("seg")),
        }
    else:
        # No separate processed → raw IS compute, no native distinction
        result = {
            "img_dir":         _t(raw.get("img")),
            "seg_dir":         _t(raw.get("seg")),
            "kpt_dir":         _t(raw.get("kpt")),
            "pair_csv":        _t(raw.get("pair_csv")),
            "native_img_dir":  native_img_dir,
            "native_seg_dir":  native_seg_dir,
        }

    return result


# ════════════════════════════════════════════════════════════════════════════════
# 3. SUBJECT / PAIR data classes
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class Subject:
    img_path: str
    seg_path: Optional[str] = None
    meta: Dict = field(default_factory=dict)


@dataclass
class Pair:
    moving: Subject
    fixed:  Subject


# ════════════════════════════════════════════════════════════════════════════════
# 4. DIRECTORY SCANNERS
# ════════════════════════════════════════════════════════════════════════════════

def _scan_nifti_subjects(
    img_dir: str,
    seg_dir: Optional[str] = None,
    native_img_dir: Optional[str] = None,
    native_seg_dir: Optional[str] = None,
) -> List[Subject]:
    """
    NIfTI folder structure:
      <img_dir>/case_0001_0000.nii.gz  (or train_000_0000.nii.gz)
      <seg_dir>/case_0001.nii.gz       (or train_000.nii.gz)
    """
    subjects = []
    for ip in sorted(glob.glob(os.path.join(img_dir, "*.nii.gz"))):
        fname   = os.path.basename(ip)
        case_id = fname.replace("_0000.nii.gz", "").replace(".nii.gz", "")
        sp = None
        if seg_dir:
            for seg_candidate_name in [case_id + ".nii.gz",
                                       case_id + "_0000.nii.gz"]:
                candidate = os.path.join(seg_dir, seg_candidate_name)
                if os.path.exists(candidate):
                    sp = candidate
                    break
        meta: Dict[str, Any] = {"case_id": case_id}
        if native_img_dir:
            meta["native_img_path"] = os.path.join(native_img_dir, fname)
        if native_seg_dir:
            for seg_candidate_name in [case_id + ".nii.gz",
                                       case_id + "_0000.nii.gz"]:
                candidate = os.path.join(native_seg_dir, seg_candidate_name)
                if os.path.exists(candidate):
                    meta["native_seg_path"] = candidate
                    break
        subjects.append(Subject(img_path=ip, seg_path=sp, meta=meta))
    return subjects


def _scan_oasis_official_pairs(
    img_dir: str,
    seg_dir: Optional[str] = None,
    split: str = "val",
) -> List[Pair]:
    """
    OASIS Learn2Reg official registration pairs.

    For current training stage, we only use registration_val because
    registration_test has no public labels.

    Direction follows OASIS_dataset.json:
        fixed  = Patient A
        moving = Patient B

    Returned sample is:
        moving -> fixed
    """
    root = Path(img_dir).resolve().parent

    json_candidates = [
        root / "OASIS_dataset.json",
        root / "dataset.json",
    ]

    json_path = None
    for p in json_candidates:
        if p.exists():
            json_path = p
            break

    if json_path is None:
        raise FileNotFoundError(
            f"Cannot find OASIS_dataset.json or dataset.json under {root}"
        )

    with open(json_path, "r") as f:
        info = json.load(f)

    if split == "val":
        pair_key = "registration_val"
    else:
        raise ValueError(
            f"OASIS split '{split}' is not enabled because test labels are unavailable. "
            f"Use split='val' only for now."
        )

    if pair_key not in info:
        raise KeyError(f"'{pair_key}' not found in {json_path}")

    def _resolve_img(rel_path: str) -> str:
        rel_path = rel_path.replace("./", "")
        p = root / rel_path
        if not p.exists():
            raise FileNotFoundError(f"OASIS image not found: {p}")
        return str(p)

    def _resolve_seg(img_path: str) -> Optional[str]:
        if seg_dir is None:
            return None

        fname = os.path.basename(img_path)
        case_id = fname.replace("_0000.nii.gz", "").replace(".nii.gz", "")

        candidates = [
            os.path.join(seg_dir, fname),
            os.path.join(seg_dir, case_id + "_0000.nii.gz"),
            os.path.join(seg_dir, case_id + ".nii.gz"),
        ]

        for c in candidates:
            if os.path.exists(c):
                return c

        return None

    pairs: List[Pair] = []

    for item in info[pair_key]:
        fixed_img = _resolve_img(item["fixed"])
        moving_img = _resolve_img(item["moving"])

        fixed_seg = _resolve_seg(fixed_img)
        moving_seg = _resolve_seg(moving_img)

        fixed_id = os.path.basename(fixed_img).replace(".nii.gz", "")
        moving_id = os.path.basename(moving_img).replace(".nii.gz", "")

        pairs.append(
            Pair(
                moving=Subject(
                    img_path=moving_img,
                    seg_path=moving_seg,
                    meta={
                        "case_id": f"{moving_id}_to_{fixed_id}",
                        "subject_id": moving_id,
                        "role": "moving",
                    },
                ),
                fixed=Subject(
                    img_path=fixed_img,
                    seg_path=fixed_seg,
                    meta={
                        "case_id": f"{moving_id}_to_{fixed_id}",
                        "subject_id": fixed_id,
                        "role": "fixed",
                    },
                ),
            )
        )

    if len(pairs) == 0:
        raise RuntimeError(f"No OASIS {split} pairs found from {json_path}")

    return pairs


def _resolve_json_path(path_value: Optional[str], base_dir: Path) -> Optional[str]:
    """Resolve an absolute/relative path stored in a pair JSON file."""
    if path_value is None:
        return None
    path_value = str(path_value).strip()
    if path_value == "":
        return None
    pp = Path(path_value)
    if pp.is_absolute():
        return str(pp)
    return str(base_dir / path_value.replace("./", ""))


def _scan_oasis_json_pairs(pair_json: str, split: str) -> List[Pair]:
    """
    Read OASIS/LUMIR-style paired registration pairs from json.

    Supported split keys, checked in order:
      train: training_paired_images, training, train
      val:   registration_val, validation, val
      test:  registration_test, test, testing

    Each item should contain at least:
      moving, fixed

    Optional label keys:
      moving_label / moving_seg / moving_mask / moving_label_path
      fixed_label  / fixed_seg  / fixed_mask  / fixed_label_path

    Direction:
      moving -> fixed
    """
    if pair_json is None or not os.path.exists(pair_json):
        raise FileNotFoundError(f"pair_json not found: {pair_json}")

    pair_json_path = Path(pair_json)
    base_dir = pair_json_path.resolve().parent

    with open(pair_json, "r") as f:
        info = json.load(f)

    # Some json files may directly store a list of pairs.
    if isinstance(info, list):
        items = info
        key = "<root-list>"
    else:
        if split == "train":
            candidate_keys = ["training_paired_images", "training", "train"]
        elif split == "val":
            candidate_keys = ["registration_val", "validation", "val"]
        elif split == "test":
            candidate_keys = ["registration_test", "test", "testing"]
        else:
            candidate_keys = [split]

        key = None
        for k in candidate_keys:
            if k in info:
                key = k
                break

        if key is None:
            raise KeyError(
                f"None of keys {candidate_keys} found in {pair_json}. "
                f"Available keys: {list(info.keys())}"
            )
        items = info[key]

    pairs: List[Pair] = []

    def _first_existing_key(item: Dict[str, Any], keys: List[str]) -> Optional[str]:
        for k in keys:
            v = item.get(k)
            if v is not None and str(v).strip() != "":
                return v
        return None

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise TypeError(f"Bad OASIS pair item at index {idx}: expected dict, got {type(item)}")

        fixed_img = _resolve_json_path(_first_existing_key(item, ["fixed", "fixed_img", "fixed_image"]), base_dir)
        moving_img = _resolve_json_path(_first_existing_key(item, ["moving", "moving_img", "moving_image"]), base_dir)

        fixed_seg = _resolve_json_path(
            _first_existing_key(item, ["fixed_label", "fixed_seg", "fixed_mask", "fixed_label_path"]),
            base_dir,
        )
        moving_seg = _resolve_json_path(
            _first_existing_key(item, ["moving_label", "moving_seg", "moving_mask", "moving_label_path"]),
            base_dir,
        )

        if fixed_img is None or moving_img is None:
            raise KeyError(
                f"OASIS pair item {idx} must contain moving/fixed image paths. item={item}"
            )

        for pp in [fixed_img, moving_img]:
            if not os.path.exists(pp):
                raise FileNotFoundError(f"Image not found: {pp}")

        for pp in [fixed_seg, moving_seg]:
            if pp is not None and not os.path.exists(pp):
                raise FileNotFoundError(f"Label not found: {pp}")

        fixed_id = item.get("fixed_id") or os.path.basename(fixed_img).replace(".nii.gz", "")
        moving_id = item.get("moving_id") or os.path.basename(moving_img).replace(".nii.gz", "")
        case_id = item.get("case_id") or item.get("subject_id") or f"{moving_id}_to_{fixed_id}"

        pairs.append(
            Pair(
                moving=Subject(
                    img_path=moving_img,
                    seg_path=moving_seg,
                    meta={
                        "case_id": str(case_id),
                        "subject_id": str(moving_id),
                        "role": "moving",
                        "image_id": str(moving_id),
                    },
                ),
                fixed=Subject(
                    img_path=fixed_img,
                    seg_path=fixed_seg,
                    meta={
                        "case_id": str(case_id),
                        "subject_id": str(fixed_id),
                        "role": "fixed",
                        "image_id": str(fixed_id),
                    },
                ),
            )
        )

    if len(pairs) == 0:
        raise RuntimeError(f"No OASIS {split} pairs found in {pair_json} under key {key}")

    return pairs


def _scan_acdc_json_pairs(pair_json: str, split: str) -> List[Pair]:
    """
    Read ACDC paired ED/ES registration pairs from json.

    train:
      info["training_paired_images"]

    val:
      info["registration_val"]

    test:
      info["registration_test"]

    Direction:
      moving -> fixed
    """
    if pair_json is None or not os.path.exists(pair_json):
        raise FileNotFoundError(f"pair_json not found: {pair_json}")

    with open(pair_json, "r") as f:
        info = json.load(f)

    if split == "train":
        key = "training_paired_images"
    elif split == "val":
        key = "registration_val"
    elif split == "test":
        key = "registration_test"
    else:
        raise ValueError(f"Unsupported ACDC split: {split}")

    if key not in info:
        raise KeyError(f"{key} not found in {pair_json}")

    pairs: List[Pair] = []

    for item in info[key]:
        fixed_img = item["fixed"]
        moving_img = item["moving"]
        fixed_seg = item.get("fixed_label", "") or None
        moving_seg = item.get("moving_label", "") or None

        for p in [fixed_img, moving_img]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Image not found: {p}")

        for p in [fixed_seg, moving_seg]:
            if p is not None and not os.path.exists(p):
                raise FileNotFoundError(f"Label not found: {p}")

        sid = item.get("subject_id", "unknown")
        direction = item.get("direction", "unknown")

        fixed_id = os.path.basename(fixed_img).replace(".nii.gz", "")
        moving_id = os.path.basename(moving_img).replace(".nii.gz", "")

        pairs.append(
            Pair(
                moving=Subject(
                    img_path=moving_img,
                    seg_path=moving_seg,
                    meta={
                        "case_id": f"{sid}_{direction}",
                        "subject_id": sid,
                        "direction": direction,
                        "role": "moving",
                        "image_id": moving_id,
                    },
                ),
                fixed=Subject(
                    img_path=fixed_img,
                    seg_path=fixed_seg,
                    meta={
                        "case_id": f"{sid}_{direction}",
                        "subject_id": sid,
                        "direction": direction,
                        "role": "fixed",
                        "image_id": fixed_id,
                    },
                ),
            )
        )

    if len(pairs) == 0:
        raise RuntimeError(f"No ACDC {split} pairs found in {pair_json}")

    return pairs


def _scan_dirqa(data_dir: str, bidirectional: bool = True) -> List[Dict]:
    """
    DIR-QA structure (pre-processed 192×192×96):
      <data_dir>/case{N}/
          case{N}_img1_192x192x96.nii.gz   (img1)
          case{N}_img2_192x192x96.nii.gz   (img2)
          case{N}_landmarks1_192x192x96.txt (img1 landmarks)
          case{N}_landmarks2_192x192x96.txt (img2 landmarks)

    When bidirectional=True (default, matching FVM-REG.py):
      Fwd: moving=img1, fixed=img2   (30 cases)
      Bwd: moving=img2, fixed=img1   (30 cases)
      Total: 60 entries
    """
    entries = []
    for case_dir in sorted(Path(data_dir).iterdir()):
        if not case_dir.is_dir() or not case_dir.name.startswith("case"):
            continue
        cid = case_dir.name
        img1 = case_dir / f"{cid}_img1_192x192x96.nii.gz"
        img2 = case_dir / f"{cid}_img2_192x192x96.nii.gz"
        lm1  = case_dir / f"{cid}_landmarks1_192x192x96.txt"
        lm2  = case_dir / f"{cid}_landmarks2_192x192x96.txt"
        # Tolerate .nii fallback
        if not img1.exists():
            alt = case_dir / f"{cid}_img1_192x192x96.nii"
            if alt.exists():
                img1 = alt
        if not img2.exists():
            alt = case_dir / f"{cid}_img2_192x192x96.nii"
            if alt.exists():
                img2 = alt
        if not (img1.exists() and img2.exists()):
            continue

        lm1_s = str(lm1) if lm1.exists() else None
        lm2_s = str(lm2) if lm2.exists() else None

        # Fwd: mov=img1 → fix=img2
        entries.append({
            "moving":           str(img1),
            "fixed":            str(img2),
            "landmarks_moving": lm1_s,
            "landmarks_fixed":  lm2_s,
            "case_id":          f"{cid}_Fwd",
        })
        if bidirectional:
            # Bwd: mov=img2 → fix=img1
            entries.append({
                "moving":           str(img2),
                "fixed":            str(img1),
                "landmarks_moving": lm2_s,
                "landmarks_fixed":  lm1_s,
                "case_id":          f"{cid}_Bwd",
            })
    return entries


# ════════════════════════════════════════════════════════════════════════════════
# 5. PYTORCH DATASETS
# ════════════════════════════════════════════════════════════════════════════════

class _Base(Dataset):
    """Shared volume loading with norm + resize."""

    def __init__(self, compute_size: Shape3D, norm: str, return_native: bool = False, prealign_dir: Optional[str] = None):
        self.compute_size = compute_size
        self.norm         = norm
        self.return_native = return_native
        self.prealign_dir = prealign_dir

    def _load_img(self, path: str) -> torch.Tensor:
        vol, _, _ = _load_nifti_dhw(path)
        vol = apply_norm(vol, self.norm)
        t   = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)
        t   = F.interpolate(t, size=self.compute_size, mode="trilinear", align_corners=True)
        return t.squeeze(0)   # [1,D,H,W]

    def _load_seg(self, path: Optional[str]) -> Optional[torch.Tensor]:
        if path is None:
            return None
        seg, _ = _load_seg_dhw(path)
        t = torch.from_numpy(seg).float().unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=self.compute_size, mode="nearest")
        return t.long().squeeze(0)   # [1,D,H,W]

    def _load_mask(self, path: Optional[str]) -> Optional[torch.Tensor]:
        """Load a binary mask and resize to compute_size.

        Returned shape: [1, D, H, W], dtype=float32, values in {0,1}.
        Used only for pre-align samples/tasks that explicitly enable
        use_pre_align. Non-prealign tasks are unaffected.
        """
        if path is None:
            return None
        if not os.path.exists(path):
            return None
        mask, _ = _load_seg_dhw(path)
        mask = (mask > 0).astype(np.float32)
        t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=self.compute_size, mode="nearest")
        return (t > 0.5).float().squeeze(0)  # [1,D,H,W]

    def _make(self, mv_subj: Subject, fx_subj: Subject) -> Dict:
        mv_img = self._load_img(mv_subj.img_path)
        fx_img = self._load_img(fx_subj.img_path)
        mv_seg = self._load_seg(mv_subj.seg_path)
        fx_seg = self._load_seg(fx_subj.seg_path)

        out: Dict[str, Any] = {
            "moving": mv_img,
            "fixed":  fx_img,
            "meta":   {"moving": mv_subj.meta, "fixed": fx_subj.meta},
        }
        if mv_seg is not None:
            out["moving_seg"] = mv_seg
        if fx_seg is not None:
            out["fixed_seg"] = fx_seg
        if self.return_native:
            mv_native_seg = mv_subj.meta.get("native_seg_path") or mv_subj.seg_path
            fx_native_seg = fx_subj.meta.get("native_seg_path") or fx_subj.seg_path
            fx_native_img = fx_subj.meta.get("native_img_path") or fx_subj.img_path

            if mv_native_seg:
                raw_ms, _ = _load_seg_dhw(mv_native_seg)
                out["moving_seg_raw"] = raw_ms
            if fx_native_seg:
                raw_fs, _ = _load_seg_dhw(fx_native_seg)
                out["fixed_seg_raw"] = raw_fs

            img = nib.load(fx_native_img)
            out["raw_shape_xyz"] = tuple(int(x) for x in img.shape[:3])
            out["raw_shape"] = (int(img.shape[2]), int(img.shape[1]), int(img.shape[0]))
        
        prealign_dir = getattr(self, "prealign_dir", None)
        if prealign_dir is not None:
            mv_id = mv_subj.meta.get("case_id") or mv_subj.meta.get("subject_id")
            fx_id = fx_subj.meta.get("case_id") or fx_subj.meta.get("subject_id")

            if mv_id is not None and fx_id is not None:
                pre_phi = _load_prealign_phi(prealign_dir, str(mv_id), str(fx_id))
                if pre_phi is not None:
                    out["pre_align"] = pre_phi

        # Landmark-based datasets: load keypoints and scale to compute space for TRE.
        kpt_mov_path = mv_subj.meta.get("kpt_path")
        kpt_fix_path = fx_subj.meta.get("kpt_path")
        if kpt_mov_path and kpt_fix_path and os.path.isfile(kpt_mov_path) and os.path.isfile(kpt_fix_path):
            pts_mov = _load_keypoints_csv(kpt_mov_path)   # (N, 3) voxel (x,y,z) = (W, H, D)
            pts_fix = _load_keypoints_csv(kpt_fix_path)
            # NIfTI shape is (X, Y, Z) = (W, H, D)
            mov_shape = nib.load(mv_subj.img_path).shape
            fix_shape = nib.load(fx_subj.img_path).shape
            D, H, W = self.compute_size[0], self.compute_size[1], self.compute_size[2]
            scale_mov = (W / max(1, mov_shape[0]), H / max(1, mov_shape[1]), D / max(1, mov_shape[2]))
            scale_fix = (W / max(1, fix_shape[0]), H / max(1, fix_shape[1]), D / max(1, fix_shape[2]))
            pts_mov_scaled = pts_mov * np.array(scale_mov, dtype=np.float32)
            pts_fix_scaled = pts_fix * np.array(scale_fix, dtype=np.float32)
            out["pts_mov"] = pts_mov_scaled
            out["pts_fix"] = pts_fix_scaled
            out["shape_xyz"] = (W, H, D)
            out["spacing_xyz"] = (1.5, 1.5, 1.5)  # Default compute-space spacing for landmark TRE.
        return out


class RandomPairDataset(_Base):
    """Training dataset: each __getitem__(i) randomly draws a DIFFERENT subject."""
    def __init__(self, subjects: List[Subject], compute_size: Shape3D,
                 norm: str, num_repeats: int = 10, seed: Optional[int] = None, **kwargs):
        super().__init__(compute_size, norm, **kwargs)
        self.subjects    = subjects
        self.num_repeats = num_repeats
        self._rng        = random.Random(seed)

    def __len__(self): return len(self.subjects) * self.num_repeats

    def __getitem__(self, idx):
        i = idx % len(self.subjects)
        j = i
        while j == i:
            j = self._rng.randrange(len(self.subjects))
        return self._make(self.subjects[i], self.subjects[j])


class PairwiseDataset(_Base):
    """Half-symmetric eval (i < j)."""
    def __init__(self, subjects: List[Subject], compute_size: Shape3D, norm: str, **kwargs):
        super().__init__(compute_size, norm, **kwargs)
        n = len(subjects)
        self.pairs = [(subjects[i], subjects[j])
                      for i in range(n) for j in range(i + 1, n)]

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        mv, fx = self.pairs[idx]
        return self._make(mv, fx)


class AllPairsDataset(_Base):
    """All-pairs eval (i ≠ j, both directions)."""
    def __init__(self, subjects: List[Subject], compute_size: Shape3D, norm: str, **kwargs):
        super().__init__(compute_size, norm, **kwargs)
        n = len(subjects)
        self.pairs = [(subjects[i], subjects[j])
                      for i in range(n) for j in range(n) if i != j]

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        mv, fx = self.pairs[idx]
        return self._make(mv, fx)


class SequentialPairDataset(_Base):
    """Sequential adjacent pairs (i, i+1): 74 subjects → 73 pairs."""
    def __init__(self, subjects: List[Subject], compute_size: Shape3D, norm: str, **kwargs):
        super().__init__(compute_size, norm, **kwargs)
        self.pairs = [(subjects[i], subjects[i + 1])
                      for i in range(len(subjects) - 1)]

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        mv, fx = self.pairs[idx]
        return self._make(mv, fx)


class FixedPairDataset(_Base):
    """Pre-defined moving/fixed evaluation pairs."""
    def __init__(self, pairs: List[Pair], compute_size: Shape3D, norm: str, **kwargs):
        super().__init__(compute_size, norm, **kwargs)
        self.pairs = pairs

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        return self._make(p.moving, p.fixed)


class UnifiedPairDataset(_Base):
    """
    Unified pair-based dataset.

    Expected structure:
      RegDataUnified/Chest/
        images/
        labels/
        masks/
        pre_align/
        pairs/train.csv
        pairs/val.csv
        pairs/test.csv

    CSV columns:
      moving_id,fixed_id,moving_img,fixed_img,moving_label,fixed_label,pre_align
    """

    def __init__(
        self,
        dataset_root: str,
        pair_csv: str,
        compute_size: Shape3D,
        norm: str,
        return_native: bool = False,
        require_pre_align: bool = False,
        load_masks_for_prealign: bool = False,
        **kwargs,
    ):
        super().__init__(compute_size, norm, return_native=return_native)

        self.dataset_root = Path(dataset_root)
        self.pair_csv = Path(pair_csv)
        self.require_pre_align = bool(require_pre_align)
        self.load_masks_for_prealign = bool(load_masks_for_prealign)

        if not self.pair_csv.exists():
            raise FileNotFoundError(f"Missing pair csv: {self.pair_csv}")

        with open(self.pair_csv, "r", newline="") as f:
            self.rows = list(csv.DictReader(f))

        if len(self.rows) == 0:
            raise RuntimeError(f"No rows found in {self.pair_csv}")

    def __len__(self):
        return len(self.rows)

    def _resolve(self, p: str) -> Optional[str]:
        if p is None:
            return None
        p = str(p).strip()
        if p == "":
            return None
        pp = Path(p)
        if pp.is_absolute():
            return str(pp)
        return str(self.dataset_root / pp)

    def _load_pre_align(self, path: Optional[str]) -> Optional[torch.Tensor]:
        if path is None:
            if self.require_pre_align:
                raise FileNotFoundError("pre_align is required but empty in pair csv.")
            return None

        if not os.path.exists(path):
            if self.require_pre_align:
                raise FileNotFoundError(f"Missing pre_align: {path}")
            return None

        phi = np.load(path).astype(np.float32)

        if phi.ndim == 5:
            phi = phi[0]
        if phi.shape[0] != 3:
            raise RuntimeError(f"Bad pre_align shape {phi.shape}, expected [3,D,H,W]. path={path}")

        return torch.from_numpy(phi).float()

    def __getitem__(self, idx):
        row = self.rows[idx]

        moving_id = row.get("moving_id", "").strip()
        fixed_id = row.get("fixed_id", "").strip()

        moving_img = self._resolve(row.get("moving_img", f"images/{moving_id}.nii.gz"))
        fixed_img = self._resolve(row.get("fixed_img", f"images/{fixed_id}.nii.gz"))

        moving_label = self._resolve(row.get("moving_label", f"labels/{moving_id}.nii.gz"))
        fixed_label = self._resolve(row.get("fixed_label", f"labels/{fixed_id}.nii.gz"))

        # Optional masks. Prefer explicit CSV columns, otherwise use
        # masks/{case_id}.nii.gz. They are loaded only for rows with pre_align
        # and only when the task/config explicitly enables use_pre_align.
        moving_mask_path = self._resolve(row.get("moving_mask", f"masks/{moving_id}.nii.gz"))
        fixed_mask_path = self._resolve(row.get("fixed_mask", f"masks/{fixed_id}.nii.gz"))

        pre_align_path = self._resolve(row.get("pre_align", ""))

        mv_meta = {
            "case_id": moving_id,
        }

        # 只有真的存在 pre_align_path 时才写入 meta，避免 DataLoader collate None 报错
        if pre_align_path is not None:
            mv_meta["pre_align_path"] = pre_align_path

        mv = Subject(
            img_path=moving_img,
            seg_path=moving_label,
            meta=mv_meta,
        )

        fx = Subject(
            img_path=fixed_img,
            seg_path=fixed_label,
            meta={
                "case_id": fixed_id,
            },
        )

        out = self._make(mv, fx)
        
        pre_align = self._load_pre_align(pre_align_path)
            
        if pre_align is not None:
            out["pre_align"] = pre_align

            # Only read/return masks for tasks that explicitly enable pre-align.
            if self.load_masks_for_prealign:
                moving_mask = self._load_mask(moving_mask_path)
                fixed_mask = self._load_mask(fixed_mask_path)
                if moving_mask is not None:
                    out["moving_mask"] = moving_mask
                if fixed_mask is not None:
                    out["fixed_mask"] = fixed_mask

        out["dataset"] = row.get("dataset", self.dataset_root.name)
        return out
    
    
class DirectQADataset(_Base):
    """DIR-QA evaluation (external validation): loads landmark points for TRE."""
    def __init__(self, entries: List[Dict], compute_size: Shape3D, norm: str, **kwargs):
        super().__init__(compute_size, norm, **kwargs)
        self.entries = entries

    def __len__(self): return len(self.entries)

    def __getitem__(self, idx):
        e  = self.entries[idx]
        mv = Subject(img_path=e["moving"], meta={"case_id": e["case_id"]})
        fx = Subject(img_path=e["fixed"],  meta={"case_id": e["case_id"]})
        item = self._make(mv, fx)
        lm_fix = e.get("landmarks_fixed")
        lm_mov = e.get("landmarks_moving")
        item["landmarks_fixed"]  = lm_fix
        item["landmarks_moving"] = lm_mov

        # If landmarks exist, attach scaled points + spacing/shape for TRE in compute space
        if lm_fix and lm_mov and os.path.isfile(lm_fix) and os.path.isfile(lm_mov):
            pts_fix = _load_landmarks_txt(lm_fix)  # (N,3) in voxel (x,y,z)
            pts_mov = _load_landmarks_txt(lm_mov)

            # Scale from native voxel grid to compute_size grid (we always interpolate to compute_size)
            mov_shape = nib.load(mv.img_path).shape  # (X,Y,Z)=(W,H,D)
            fix_shape = nib.load(fx.img_path).shape
            D, H, W = self.compute_size[0], self.compute_size[1], self.compute_size[2]
            scale_mov = (W / max(1, mov_shape[0]), H / max(1, mov_shape[1]), D / max(1, mov_shape[2]))
            scale_fix = (W / max(1, fix_shape[0]), H / max(1, fix_shape[1]), D / max(1, fix_shape[2]))
            item["pts_mov"] = pts_mov * np.array(scale_mov, dtype=np.float32)
            item["pts_fix"] = pts_fix * np.array(scale_fix, dtype=np.float32)

            # Use fixed image spacing when available; shape_xyz is compute grid size (W,H,D)
            _, _, spacing_xyz = _load_nifti_dhw(fx.img_path)
            item["spacing_xyz"] = spacing_xyz
            item["shape_xyz"] = (W, H, D)
        return item


# ════════════════════════════════════════════════════════════════════════════════
# 6. PAIRING RESOLUTION
# ════════════════════════════════════════════════════════════════════════════════

def _resolve_pairing(dataset: str, split: str, pair_mode: Optional[str] = None) -> str:
    """
    Determine pairing strategy from: explicit override > profile > default.

    Returns canonical string:
        'random' | 'pairwise' | 'sequential' | 'all_pairs' |
        'fixed' | 'fixed_bidirectional'
    """
    if pair_mode is not None:
        return pair_mode

    profile = _get_dataset_profile(dataset)
    if profile and "pairing" in profile:
        prof_val = (profile["pairing"] or {}).get(split)
        if prof_val:
            _MAP = {
                "random":              "random",
                "pairwise_i_lt_j":     "pairwise",
                "sequential_i_ip1":    "sequential",
                "all_pairs":           "all_pairs",
                "fixed_0000_to_0001":  "fixed",
                "fixed_bidirectional": "fixed_bidirectional",
                "fixed":               "fixed",
            }
            return _MAP.get(prof_val, prof_val)

    return "random" if split == "train" else "pairwise"


def _resolve_norm(dataset: str, norm: Optional[str] = None) -> str:
    """Determine norm strategy from: explicit > profile > default table."""
    if norm is not None:
        return norm
    profile = _get_dataset_profile(dataset)
    if profile and "norm" in profile:
        return str(profile["norm"])
    return _DEFAULT_NORM.get(dataset, "none")


# ════════════════════════════════════════════════════════════════════════════════
# 7. MAIN ENTRY POINT: build_dataset
# ════════════════════════════════════════════════════════════════════════════════

from unireg.data.registry import _DATASET_REGISTRY, register_dataset  # noqa: E402


def build_dataset(
    dataset: str,
    split: str,
    compute_size: Shape3D,
    norm: Optional[str] = None,
    return_native: bool = False,
    # ── path overrides (if not passed, resolved from datasets.yaml) ──
    img_dir:  Optional[str] = None,
    seg_dir:  Optional[str] = None,
    native_img_dir: Optional[str] = None,
    native_seg_dir: Optional[str] = None,
    # ── pairing options ──
    pair_mode: Optional[str] = None,
    num_repeats: int = 10,
    seed: int = 42,
    prealign_dir: Optional[str] = None,
    pair_json: Optional[str] = None,
    use_pre_align: bool = False,
) -> Dataset:
    """
    Build the correct Dataset for a given dataset name and split.

    ALL paths, normalization, and pairing strategies are resolved from
    configs/datasets/datasets.yaml unless explicitly overridden by arguments.

    Custom datasets registered via @register_dataset are found automatically.
    """
    dataset  = dataset.lower()
    split    = split.lower()

    if dataset not in _DATASET_REGISTRY:
        from unireg.data.registry import registered_datasets
        raise ValueError(
            f"Unknown dataset '{dataset}'.\n"
            f"Registered: {registered_datasets()}"
        )

    norm_key = _resolve_norm(dataset, norm)
    mode     = _resolve_pairing(dataset, split, pair_mode)
    paths    = _resolve_paths(dataset, split, img_dir, seg_dir,
                              native_img_dir, native_seg_dir)

    return _DATASET_REGISTRY[dataset](
        paths=paths,
        compute_size=compute_size,
        norm_key=norm_key,
        mode=mode,
        split=split,
        return_native=return_native,
        num_repeats=num_repeats,
        seed=seed,
        prealign_dir=prealign_dir,
        pair_json=pair_json,
        use_pre_align=use_pre_align,
    )


# ════════════════════════════════════════════════════════════════════════════════
# 8. BUILT-IN DATASET REGISTRATIONS
# ════════════════════════════════════════════════════════════════════════════════

@register_dataset("dirqa")
def _build_dirqa(paths, compute_size, norm_key, mode,
                 return_native=False, **_):
    bidir = (mode == "fixed_bidirectional")
    entries = _scan_dirqa(paths["img_dir"], bidirectional=bidir)
    return DirectQADataset(entries, compute_size, norm_key,
                           return_native=return_native)


@register_dataset("chest_unified")
@register_dataset("abdomen_unified")
@register_dataset("head_unified")
@register_dataset("liver_unified")
def _build_chest_unified(paths, compute_size, norm_key, mode,
                         return_native=False, **kwargs):
    root = paths["img_dir"]
    pair_csv = paths.get("pair_csv")
    if pair_csv is None:
        raise RuntimeError("unified dataset requires paths['pair_csv'].")

    # Unified datasets share the same pair-file structure; Liver does not require pre-align.
    root_name = Path(root).name.lower()
    require_pre_align = root_name not in ("liver", "liver_unified")

    return UnifiedPairDataset(
        dataset_root=root,
        pair_csv=pair_csv,
        compute_size=compute_size,
        norm=norm_key,
        return_native=return_native,
        require_pre_align=require_pre_align,
        load_masks_for_prealign=bool(kwargs.get("use_pre_align", False)),
    )

@register_dataset("oasis")
def _build_oasis(
    paths,
    compute_size,
    norm_key,
    mode,
    split="train",
    pair_json=None,
    return_native=False,
    num_repeats=10,
    seed=42,
    **_,
):
    """
    OASIS Learn2Reg task.

    train:
        random unpaired MR-MR registration from imagesTr / labelsTr

    val:
        official registration_val pairs from OASIS_dataset.json
        with labelsTr for Dice evaluation

    test:
        disabled for now because labelsTs are unavailable
    """
    img_dir = paths["img_dir"]
    seg_dir = paths.get("seg_dir")

    # IMPORTANT for LUMIR-train + OASIS-val/test configs:
    # train must use random pairs from the LUMIR image folder even when a
    # pair_json is provided for validation/test.  The old implementation
    # checked pair_json first, so split=train incorrectly tried to read
    # pair_json["training_paired_images"] and crashed when the JSON only
    # contained OASIS val/test pairs.
    if split == "train" and mode == "random":
        print(f"[OASIS] split={split}, mode=random: use folder random pairing; ignore pair_json={pair_json}")
        subjects = _scan_nifti_subjects(img_dir, seg_dir)
        return RandomPairDataset(
            subjects,
            compute_size,
            norm_key,
            num_repeats=num_repeats,
            seed=seed,
        )

    # For val/test, prefer explicit JSON pairs when provided.
    # This supports LUMIR_train_OASIS_val_test500.json and similar files.
    if pair_json is not None:
        print(f"[OASIS] split={split}, mode={mode}: use fixed pairs from pair_json={pair_json}")
        pairs = _scan_oasis_json_pairs(
            pair_json=pair_json,
            split=split,
        )
        return FixedPairDataset(
            pairs,
            compute_size,
            norm_key,
            return_native=return_native,
        )

    if mode == "random":
        subjects = _scan_nifti_subjects(img_dir, seg_dir)
        return RandomPairDataset(
            subjects,
            compute_size,
            norm_key,
            num_repeats=num_repeats,
            seed=seed,
        )

    if mode in ("oasis_val", "official_val", "registration_val"):
        pairs = _scan_oasis_official_pairs(
            img_dir=img_dir,
            seg_dir=seg_dir,
            split="val",
        )
        return FixedPairDataset(
            pairs,
            compute_size,
            norm_key,
            return_native=return_native,
        )

    raise ValueError(
        f"Unknown OASIS pairing mode: {mode}. "
        f"Use mode='random' for train or mode='oasis_val' for val."
    )
    

@register_dataset("acdc")
def _build_acdc(
    paths,
    compute_size,
    norm_key,
    mode,
    split="train",
    pair_json=None,
    return_native=False,
    num_repeats=10,
    seed=42,
    **_,
):
    """
    ACDC ED/ES bidirectional intra-subject registration.

    This dataset is naturally paired:
      ED -> ES
      ES -> ED

    Therefore train/val/test all use FixedPairDataset.
    """
    if pair_json is None:
        raise ValueError("ACDC requires pair_json. Please set pair_json in config.")

    pairs = _scan_acdc_json_pairs(
        pair_json=pair_json,
        split=split,
    )

    return FixedPairDataset(
        pairs,
        compute_size,
        norm_key,
        return_native=return_native,
    )
    