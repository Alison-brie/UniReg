"""
unireg/data/loaders.py
====================
NIfTI volume loading utilities.
Returns (volume_np, affine, spacing_xyz) where:
    volume_np  : float32 numpy array (D, H, W)
    affine     : (4, 4) numpy array
    spacing_xyz: (sx, sy, sz) mm/voxel tuple in NIfTI world axis order (x, y, z)

IMPORTANT CONVENTION
--------------------
Nibabel loads arrays in (X, Y, Z) voxel index order, where:
  X ~ left-right   (W),  Y ~ posterior-anterior (H),  Z ~ inferior-superior (D)

This project uses PyTorch tensors in (D, H, W) order for 3D ops.
Therefore we transpose NIfTI arrays from (X, Y, Z) -> (Z, Y, X) before returning.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Tuple, Union

try:
    import nibabel as nib
except ImportError:
    raise ImportError("nibabel is required: pip install nibabel")


def load_nifti(
    path: Union[str, Path],
    dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    """
    Load a NIfTI volume.

    Args:
        path  : path to .nii or .nii.gz file
        dtype : output numpy dtype (default float32)

    Returns:
        vol      : (D, H, W) float32 array (Z, Y, X)
        affine   : (4, 4) float64 array
        spacing  : (sx, sy, sz) mm/voxel in NIfTI axis order (x, y, z)
    """
    img = nib.load(str(path))
    vol_xyz = img.get_fdata(dtype=np.float32).astype(dtype)  # (X, Y, Z)
    affine = img.affine.astype(np.float64)

    # nibabel header zooms: (x, y, z) = (W, H, D) typically
    header = img.header
    zooms  = header.get_zooms()[:3]
    spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))

    # Convert (X, Y, Z) -> (Z, Y, X) to match (D, H, W)
    vol_dhw = np.transpose(vol_xyz, (2, 1, 0)).copy()
    return vol_dhw, affine, spacing


def load_seg(
    path: Union[str, Path],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a segmentation NIfTI (integer labels).

    Returns:
        seg    : (D, H, W) int32 array (Z, Y, X)
        affine : (4, 4) float64
    """
    img = nib.load(str(path))
    seg_xyz = np.round(img.get_fdata()).astype(np.int32)  # (X, Y, Z)
    seg = np.transpose(seg_xyz, (2, 1, 0)).copy()
    return seg, img.affine.astype(np.float64)


def save_nifti(
    vol: np.ndarray,
    affine: np.ndarray,
    path: Union[str, Path],
    dtype: np.dtype = np.float32,
):
    """Save a numpy volume as NIfTI."""
    img = nib.Nifti1Image(vol.astype(dtype), affine)
    nib.save(img, str(path))
