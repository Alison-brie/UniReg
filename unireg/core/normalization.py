"""Normalization re-export shim for UniReg."""
from unireg.data.dataset import apply_norm, NORM_REGISTRY, _DEFAULT_NORM as DEFAULT_NORM
__all__ = ["apply_norm", "NORM_REGISTRY", "DEFAULT_NORM"]
