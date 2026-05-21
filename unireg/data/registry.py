"""
unireg/data/registry.py
======================
Registry for datasets.

Adding a new dataset
--------------------
1. Add its profile in configs/datasets/datasets.yaml (paths, norm, pairing)
2. Create unireg/data/datasets/my_dataset.py with a builder function:

    from unireg.data.registry import register_dataset
    from unireg.data.dataset import (
        Subject, RandomPairDataset, PairwiseDataset, _scan_nifti_subjects,
    )

    def _scan(img_dir, seg_dir=None, **kw):
        # return List[Subject] or List[Pair]
        ...

    @register_dataset("my_dataset")
    def _build(paths, compute_size, norm_key, mode,
               return_native=False, num_repeats=10, seed=42, **kw):
        subjects = _scan(paths["img_dir"], paths.get("seg_dir"))
        if mode == "random":
            return RandomPairDataset(subjects, compute_size, norm_key,
                                     num_repeats=num_repeats, seed=seed)
        return PairwiseDataset(subjects, compute_size, norm_key,
                               return_native=return_native)

3. Add one import line in unireg/data/__init__.py:
    import unireg.data.datasets.my_dataset

4. Set dataset: my_dataset in your YAML config. Done.
"""

from __future__ import annotations
from typing import Callable, Dict, List

_DATASET_REGISTRY: Dict[str, Callable] = {}


def register_dataset(name: str):
    """Decorator: register a dataset builder.

    Builder signature:
        (paths, compute_size, norm_key, mode,
         return_native=False, num_repeats=10, seed=42, **kw) -> Dataset
    """
    def decorator(fn):
        _DATASET_REGISTRY[name.lower()] = fn
        return fn
    return decorator


def registered_datasets() -> List[str]:
    return sorted(_DATASET_REGISTRY.keys())
