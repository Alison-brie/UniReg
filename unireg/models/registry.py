"""Model registry for UniReg."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import torch.nn as nn

_MODEL_REGISTRY: Dict[str, Callable] = {}
_BUILTINS_LOADED: bool = False


def register_model(name: str):
    def decorator(fn):
        _MODEL_REGISTRY[name.lower()] = fn
        return fn

    return decorator


def registered_models() -> List[str]:
    _ensure_builtins()
    return sorted(_MODEL_REGISTRY.keys())


# def _make_baseline_factory(cls):
#     def factory(cfg: dict, device: str = "cuda") -> nn.Module:
#         return cls(
#             tuple(cfg["compute_size"]),
#             first_channel=cfg.get("first_channel", 8),
#             n_steps=cfg.get("n_steps", 2),
#             shared_encoder=cfg.get("shared_encoder", True),
#         ).to(device)

#     return factory

def _make_baseline_factory(cls):
    def factory(cfg: dict, device: str = "cuda") -> nn.Module:
        kwargs = dict(cfg)
        compute_size = tuple(kwargs.pop("compute_size"))

        return cls(
            compute_size,
            **kwargs,
        ).to(device)

    return factory


def _ensure_builtins():
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True

    from unireg.baselines.registry import _REGISTRY as _BASELINE_REGISTRY

    for name, cls in _BASELINE_REGISTRY.items():
        _MODEL_REGISTRY.setdefault(name, _make_baseline_factory(cls))


# def build_model(cfg: Dict[str, Any], device: str = "cuda") -> nn.Module:
#     _ensure_builtins()
#     arch = cfg.get("arch", "rpnet").lower()
#     if arch not in _MODEL_REGISTRY:
#         raise ValueError(
#             f"Unknown arch '{arch}'.\n"
#             f"Available models: {sorted(_MODEL_REGISTRY.keys())}"
#         )
#     return _MODEL_REGISTRY[arch](cfg, device)

def build_model(cfg: Dict[str, Any], device: str = "cuda") -> nn.Module:
    _ensure_builtins()
    arch = cfg.get("arch", "rpnet").lower()
    if arch not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown arch '{arch}'.\n"
            f"Available models: {sorted(_MODEL_REGISTRY.keys())}"
        )

    model = _MODEL_REGISTRY[arch](cfg, device)

    # Wrap once when either the global config or at least one multi-task
    # sub-task explicitly enables pre-align. This lets joint configs keep
    # global use_pre_align=false while enabling it only for e.g. Head/Chest.
    use_pre_align = bool(cfg.get("use_pre_align", False)) or any(
        bool(t.get("use_pre_align", False)) for t in cfg.get("tasks", [])
    )
    if use_pre_align:
        from unireg.core.prealign import PreAlignWrapper
        model = PreAlignWrapper(model)

    return model.to(device)