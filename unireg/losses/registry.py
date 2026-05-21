"""
unireg/losses/registry.py
========================
Registry for similarity and regularisation losses.

Adding a new loss
-----------------
1. Create your loss class in a new file, e.g. unireg/losses/my_loss.py
2. At the bottom of that file:

    from unireg.losses.registry import register_sim_loss

    @register_sim_loss("my_loss")
    def _build(cfg: dict):
        return MyLoss(param=cfg.get("my_param", 1.0))

3. Add one import line in unireg/losses/__init__.py:
    import unireg.losses.my_loss

4. Set sim_loss: my_loss in your YAML config. Done.
"""

from __future__ import annotations
from typing import Callable, Dict, Any

_SIM_LOSS_REGISTRY: Dict[str, Callable[[dict], Any]] = {}
_REG_LOSS_REGISTRY: Dict[str, Callable[[dict], Any]] = {}
_BUILTINS_LOADED: bool = False


def register_sim_loss(name: str):
    """Decorator: register a similarity loss factory.  Signature: (cfg) -> nn.Module."""
    def decorator(fn):
        _SIM_LOSS_REGISTRY[name.lower()] = fn
        return fn
    return decorator


def register_reg_loss(name: str):
    """Decorator: register a regularisation loss factory.  Signature: (cfg) -> nn.Module."""
    def decorator(fn):
        _REG_LOSS_REGISTRY[name.lower()] = fn
        return fn
    return decorator


def _ensure_builtins():
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    import unireg.losses.similarity        # noqa: F401
    import unireg.losses.regularization    # noqa: F401


def build_sim_loss(cfg: dict):
    """Build a similarity loss from config.  Key: cfg['sim_loss']."""
    _ensure_builtins()
    name = cfg.get("sim_loss", "ncc").lower()
    if name not in _SIM_LOSS_REGISTRY:
        raise ValueError(
            f"Unknown sim_loss '{name}'. "
            f"Registered: {sorted(_SIM_LOSS_REGISTRY.keys())}"
        )
    return _SIM_LOSS_REGISTRY[name](cfg)


def build_reg_loss(cfg: dict):
    """Build a regularisation loss from config.  Key: cfg['reg_loss']."""
    _ensure_builtins()
    name = cfg.get("reg_loss", "grad_l2").lower()
    if name not in _REG_LOSS_REGISTRY:
        raise ValueError(
            f"Unknown reg_loss '{name}'. "
            f"Registered: {sorted(_REG_LOSS_REGISTRY.keys())}"
        )
    return _REG_LOSS_REGISTRY[name](cfg)


def registered_sim_losses():
    _ensure_builtins()
    return sorted(_SIM_LOSS_REGISTRY.keys())


def registered_reg_losses():
    _ensure_builtins()
    return sorted(_REG_LOSS_REGISTRY.keys())
