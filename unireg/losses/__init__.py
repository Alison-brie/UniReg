"""
Loss modules for registration training.

To add a new loss, create a file and use @register_sim_loss / @register_reg_loss.
Then add one import line here to ensure registration on startup.
"""

from unireg.losses.similarity import NCC3D, MSELoss, MINDSSCLoss
from unireg.losses.regularization import GradLoss
from unireg.losses.registry import (
    register_sim_loss, register_reg_loss,
    build_sim_loss, build_reg_loss,
)

__all__ = [
    "NCC3D", "MSELoss", "MINDSSCLoss", "GradLoss",
    "register_sim_loss", "register_reg_loss",
    "build_sim_loss", "build_reg_loss",
]
