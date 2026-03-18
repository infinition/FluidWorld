"""
FluidWorld -- Reaction-diffusion world model. Public API for the core module.
"""

from .action_force import ActionForce
from .fluid_world_layer import FluidWorldLayer2D, ContentGatedDiffusion, compute_equilibrium_loss
from .target_encoder import EMATargetEncoder, cosine_momentum_schedule
from .vicreg import variance_loss, covariance_loss, vicreg_loss
from .world_model import FluidWorldModel, OnlineEncoder
from .proprio_model import ProprioWorldModel, MultiStepProprioModel

__all__ = [
    # Core
    "FluidWorldModel",
    "OnlineEncoder",
    "FluidWorldLayer2D",
    "ActionForce",
    # Target encoder
    "EMATargetEncoder",
    "cosine_momentum_schedule",
    # Losses
    "variance_loss",
    "covariance_loss",
    "vicreg_loss",
    "compute_equilibrium_loss",
    # Phase 0
    "ProprioWorldModel",
    "MultiStepProprioModel",
]
