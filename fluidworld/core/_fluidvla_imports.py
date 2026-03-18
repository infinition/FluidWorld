"""
_fluidvla_imports.py -- Local imports (modules copied from FluidVLA).

These modules are now bundled directly in fluidworld/core/.
No external FluidVLA dependency required.
"""

from .diffusion import Laplacian2D
from .fluid_layer import FluidLayer2D, ReactionMLP, MemoryPump, RMSNorm
from .vision_models import PatchEmbed
