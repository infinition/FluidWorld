"""
_fluidvla_imports.py -- Safe import bridge for FluidVLA modules.
"""

import sys
import os

_here = os.path.dirname(os.path.abspath(__file__))
_fluidvla_core = os.path.normpath(
    os.path.join(_here, "..", "..", "..", "FluidVLA", "src", "core")
)

if not os.path.isdir(_fluidvla_core):
    raise ImportError(
        f"FluidVLA not found at {_fluidvla_core}. "
        f"FluidVLA must be at the same level as FluidWorld."
    )

# Add FluidVLA/src/core/ to sys.path for fallback imports.
if _fluidvla_core not in sys.path:
    sys.path.insert(0, _fluidvla_core)

# Direct imports from FluidVLA core:
from diffusion import Laplacian2D  # noqa: E402
from fluid_layer import (  # noqa: E402
    FluidLayer2D,
    ReactionMLP,
    MemoryPump,
    RMSNorm,
)
from vision_models import PatchEmbed  # noqa: E402
