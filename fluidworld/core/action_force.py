"""
action_force.py -- Projects robot actions into spatial force fields for PDE dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionForce(nn.Module):
    """
    Projects a robot action vector into a spatial force field for the PDE.

    Maps (B, action_dim) -> MLP -> reshape (B, C, S, S) -> interpolate (B, C, H, W).

    Args:
        action_dim: action vector dimension (e.g. 6 for SO-101)
        channels: latent channel count (= d_model)
        force_spatial_size: internal force grid resolution (default 4)
    """

    def __init__(
        self,
        action_dim: int,
        channels: int,
        force_spatial_size: int = 4,
    ):
        super().__init__()
        self.channels = channels
        self.force_spatial_size = force_spatial_size
        out_dim = channels * force_spatial_size * force_spatial_size

        self.mlp = nn.Sequential(
            nn.Linear(action_dim, channels),
            nn.GELU(),
            nn.LayerNorm(channels),
            nn.Linear(channels, out_dim),
        )

        self._init_small()

    def _init_small(self):
        """Initialize last layer with small weights to avoid destabilizing PDE dynamics early on."""
        with torch.no_grad():
            self.mlp[-1].weight.mul_(0.01)
            self.mlp[-1].bias.zero_()

    def forward(self, action: torch.Tensor, spatial_shape: tuple) -> torch.Tensor:
        """
        Convert action vector to spatial force field.

        Args:
            action: (B, action_dim) robot action vector
            spatial_shape: (H, W) target latent spatial size

        Returns:
            (B, channels, H, W) spatial force field
        """
        B = action.shape[0]
        S = self.force_spatial_size
        H, W = spatial_shape

        # (B, action_dim) -> (B, channels * S * S)
        force = self.mlp(action)

        # Reshape to spatial grid
        force = force.view(B, self.channels, S, S)

        # Interpolate to actual latent spatial size
        if (H, W) != (S, S):
            force = F.interpolate(
                force, size=(H, W), mode="bilinear", align_corners=False
            )

        return force
