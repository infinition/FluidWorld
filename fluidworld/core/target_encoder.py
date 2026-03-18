"""
target_encoder.py -- EMA target encoder for self-supervised anti-collapse (BYOL/I-JEPA style).
"""

import copy
import math

import torch
import torch.nn as nn


class EMATargetEncoder(nn.Module):
    """
    Wraps an encoder with EMA weight updates for stable self-supervised targets.

    Args:
        online_encoder: online encoder (will be deep-copied)
        momentum: initial tau (typically 0.996, ramps toward 1.0)
    """

    def __init__(self, online_encoder: nn.Module, momentum: float = 0.996):
        super().__init__()
        self.momentum = momentum

        # Deep copy: same initial weights, independent parameters
        self.encoder = copy.deepcopy(online_encoder)

        # Target encoder never receives gradients
        for param in self.encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, online_encoder: nn.Module) -> None:
        """
        Update target weights via EMA. Call after optimizer.step().

        Args:
            online_encoder: online encoder (after gradient update)
        """
        for p_target, p_online in zip(
            self.encoder.parameters(), online_encoder.parameters()
        ):
            p_target.data.mul_(self.momentum).add_(
                p_online.data, alpha=1.0 - self.momentum
            )

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        """Forward pass without gradient through the target encoder."""
        return self.encoder(*args, **kwargs)


def cosine_momentum_schedule(
    base_momentum: float,
    final_momentum: float,
    current_step: int,
    total_steps: int,
) -> float:
    """
    Cosine schedule for EMA momentum: ramps from base_momentum to final_momentum.

    Args:
        base_momentum: initial momentum (e.g. 0.996)
        final_momentum: final momentum (e.g. 1.0)
        current_step: current iteration
        total_steps: total iterations

    Returns:
        Momentum value for current iteration
    """
    progress = current_step / max(total_steps, 1)
    cosine = (1.0 + math.cos(math.pi * progress)) / 2.0
    return final_momentum - (final_momentum - base_momentum) * cosine
