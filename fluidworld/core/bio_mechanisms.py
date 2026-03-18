"""
bio_mechanisms.py -- Bio-inspired mechanisms: SynapticFatigue, LateralInhibition, HebbianDiffusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SynapticFatigue(nn.Module):
    """
    Per-channel synaptic fatigue. High activation depletes health; passive recovery restores it.

    effective_output = input * health (gradient flows through input; health is treated as constant).

    Args:
        channels: number of channels
        cost: depletion rate per unit activation
        recovery: passive recovery per step
        min_health: health floor (prevents channel death)
    """

    def __init__(
        self,
        channels: int,
        cost: float = 0.1,
        recovery: float = 0.02,
        min_health: float = 0.1,
    ):
        super().__init__()
        self.cost = cost
        self.recovery = recovery
        self.min_health = min_health
        # health: (1, C, 1, 1) - broadcasts over batch and spatial dims
        self.register_buffer("health", torch.ones(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) encoder features

        Returns:
            (B, C, H, W) health-modulated features
        """
        # Snapshot health BEFORE in-place update (required for autograd correctness)
        health_snapshot = self.health.clone()

        # Forward: multiply by health (differentiable w.r.t. x)
        out = x * health_snapshot

        # Update health (side effect, no gradient)
        with torch.no_grad():
            # Per-channel activation intensity (mean over batch + spatial)
            activation = x.abs().mean(dim=(0, 2, 3), keepdim=True)  # (1, C, 1, 1)

            # Fatigue: higher activation = more depletion
            self.health.sub_(self.cost * activation)

            # Passive recovery
            self.health.add_(self.recovery)

            # Clamp
            self.health.clamp_(self.min_health, 1.0)

        return out

    def get_stats(self) -> dict:
        """Stats for TensorBoard."""
        h = self.health.squeeze()
        return {
            "health_mean": h.mean().item(),
            "health_min": h.min().item(),
            "health_max": h.max().item(),
            "fatigued_channels": (h < 0.5).sum().item(),
        }


class LateralInhibition(nn.Module):
    """
    Cross-channel lateral inhibition. Strong channels suppress weak ones per spatial position.

    Args:
        strength: inhibition strength (0 = off, 0.5 = moderate)
        min_factor: inhibition floor (prevents total suppression)
    """

    def __init__(self, strength: float = 0.3, min_factor: float = 0.2):
        super().__init__()
        self.strength = strength
        self.min_factor = min_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) features

        Returns:
            (B, C, H, W) laterally inhibited features
        """
        energy = x.abs()  # (B, C, H, W)
        max_energy = energy.max(dim=1, keepdim=True).values  # (B, 1, H, W)

        # Ratio: 1.0 for strongest channel, < 1 for others
        ratio = energy / (max_energy + 1e-6)

        # Inhibition: weak channels are suppressed
        inhibition = 1.0 - self.strength * (1.0 - ratio)
        inhibition = inhibition.clamp(min=self.min_factor)

        return x * inhibition


class HebbianDiffusion(nn.Module):
    """
    Hebbian modulation of PDE diffusion. Co-active paths strengthen over time.

    diff_modulated = diff * (1 + hebbian_gain * hebbian_map).
    Updates without gradient -- works during inference.

    Args:
        channels: number of channels
        spatial_hw: spatial resolution
        hebbian_lr: Hebbian learning rate
        hebbian_decay: forgetting rate
        hebbian_gain: modulation amplitude
        hebbian_max: upper bound of the map
    """

    def __init__(
        self,
        channels: int,
        spatial_hw: int = 16,
        hebbian_lr: float = 0.01,
        hebbian_decay: float = 0.99,
        hebbian_gain: float = 0.5,
        hebbian_max: float = 2.0,
    ):
        super().__init__()
        self.hebbian_lr = hebbian_lr
        self.hebbian_decay = hebbian_decay
        self.hebbian_gain = hebbian_gain
        self.hebbian_max = hebbian_max

        # Persistent Hebbian map (averaged over batches)
        self.register_buffer(
            "hebbian_map",
            torch.zeros(1, channels, spatial_hw, spatial_hw),
        )

    def update_and_modulate(
        self,
        u: torch.Tensor,
        diff: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update the Hebbian map and modulate diffusion.

        Args:
            u: (B, C, H, W) current BeliefField state
            diff: (B, C, H, W) raw PDE diffusion term

        Returns:
            (B, C, H, W) modulated diffusion
        """
        # Resize if needed
        if self.hebbian_map.shape[-2:] != u.shape[-2:]:
            self.hebbian_map = F.interpolate(
                self.hebbian_map,
                size=u.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # Co-activation: local correlation between each position and its neighbors
        with torch.no_grad():
            u_smooth = F.avg_pool2d(
                u, kernel_size=3, stride=1, padding=1
            )
            # Positive when u and neighbors have the same direction
            co_act = (u * u_smooth).clamp(min=0)

            # Mean over batch (population-level accumulation)
            co_act_mean = co_act.mean(dim=0, keepdim=True)  # (1, C, H, W)

            # Hebbian update
            self.hebbian_map.mul_(self.hebbian_decay)
            self.hebbian_map.add_(self.hebbian_lr * co_act_mean)
            self.hebbian_map.clamp_(0, self.hebbian_max)

        # Modulate diffusion (differentiable)
        modulated = diff * (1.0 + self.hebbian_gain * self.hebbian_map)

        return modulated

    def get_stats(self) -> dict:
        """Stats for TensorBoard."""
        m = self.hebbian_map.squeeze()
        return {
            "hebbian_mean": m.mean().item(),
            "hebbian_max": m.max().item(),
            "hebbian_std": m.std().item(),
            "hebbian_active": (m > 0.1).float().mean().item(),
        }
