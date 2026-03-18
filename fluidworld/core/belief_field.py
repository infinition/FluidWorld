"""
belief_field.py -- Persistent world memory as a 2D latent field (write/evolve/read).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._fluidvla_imports import Laplacian2D, ReactionMLP, MemoryPump, RMSNorm
from .bio_mechanisms import HebbianDiffusion


class BeliefField(nn.Module):
    """
    Persistent 2D latent field encoding world history.

    Three operations: write (integrate observation), evolve (internal PDE dynamics),
    read (extract usable representation via spatial pooling + projection).

    Args:
        channels: latent channel count (= d_model)
        spatial_hw: spatial resolution (H_b = W_b)
        decay: initial retention rate (0.95 = keep 95% per step)
        n_evolve_steps: PDE integration steps in evolve()
        dilations: spatial diffusion scales
    """

    def __init__(
            self,
            channels: int,
            stimulus_dim: int = 6,
            spatial_hw: int = 16,
            decay: float = 0.95,
            n_evolve_steps: int = 3,
            dilations: list = [1, 4],
            use_memory_pump: bool = True,
            use_hebbian: bool = True,
            hebbian_lr: float = 0.01,
            hebbian_decay: float = 0.99,
        ):
            super().__init__()

            self.channels = channels
            self.spatial_hw = spatial_hw
            self.n_evolve_steps = n_evolve_steps
            self.use_memory_pump = use_memory_pump
            self.use_hebbian = use_hebbian

            # ── Write mechanism ──
            self.gate_proj = nn.Sequential(
                nn.Linear(channels, channels),
                nn.Sigmoid(),
            )
            self.value_proj = nn.Sequential(
                nn.Linear(channels, channels),
                nn.Tanh(),
            )

            # ── Internal dynamics (evolve) ──
            self.diffusion = Laplacian2D(
                channels=channels,
                dilations=dilations,
                signed_diffusion=False,
                diffusion_scale=0.25,
            )
            self.reaction = ReactionMLP(channels)
            self.norm = RMSNorm(channels)

            # ── Global Memory Pump (RESEARCH.md #3) ──
            # O(1) global summary broadcast to all positions.
            if use_memory_pump:
                self.memory_pump = MemoryPump(channels)
                self.alpha_pump = nn.Parameter(torch.tensor(0.1))
                # h_global stored in module, not in state tensor
                self._h_global = None

            # ── Hebbian Diffusion (RESEARCH.md #14 weight accumulation) ──
            # Co-active paths strengthen. Works during inference (no backprop needed).
            if use_hebbian:
                self.hebbian = HebbianDiffusion(
                    channels=channels,
                    spatial_hw=spatial_hw,
                    hebbian_lr=hebbian_lr,
                    hebbian_decay=hebbian_decay,
                )

            # ── External stimulus injection (optional) ──
            self.stimulus_proj = nn.Sequential(
                nn.Linear(stimulus_dim, channels),
                nn.GELU(),
                nn.Linear(channels, channels),
            )

            # ── Read mechanism ──
            self.read_proj = nn.Sequential(
                nn.LayerNorm(channels),
                nn.Linear(channels, channels),
            )

            # ── Learned parameters ──
            self.log_decay = nn.Parameter(torch.log(torch.tensor(decay)))
            self.log_dt = nn.Parameter(torch.log(torch.tensor(0.1)))

            self._init_small()

    def _init_small(self):
        """Small init for stimulus projection last layer."""
        with torch.no_grad():
            self.stimulus_proj[-1].weight.mul_(0.01)
            self.stimulus_proj[-1].bias.zero_()

    @property
    def decay(self) -> torch.Tensor:
        return self.log_decay.exp().clamp(0.5, 0.99)

    @property
    def dt(self) -> torch.Tensor:
        return self.log_dt.exp().clamp(0.01, 0.3)

    def init_state(
        self,
        batch_size: int,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Create an empty initial state (zeros).

        Returns:
            (B, C, H_b, W_b) initial state
        """
        # Reset global memory pump
        if self.use_memory_pump:
            self._h_global = torch.zeros(
                batch_size, self.channels,
                device=device, dtype=dtype,
            )

        return torch.zeros(
            batch_size, self.channels,
            self.spatial_hw, self.spatial_hw,
            device=device, dtype=dtype,
        )

    def detach_hidden(self):
        """Detach h_global for TBPTT (called between temporal segments)."""
        if self.use_memory_pump and self._h_global is not None:
            self._h_global = self._h_global.detach()

    def write(
        self,
        state: torch.Tensor,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Integrate an observation into the belief field via gated update.

        Args:
            state: (B, C, H_b, W_b) current field state
            observation: (B, C, H, W) or (B, C) encoded observation

        Returns:
            (B, C, H_b, W_b) updated state
        """
        B, C, H_b, W_b = state.shape

        # Adapt observation to field resolution
        if observation.dim() == 4:
            # (B, C, H, W) -> (B, C, H_b, W_b)
            obs = F.adaptive_avg_pool2d(observation, (H_b, W_b))
        elif observation.dim() == 2:
            # (B, C) -> (B, C, H_b, W_b) broadcast spatial
            obs = observation.unsqueeze(-1).unsqueeze(-1).expand(B, C, H_b, W_b)
        else:
            raise ValueError(
                f"observation must be (B, C, H, W) or (B, C), "
                f"got shape {observation.shape}"
            )

        # Gate/value projections in (B, n, C) space for Linear layers
        obs_flat = obs.permute(0, 2, 3, 1).reshape(B * H_b * W_b, C)
        gate = self.gate_proj(obs_flat).reshape(B, H_b, W_b, C).permute(0, 3, 1, 2)
        value = self.value_proj(obs_flat).reshape(B, H_b, W_b, C).permute(0, 3, 1, 2)

        # Update field
        decay = self.decay
        new_state = decay * state + gate * value

        return new_state

    def evolve(
        self,
        state: torch.Tensor,
        stimulus: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Evolve the field via internal PDE (diffusion + reaction + optional stimulus).

        Args:
            state: (B, C, H_b, W_b) current state
            stimulus: (B, D) optional perturbation vector

        Returns:
            (B, C, H_b, W_b) evolved state
        """
        B, C, H_b, W_b = state.shape
        n = H_b * W_b
        u = state

        # Precompute spatial stimulus (constant during integration)
        if stimulus is not None:
            stim = self.stimulus_proj(stimulus)  # (B, C)
            stim_field = stim.unsqueeze(-1).unsqueeze(-1).expand(B, C, H_b, W_b)
            stim_flat = stim_field.permute(0, 2, 3, 1).reshape(B, n, C)
        else:
            stim_flat = None

        dt = self.dt

        # Memory Pump: detach for TBPTT
        if self.use_memory_pump and self._h_global is not None:
            self._h_global = self._h_global.detach()

        for step in range(self.n_evolve_steps):
            # Spatial diffusion
            diff = self.diffusion(u)

            # Hebbian: modulate diffusion by co-activation
            if self.use_hebbian:
                diff = self.hebbian.update_and_modulate(u, diff)

            # Flatten for reaction
            u_flat = u.permute(0, 2, 3, 1).reshape(B, n, C)
            diff_flat = diff.permute(0, 2, 3, 1).reshape(B, n, C)

            # Local reaction
            react = self.reaction(u_flat)

            # Assembly
            du = diff_flat + react
            if stim_flat is not None:
                du = du + stim_flat

            # Global Memory Pump: global summary broadcast
            if self.use_memory_pump and self._h_global is not None:
                pooled = u_flat.mean(dim=1)  # (B, C)
                self._h_global = self.memory_pump(self._h_global, pooled)
                alpha_pump = F.softplus(self.alpha_pump)
                h_broadcast = self._h_global.unsqueeze(1).expand(-1, n, -1)
                du = du + alpha_pump * h_broadcast

            # Euler integration
            u_flat = u_flat + dt * du

            # Periodic normalization
            if (step + 1) % 2 == 0:
                u_flat = self.norm(u_flat)

            u = u_flat.reshape(B, H_b, W_b, C).permute(0, 3, 1, 2).contiguous()

        return u

    def read(self, state: torch.Tensor) -> torch.Tensor:
        """
        Extract a vector representation from the belief field (spatial pool + projection).

        Args:
            state: (B, C, H_b, W_b) current state

        Returns:
            (B, C) vector representation (in autograd graph)
        """
        # Spatial pool
        pooled = state.mean(dim=(-2, -1))  # (B, C)
        # Project
        return self.read_proj(pooled)  # (B, C)

    def read_spatial(self, state: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        """
        Extract a 2D representation at the desired resolution.

        Args:
            state: (B, C, H_b, W_b) current state
            target_hw: (H, W) target resolution

        Returns:
            (B, C, H, W) interpolated field
        """
        H, W = target_hw
        if (H, W) == (state.shape[2], state.shape[3]):
            return state
        return F.interpolate(
            state, size=(H, W), mode="bilinear", align_corners=False
        )

    def forward(
        self,
        state: torch.Tensor,
        observation: Optional[torch.Tensor] = None,
        stimulus: Optional[torch.Tensor] = None,
        detach_state: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full cycle: write + evolve + read.

        Args:
            state: (B, C, H_b, W_b) current state
            observation: (B, C, H, W) or (B, C) new observation
            stimulus: (B, D) optional perturbation

        Returns:
            (new_state, representation): updated state and vector for loss
        """
        if observation is not None:
            state = self.write(state, observation)
        state = self.evolve(state, stimulus=stimulus)
        representation = self.read(state)
        if detach_state:
            state = state.detach()

        return state, representation
