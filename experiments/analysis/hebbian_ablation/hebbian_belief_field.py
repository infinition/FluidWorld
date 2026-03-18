"""
hebbian_belief_field.py -- BeliefField with Hebbian plasticity

"What fires together wires together": frequently used diffusion paths
grow stronger, creating emergent associative memory within the
BeliefField.
"""

from typing import Optional

import torch
import torch.nn as nn

from fluidworld.core.belief_field import BeliefField


class HebbianBeliefField(BeliefField):
    """
    Extends the standard BeliefField with a Hebbian gating mechanism.

    The Hebbian trace tracks cumulative activity and modulates diffusion
    coefficients: active regions diffuse faster, creating propagation
    "highways".

    Args:
        alpha: Hebbian reinforcement strength (0 = no effect)
        gamma: trace decay rate (0.95 = long memory)
        normalize_trace: normalize trace to prevent explosion
    """

    def __init__(
        self,
        channels: int,
        stimulus_dim: int = 6,
        spatial_hw: int = 16,
        decay: float = 0.95,
        n_evolve_steps: int = 3,
        dilations: list = [1, 4],
        alpha: float = 0.1,
        gamma: float = 0.95,
        normalize_trace: bool = True,
    ):
        super().__init__(
            channels=channels,
            stimulus_dim=stimulus_dim,
            spatial_hw=spatial_hw,
            decay=decay,
            n_evolve_steps=n_evolve_steps,
            dilations=dilations,
        )
        self.alpha = alpha
        self.gamma = gamma
        self.normalize_trace = normalize_trace

        # Projection to modulate diffusion based on the trace
        self.trace_gate = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )

    def init_trace(self, batch_size, device, dtype=torch.float32):
        """Initialize the Hebbian trace to zero."""
        return torch.zeros(
            batch_size, self.channels,
            self.spatial_hw, self.spatial_hw,
            device=device, dtype=dtype,
        )

    def evolve_with_hebbian(
        self,
        state: torch.Tensor,
        trace: torch.Tensor,
        stimulus: Optional[torch.Tensor] = None,
    ):
        """
        Evolve with Hebbian modulation.

        The trace modulates diffusion coefficients:
        diff_coeff = base_coeff * (1 + alpha * trace_gate(trace))

        Returns:
            (new_state, new_trace)
        """
        B, C, H_b, W_b = state.shape
        n = H_b * W_b
        u = state

        # Precompute stimulus
        if stimulus is not None:
            stim = self.stimulus_proj(stimulus)
            stim_field = stim.unsqueeze(-1).unsqueeze(-1).expand(B, C, H_b, W_b)
            stim_flat = stim_field.permute(0, 2, 3, 1).reshape(B, n, C)
        else:
            stim_flat = None

        dt = self.dt

        for step in range(self.n_evolve_steps):
            # Diffusion spatiale standard
            diff = self.diffusion(u)

            # Hebbian modulation of diffusion
            trace_flat = trace.permute(0, 2, 3, 1).reshape(B * n, C)
            gate = self.trace_gate(trace_flat).reshape(B, H_b, W_b, C)
            gate = gate.permute(0, 3, 1, 2)  # (B, C, H_b, W_b)
            diff = diff * (1.0 + self.alpha * gate)

            # Flatten for reaction
            u_flat = u.permute(0, 2, 3, 1).reshape(B, n, C)
            diff_flat = diff.permute(0, 2, 3, 1).reshape(B, n, C)

            react = self.reaction(u_flat)

            du = diff_flat + react
            if stim_flat is not None:
                du = du + stim_flat

            u_flat = u_flat + dt * du

            if (step + 1) % 2 == 0:
                u_flat = self.norm(u_flat)

            u = u_flat.reshape(B, H_b, W_b, C).permute(0, 3, 1, 2).contiguous()

        # Update Hebbian trace
        activity = u.abs()  # activity = state magnitude
        new_trace = self.gamma * trace + (1.0 - self.gamma) * activity

        if self.normalize_trace:
            trace_max = new_trace.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
            new_trace = new_trace / trace_max

        return u, new_trace
