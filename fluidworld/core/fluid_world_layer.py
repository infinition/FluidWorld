"""
fluid_world_layer.py -- Reaction-diffusion PDE layer with optional action forcing.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── FluidVLA core building blocks ──
from ._fluidvla_imports import Laplacian2D, ReactionMLP, MemoryPump, RMSNorm

from .action_force import ActionForce


class ContentGatedDiffusion(nn.Module):
    """Content-gated anisotropic diffusion -- O(N) routing without attention.

    gate = sigmoid(Conv1x1(u)); diff = gate * Laplacian2D(u).
    Learns where information should propagate (gate~1) or be blocked (gate~0).

    Args:
        channels: latent channel count
        gate_bias: initial gate bias (> 0 starts near isotropic diffusion)
    """

    def __init__(self, channels: int, gate_bias: float = 2.0):
        super().__init__()
        self.gate_conv = nn.Conv2d(channels, channels, kernel_size=1)
        # Init bias so sigmoid(bias) ~ 0.88 (starts near isotropic)
        nn.init.zeros_(self.gate_conv.weight)
        nn.init.constant_(self.gate_conv.bias, gate_bias)

    def forward(
        self, u: torch.Tensor, raw_diffusion: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            u: (B, C, H, W) current latent state (for gate computation)
            raw_diffusion: (B, C, H, W) raw Laplacian2D output

        Returns:
            (B, C, H, W) spatially modulated diffusion
        """
        gate = torch.sigmoid(self.gate_conv(u))  # (B, C, H, W) in [0, 1]
        return gate * raw_diffusion


class FluidWorldLayer2D(nn.Module):
    """
    2D reaction-diffusion layer with optional action forcing.

    Extends FluidLayer2D from FluidVLA. When action=None, equivalent to FluidLayer2D.

    Args:
        channels: latent width (= d_model)
        action_dim: robot action vector dimension
        dilations: spatial diffusion scales
        max_steps: max internal integration steps
        dt: initial integration timestep
        epsilon: turbulence threshold for early stopping
        alpha: initial global memory weight
        use_pde: enable/disable diffusion (for ablation)
        norm_type: normalization type ("rmsnorm" or "layernorm")
        norm_every: normalization frequency (every N steps)
        local_memory_hw: low-resolution local context size
        signed_diffusion: allow signed diffusion coefficients
        diffusion_scale: max learned diffusion amplitude
        stop_patience: consecutive steps below epsilon before stopping
        min_steps: minimum steps before early stopping
        stop_probe_hw: turbulence probe resolution
        force_spatial_size: internal force field resolution
    """

    def __init__(
        self,
        channels: int,
        action_dim: int = 6,
        dilations: List[int] = [1, 4, 16],
        max_steps: int = 12,
        dt: float = 0.1,
        epsilon: float = 0.08,
        alpha: float = 0.1,
        use_pde: bool = True,
        norm_type: str = "rmsnorm",
        norm_every: int = 2,
        local_memory_hw: int = 4,
        signed_diffusion: bool = False,
        diffusion_scale: float = 0.25,
        stop_patience: int = 2,
        min_steps: int = 3,
        stop_probe_hw: int = 8,
        force_spatial_size: int = 4,
        anisotropic_diffusion: bool = False,
        anisotropic_gate_bias: float = 2.0,
    ):
        super().__init__()

        self.channels = channels
        self.max_steps = int(max_steps)
        self.epsilon = float(epsilon)
        self.use_pde = bool(use_pde)
        self.norm_every = max(1, int(norm_every))
        self.stop_patience = max(1, int(stop_patience))
        self.min_steps = max(1, int(min_steps))
        self.stop_probe_hw = max(1, int(stop_probe_hw))
        self.local_memory_hw = max(1, int(local_memory_hw))

        # ── FluidVLA building blocks (identical) ──
        self.reaction = ReactionMLP(channels)
        self.memory = MemoryPump(channels)
        self.diffusion = Laplacian2D(
            channels=channels,
            dilations=dilations,
            signed_diffusion=signed_diffusion,
            diffusion_scale=diffusion_scale,
        )
        self.local_proj = nn.Conv2d(channels, channels, kernel_size=1)

        # ── Anisotropic diffusion (optional) ──
        self.anisotropic_diffusion = anisotropic_diffusion
        if anisotropic_diffusion:
            self.content_gate = ContentGatedDiffusion(
                channels, gate_bias=anisotropic_gate_bias
            )
        else:
            self.content_gate = None

        # ── Learned coefficients (same as FluidVLA) ──
        self.alpha_global = nn.Parameter(torch.tensor(float(alpha)))
        self.alpha_local = nn.Parameter(torch.tensor(alpha * 0.5))
        self.log_dt = nn.Parameter(torch.log(torch.tensor(float(dt))))

        norm_type = norm_type.lower()
        if norm_type == "rmsnorm":
            self.norm = RMSNorm(channels)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(channels)
        else:
            raise ValueError("norm_type must be 'rmsnorm' or 'layernorm'")

        # ── Action forcing (optional) ──
        # When action_dim=0, skip ActionForce (saves ~260K params/layer)
        self.action_dim = action_dim
        if action_dim > 0:
            self.action_force = ActionForce(
                action_dim=action_dim,
                channels=channels,
                force_spatial_size=force_spatial_size,
            )
            # beta: force/dynamics coupling, learned, starts small
            self.beta = nn.Parameter(torch.tensor(0.05))
        else:
            self.action_force = None
            self.beta = None

    def _dt(self) -> torch.Tensor:
        return self.log_dt.exp().clamp(0.005, 0.35)

    def _alpha(self) -> torch.Tensor:
        return F.softplus(self.alpha_global)

    def _make_stop_probe(self, x: torch.Tensor) -> torch.Tensor:
        """Low-resolution probe for turbulence measurement."""
        probe = F.adaptive_avg_pool2d(
            x,
            output_size=(
                min(self.stop_probe_hw, x.shape[-2]),
                min(self.stop_probe_hw, x.shape[-1]),
            ),
        )
        return probe

    def _should_stop(self, history: List[float], step_idx: int) -> bool:
        """Early stopping in eval mode only."""
        if self.training or self.epsilon <= 0.0:
            return False
        if (step_idx + 1) < self.min_steps:
            return False
        if len(history) < self.stop_patience:
            return False
        window = history[-self.stop_patience :]
        return max(window) < self.epsilon

    def forward(
        self,
        u: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        PDE integration step with optional action forcing.

        Args:
            u: (B, C, H, W) current latent state
            action: (B, action_dim) robot action, or None for perception mode

        Returns:
            u: (B, C, H, W) latent state after integration
            info: diagnostics dict (steps_used, turbulence, etc.)
        """
        B, C, H, W = u.shape
        device = u.device
        n = H * W

        h_state = torch.zeros(B, C, device=device, dtype=u.dtype)

        # ── Precompute force field (constant during integration) ──
        if action is not None and self.action_force is not None:
            force_field = self.action_force(action, (H, W))  # (B, C, H, W)
            force_flat = force_field.permute(0, 2, 3, 1).reshape(B, n, C)
            beta = F.softplus(self.beta)
        else:
            force_flat = None
            beta = None

        stop_history: List[float] = []
        diff_turbulences: List[torch.Tensor] = []
        step_energies: List[torch.Tensor] = []  # v8: fully differentiable energy
        equilibrium_step = self.max_steps
        prev_probe = self._make_stop_probe(u).detach()

        for step in range(self.max_steps):
            # ── Local diffusion (spatial propagation) ──
            diff = self.diffusion(u) if self.use_pde else torch.zeros_like(u)

            # ── Anisotropic modulation (content-based routing) ──
            if self.content_gate is not None:
                diff = self.content_gate(u, diff)

            u_flat = u.permute(0, 2, 3, 1).reshape(B, n, C)
            diff_flat_step = diff.permute(0, 2, 3, 1).reshape(B, n, C)

            # ── Per-position nonlinear reaction ──
            react = self.reaction(u_flat)

            # ── O(1) global memory ──
            pooled = u_flat.mean(dim=1)
            h_state = self.memory(h_state, pooled)
            h_global = h_state.unsqueeze(1).expand(-1, n, -1)

            # ── Low-resolution local memory ──
            local_mem = F.adaptive_avg_pool2d(
                u, output_size=(self.local_memory_hw, self.local_memory_hw)
            )
            local_mem = self.local_proj(local_mem)
            local_mem = F.interpolate(
                local_mem, size=(H, W), mode="bilinear", align_corners=False
            )
            local_mem = local_mem.permute(0, 2, 3, 1).reshape(B, n, C)

            # ── Assemble update ──
            dt = self._dt()
            alpha_global = self._alpha()
            alpha_local = F.softplus(self.alpha_local)

            du = (
                diff_flat_step
                + react
                + alpha_global * h_global
                + alpha_local * local_mem
            )

            # ── Action forcing term ──
            if force_flat is not None:
                du = du + beta * force_flat

            # ── Explicit integration ──
            u_candidate = u_flat + dt * du

            # ── Periodic normalization ──
            u_flat_next = u_candidate
            if (step + 1) % self.norm_every == 0:
                u_flat_next = self.norm(u_flat_next)

            u_candidate_img = (
                u_candidate.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            )
            u_next = (
                u_flat_next.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            )

            # ── Turbulence measurement on low-res probe ──
            current_probe = self._make_stop_probe(u_candidate_img).detach()
            stop_turb = (current_probe - prev_probe).abs().mean() / (
                prev_probe.abs().mean() + 1e-8
            )

            # ── Differentiable turbulence for regularization ──
            step_energy = du.abs().mean()
            lap_energy = (
                diff_flat_step.abs().mean()
                if self.use_pde
                else torch.zeros((), device=device, dtype=u.dtype)
            )
            diff_turb = stop_turb + 0.05 * step_energy + 0.01 * lap_energy
            diff_turbulences.append(diff_turb)
            step_energies.append(step_energy)  # v8: raw update energy

            stop_val = float(stop_turb.item())
            stop_history.append(stop_val)

            if stop_val < self.epsilon and equilibrium_step == self.max_steps:
                equilibrium_step = step + 1

            u = u_next
            prev_probe = current_probe

            if self._should_stop(stop_history, step):
                break

        diff_turb_mean = (
            torch.stack(diff_turbulences).mean()
            if diff_turbulences
            else torch.zeros((), device=device, dtype=u.dtype)
        )

        # v8: 100% differentiable PDE energy (no detached terms)
        step_energy_mean = (
            torch.stack(step_energies).mean()
            if step_energies
            else torch.zeros((), device=device, dtype=u.dtype)
        )

        # ── Anisotropic gate stats (last step) ──
        if self.content_gate is not None:
            with torch.no_grad():
                last_gate = torch.sigmoid(self.content_gate.gate_conv(u))
                gate_mean = last_gate.mean().item()
                gate_std = last_gate.std().item()
        else:
            gate_mean = 1.0  # isotropic = constant gate at 1
            gate_std = 0.0

        return u, {
            "steps_used": len(stop_history),
            "stop_history": stop_history,
            "equilibrium_step": equilibrium_step,
            "final_turbulence": stop_history[-1] if stop_history else 0.0,
            "min_turbulence": min(stop_history) if stop_history else 0.0,
            "diff_turbulence": diff_turb_mean,
            "step_energy": step_energy_mean,  # v8: 100% differentiable
            "pde_active": self.use_pde,
            "gate_mean": gate_mean,  # anisotropic diffusion gate stats
            "gate_std": gate_std,
            "action_injected": action is not None,
        }


def compute_equilibrium_loss(info_list: list) -> torch.Tensor:
    """Legacy equilibrium loss. DEPRECATED in v8 -- use compute_pde_alive_loss instead."""
    diff_turbs = [
        info["diff_turbulence"]
        for info in info_list
        if isinstance(info.get("diff_turbulence"), torch.Tensor)
    ]
    if not diff_turbs:
        return torch.tensor(0.0)
    return torch.stack(diff_turbs).mean()


def compute_pde_alive_loss(
    info_list: list,
    target_eq: float = 0.5,
) -> dict:
    """
    PDE-Alive loss (v8): two-sided regularizer penalizing deviation from target energy.

    Prevents both PDE death (features become static) and chaos (wasted compute).
    Loss: (mean_step_energy - target_eq)^2, fully differentiable.

    Args:
        info_list: list of dicts returned by FluidLayer2D
        target_eq: target PDE energy (calibrate via TensorBoard PDE/Step_Energy)

    Returns:
        dict with pde_alive_loss, mean_turbulence, mean_step_energy
    """
    # 100% differentiable PDE energy
    step_energies = [
        info["step_energy"]
        for info in info_list
        if isinstance(info.get("step_energy"), torch.Tensor)
    ]
    # Legacy turbulence for monitoring
    diff_turbs = [
        info["diff_turbulence"]
        for info in info_list
        if isinstance(info.get("diff_turbulence"), torch.Tensor)
    ]

    if not step_energies:
        return {
            "pde_alive_loss": torch.tensor(0.0),
            "mean_turbulence": 0.0,
            "mean_step_energy": 0.0,
        }

    mean_energy = torch.stack(step_energies).mean()
    mean_turb = torch.stack(diff_turbs).mean() if diff_turbs else mean_energy

    # Quadratic loss centered on target -- 100% differentiable
    pde_alive_loss = (mean_energy - target_eq) ** 2

    return {
        "pde_alive_loss": pde_alive_loss,
        "mean_turbulence": mean_turb.detach().item(),
        "mean_step_energy": mean_energy.detach().item(),
    }
