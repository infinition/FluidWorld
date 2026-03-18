"""
world_model.py -- FluidWorld v9: JEPA-based world model with spatial prediction and PDE-Alive loss.
"""

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._fluidvla_imports import FluidLayer2D, PatchEmbed
from .fluid_world_layer import compute_equilibrium_loss, compute_pde_alive_loss
from .target_encoder import EMATargetEncoder
from .vicreg import vicreg_loss
from .belief_field import BeliefField


class OnlineEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        d_model: int = 128,
        n_layers: int = 3,
        patch_size: int = 4,
        dilations: Sequence[int] = (1, 4, 16),
        max_steps: int = 12,
        dt: float = 0.1,
        epsilon: float = 0.08,
        norm_type: str = "rmsnorm",
        norm_every: int = 2,
        local_memory_hw: int = 4,
        signed_diffusion: bool = False,
        diffusion_scale: float = 0.25,
        stop_patience: int = 2,
        min_steps: int = 3,
        stop_probe_hw: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_embed = PatchEmbed(
            in_channels=in_channels, d_model=d_model,
            patch_size=patch_size, norm_type=norm_type,
        )
        self.fluid_layers = nn.ModuleList([
            FluidLayer2D(
                channels=d_model, dilations=list(dilations),
                max_steps=max_steps, dt=dt, epsilon=epsilon,
                use_pde=True, norm_type=norm_type, norm_every=norm_every,
                local_memory_hw=local_memory_hw,
                signed_diffusion=signed_diffusion,
                diffusion_scale=diffusion_scale,
                stop_patience=stop_patience, min_steps=min_steps,
                stop_probe_hw=stop_probe_hw,
            )
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> Dict:
        u = self.patch_embed(x)
        info_list = []
        for layer in self.fluid_layers:
            u, info = layer(u)
            info_list.append(info)
        return {"features": u, "info": info_list}


class FluidWorldModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        d_model: int = 128,
        stimulus_dim: int = 6,
        n_encoder_layers: int = 3,
        n_world_layers: int = 3,
        patch_size: int = 4,
        dilations: Sequence[int] = (1, 4, 16),
        max_steps_encoder: int = 12,
        max_steps_world: int = 3,
        dt: float = 0.1,
        epsilon: float = 0.08,
        momentum: float = 0.996,
        pred_hidden: int = 64,
        norm_type: str = "rmsnorm",
        norm_every: int = 2,
        local_memory_hw: int = 4,
        signed_diffusion: bool = False,
        diffusion_scale: float = 0.25,
        belief_spatial_hw: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        self.stimulus_dim = stimulus_dim

        self.encoder = OnlineEncoder(
            in_channels=in_channels, d_model=d_model,
            n_layers=n_encoder_layers, patch_size=patch_size,
            dilations=dilations, max_steps=max_steps_encoder,
            dt=dt, epsilon=epsilon, norm_type=norm_type,
            norm_every=norm_every, local_memory_hw=local_memory_hw,
            signed_diffusion=signed_diffusion,
            diffusion_scale=diffusion_scale,
        )

        evolve_steps = n_world_layers if n_world_layers > 0 else max_steps_world
        self.belief_field = BeliefField(
            channels=d_model, stimulus_dim=stimulus_dim,
            spatial_hw=belief_spatial_hw, decay=0.95,
            n_evolve_steps=evolve_steps,
            dilations=list(dilations[:2]),
        )

        # Predictor POOLED (backward compat for imagine())
        self.predictor = nn.Sequential(
            nn.Linear(d_model, pred_hidden),
            nn.GELU(),
            nn.LayerNorm(pred_hidden),
            nn.Linear(pred_hidden, d_model),
        )

        # Predictor SPATIAL (v9) -- Conv2d 1x1 bottleneck
        # Preserves spatial info; loss on (B, C, H', W') instead of (B, C).
        self.spatial_predictor = nn.Sequential(
            nn.Conv2d(d_model, pred_hidden, kernel_size=1),
            nn.GELU(),
            nn.GroupNorm(1, pred_hidden),  # LayerNorm equivalent for conv
            nn.Conv2d(pred_hidden, d_model, kernel_size=1),
        )

        self._init_weights()
        self.target_encoder = EMATargetEncoder(
            self.encoder, momentum=momentum
        )

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(self, x: torch.Tensor) -> Dict:
        return self.encoder(x)

    def imagine(
        self, z: torch.Tensor, stimulus: torch.Tensor,
        current_state: Optional[torch.Tensor] = None,
    ) -> Dict:
        if current_state is None:
            current_state = self.belief_field.init_state(
                z.shape[0], z.device, z.dtype)
        else:
            current_state = current_state.detach()
        state_updated = self.belief_field.write(current_state, z)
        next_state = self.belief_field.evolve(state_updated, stimulus=stimulus)
        z_pred_pooled = self.belief_field.read(next_state)
        z_pred_projected = self.predictor(z_pred_pooled)
        return {
            "prediction_pooled": z_pred_projected,
            "next_state": next_state.detach(),
        }

    def forward(
        self,
        x_current: torch.Tensor,
        stimulus: torch.Tensor,
        x_next: torch.Tensor,
        current_state: Optional[torch.Tensor] = None,
        var_weight: float = 5.0,
        cov_weight: float = 0.04,
        eq_weight: float = 0.5,
        eq_target: float = 0.5,
    ) -> Dict:
        B = x_current.shape[0]

        # 1. Encode x_t
        enc_out = self.encode(x_current)
        z_t = enc_out["features"]           # (B, C, H', W') - in graph
        encoder_info = enc_out["info"]

        # Pool z_t for direct VICReg on encoder (1 gradient hop)
        z_t_pooled = z_t.mean(dim=(-2, -1))  # (B, C) - in graph

        # 2. BeliefField
        if current_state is None:
            current_state = self.belief_field.init_state(
                B, x_current.device, x_current.dtype)
        else:
            current_state = current_state.detach()

        state_updated = self.belief_field.write(current_state, z_t)
        next_state = self.belief_field.evolve(state_updated, stimulus=stimulus)

        # 3. Spatial prediction (v9)
        # read_spatial returns (B, C, H_b, W_b) without pooling
        target_hw = (z_t.shape[2], z_t.shape[3])
        z_pred_spatial = self.belief_field.read_spatial(next_state, target_hw)
        z_pred_spatial = self.spatial_predictor(z_pred_spatial)  # (B, C, H', W')

        # 4. Spatial EMA target (no gradient)
        with torch.no_grad():
            target_out = self.target_encoder(x_next)
            z_target_spatial = target_out["features"]  # (B, C, H', W') -- no pooling

        # 5. LOSSES

        # (a) Spatial prediction MSE (v9) -- position-wise feature map comparison
        prediction_loss = F.mse_loss(z_pred_spatial, z_target_spatial)

        # (b) VICReg on z_t_pooled only (direct encoder gradient)
        vicreg_out = vicreg_loss(
            z_t_pooled,
            var_weight=var_weight,
            cov_weight=cov_weight,
        )

        # (c) PDE-Alive regularizer (v8) -- centers on target energy
        pde_alive_out = compute_pde_alive_loss(encoder_info, target_eq=eq_target)
        pde_alive_loss = pde_alive_out["pde_alive_loss"]

        # Total loss
        total_loss = (
            prediction_loss
            + vicreg_out["vicreg_total"]
            + eq_weight * pde_alive_loss
        )

        return {
            "loss": total_loss,
            "prediction_loss": prediction_loss,
            "var_loss": vicreg_out["var_loss"],
            "cov_loss": vicreg_out["cov_loss"],
            "vicreg_total": vicreg_out["vicreg_total"],
            "equilibrium_loss": pde_alive_loss,
            "mean_turbulence": pde_alive_out["mean_turbulence"],
            "mean_step_energy": pde_alive_out["mean_step_energy"],
            "next_state": next_state.detach(),
            "encoder_steps": [i["steps_used"] for i in encoder_info],
            "pred_std": vicreg_out.get("pred_std_mean", torch.tensor(0.0)),
            "encoder_std": z_t_pooled.float().std(dim=0).mean().detach(),
        }

    def update_target(self) -> None:
        self.target_encoder.update(self.encoder)

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}