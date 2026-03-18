"""Image-oriented FluidVLA model components."""

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn

try:
    from .fluid_layer import FluidLayer2D, RMSNorm
except ImportError:
    from fluid_layer import FluidLayer2D, RMSNorm


class PatchEmbed(nn.Module):
    """
    Non-overlapping patch embedding.

    Intentionally simple:
      Conv2d with kernel=stride=patch_size
      followed by a light channel-space normalization.
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        patch_size: int = 4,
        norm_type: str = "rmsnorm",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

        norm_type = norm_type.lower()
        if norm_type == "rmsnorm":
            self.norm = RMSNorm(d_model)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(d_model)
        else:
            raise ValueError("norm_type must be 'rmsnorm' or 'layernorm'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class FluidBotClassifier(nn.Module):
    """Image classifier with stacked FluidLayer2D."""

    CONFIGS = {
        "tiny": dict(d_model=64, n_layers=2, dilations=[1, 4], max_steps=8, patch_size=4),
        "small": dict(d_model=128, n_layers=3, dilations=[1, 4, 8], max_steps=10, patch_size=4),
        "base": dict(d_model=256, n_layers=4, dilations=[1, 4, 16], max_steps=12, patch_size=4),
    }

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        d_model: int = 128,
        n_layers: int = 3,
        dilations: Sequence[int] = (1, 4, 16),
        max_steps: int = 12,
        dt: float = 0.1,
        epsilon: float = 0.08,
        patch_size: int = 4,
        use_pde: bool = True,
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

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            d_model=d_model,
            patch_size=patch_size,
            norm_type=norm_type,
        )

        self.fluid_layers = nn.ModuleList(
            [
                FluidLayer2D(
                    channels=d_model,
                    dilations=list(dilations),
                    max_steps=max_steps,
                    dt=dt,
                    epsilon=epsilon,
                    use_pde=use_pde,
                    norm_type=norm_type,
                    norm_every=norm_every,
                    local_memory_hw=local_memory_hw,
                    signed_diffusion=signed_diffusion,
                    diffusion_scale=diffusion_scale,
                    stop_patience=stop_patience,
                    min_steps=min_steps,
                    stop_probe_hw=stop_probe_hw,
                )
                for _ in range(n_layers)
            ]
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

        self._init_weights()

    @classmethod
    def from_config(cls, config_name: str, **kwargs):
        cfg = cls.CONFIGS[config_name].copy()
        cfg.update(kwargs)
        return cls(**cfg)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        u = self.patch_embed(x)

        all_info = []
        for layer in self.fluid_layers:
            u, info = layer(u)
            all_info.append(info)

        logits = self.head(u)

        avg_steps = sum(i["steps_used"] for i in all_info) / max(len(all_info), 1)
        avg_final_turb = sum(i["final_turbulence"] for i in all_info) / max(len(all_info), 1)
        avg_min_turb = sum(i["min_turbulence"] for i in all_info) / max(len(all_info), 1)

        return logits, {
            "avg_steps": avg_steps,
            "avg_final_turbulence": avg_final_turb,
            "avg_min_turbulence": avg_min_turb,
            "layer_steps": [i["steps_used"] for i in all_info],
            "pde_active": all_info[0]["pde_active"] if all_info else True,
        }

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        return {
            "total": total,
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
