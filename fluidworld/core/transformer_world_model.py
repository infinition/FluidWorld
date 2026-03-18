"""
transformer_world_model.py -- Transformer baseline for PDE vs attention comparison.
"""

from typing import Dict, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Transformer Block (standard pre-norm)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm Transformer block : LayerNorm -> MHSA -> LayerNorm -> FFN."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )
        if dropout > 0:
            self.drop = nn.Dropout(dropout)
        else:
            self.drop = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with pre-norm
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop(attn_out)
        # FFN with pre-norm
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Simple PatchEmbed (same as FluidWorld)
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Conv2d patch embedding : (B, C, H, W) -> (B, d_model, H/p, W/p)."""

    def __init__(self, in_channels: int, d_model: int, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, d_model,
            kernel_size=patch_size, stride=patch_size,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, d_model, H/p, W/p)
        x = self.proj(x)
        B, C, H, W = x.shape
        # Normalize per-token
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)  # (B, C, H, W)
        return x


# ---------------------------------------------------------------------------
# Pixel Decoder (identical to FluidWorld)
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Residual block: Conv3x3 + GELU + Conv3x3 + skip."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GELU(),
            nn.GroupNorm(8, channels),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class PixelDecoder(nn.Module):
    """Pixel decoder identical to FluidWorld for fair comparison."""

    def __init__(self, d_model: int = 128, out_channels: int = 3, mid_channels: int = 64):
        super().__init__()
        half_mid = mid_channels // 2
        self.decoder = nn.Sequential(
            nn.Conv2d(d_model, mid_channels, kernel_size=1),
            nn.GELU(),
            ResBlock(mid_channels),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
            nn.GELU(),
            nn.GroupNorm(8, mid_channels),
            ResBlock(mid_channels),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(mid_channels, half_mid, 3, 1, 1),
            nn.GELU(),
            nn.GroupNorm(8, half_mid),
            ResBlock(half_mid),
            nn.Conv2d(half_mid, out_channels, 3, 1, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


# ---------------------------------------------------------------------------
# Gradient Loss (identical to FluidWorld)
# ---------------------------------------------------------------------------

def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 loss on spatial gradients (edges/contours)."""
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


# ---------------------------------------------------------------------------
# TransformerWorldModel
# ---------------------------------------------------------------------------

class TransformerWorldModel(nn.Module):
    """
    Transformer baseline for comparison with FluidWorldModelV2.
    Same encode/decode pipeline; temporal engine is self-attention instead of PDE.
    """

    def __init__(
        self,
        in_channels: int = 3,
        d_model: int = 128,
        n_encoder_layers: int = 2,
        n_temporal_layers: int = 1,
        n_heads: int = 8,
        ffn_dim: int = 384,
        patch_size: int = 4,
        spatial_hw: int = 16,
        recon_weight: float = 1.0,
        pred_weight: float = 1.0,
        var_weight: float = 0.0,
        var_target: float = 1.0,
        grad_weight: float = 0.0,
        decoder_mid_channels: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.spatial_hw = spatial_hw
        self.n_tokens = spatial_hw * spatial_hw  # 256
        self.recon_weight = recon_weight
        self.pred_weight = pred_weight
        self.var_weight = var_weight
        self.var_target = var_target
        self.grad_weight = grad_weight

        # --- Encoder ---
        self.patch_embed = PatchEmbed(in_channels, d_model, patch_size)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_tokens, d_model) * 0.02
        )
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # --- Temporal prediction ---
        # Merge observation + state -> predict next state
        self.state_merge = nn.Linear(d_model * 2, d_model)
        self.temporal_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_temporal_layers)
        ])
        self.temporal_norm = nn.LayerNorm(d_model)

        # --- Decoder ---
        self.decoder = PixelDecoder(d_model, in_channels, decoder_mid_channels)

        self._init_weights()

    def _init_weights(self):
        # Initialize Transformer weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _to_tokens(self, z_spatial: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, N, C)"""
        B, C, H, W = z_spatial.shape
        return z_spatial.flatten(2).transpose(1, 2)

    def _to_spatial(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, N, C) -> (B, C, H, W)"""
        B, N, C = tokens.shape
        H = W = int(math.sqrt(N))
        return tokens.transpose(1, 2).reshape(B, C, H, W)

    def encode(self, x: torch.Tensor) -> Dict:
        """Encode frame -> spatial features (B, d_model, H, W)."""
        z = self.patch_embed(x)  # (B, d_model, 16, 16)
        tokens = self._to_tokens(z) + self.pos_embed  # (B, 256, 128)

        for block in self.encoder_blocks:
            tokens = block(tokens)
        tokens = self.encoder_norm(tokens)

        features = self._to_spatial(tokens)  # (B, 128, 16, 16)
        return {"features": features}

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode features -> pixel logits."""
        return self.decoder(z)

    def decode_to_pixels(self, z: torch.Tensor) -> torch.Tensor:
        """Decode features -> pixel values [0, 1]."""
        return torch.sigmoid(self.decoder(z))

    def forward(
        self,
        x_current: torch.Tensor,
        stimulus: torch.Tensor,
        x_next: torch.Tensor,
        current_state: Optional[torch.Tensor] = None,
        eq_weight: float = 0.0,
        eq_target: float = 1.2,
    ) -> Dict:
        """
        Forward : encode, reconstruct, predict, compute losses.
        Same API as FluidWorldModelV2 for drop-in comparison.
        """
        B = x_current.shape[0]

        # 1. Encode
        enc_out = self.encode(x_current)
        z_t = enc_out["features"]  # (B, 128, 16, 16)

        # 2. Reconstruct
        x_recon_logits = self.decode(z_t)
        recon_loss = F.mse_loss(torch.sigmoid(x_recon_logits), x_current)

        # 3. Temporal prediction via Transformer
        if current_state is None:
            current_state = torch.zeros_like(z_t)

        # Merge observation + state
        z_tokens = self._to_tokens(z_t)              # (B, 256, 128)
        s_tokens = self._to_tokens(current_state)     # (B, 256, 128)
        merged = self.state_merge(
            torch.cat([z_tokens, s_tokens], dim=-1)   # (B, 256, 256)
        )  # (B, 256, 128)
        merged = merged + self.pos_embed

        for block in self.temporal_blocks:
            merged = block(merged)
        merged = self.temporal_norm(merged)

        next_state = self._to_spatial(merged)  # (B, 128, 16, 16)

        # 4. Decode prediction
        x_pred_logits = self.decode(next_state)
        pred_loss = F.mse_loss(torch.sigmoid(x_pred_logits), x_next)

        # 5. Variance loss
        var_loss = torch.tensor(0.0, device=x_current.device)
        if self.var_weight > 0:
            z_flat = z_t.permute(1, 0, 2, 3).reshape(z_t.shape[1], -1)
            channel_std = z_flat.std(dim=1)
            var_loss = torch.clamp(self.var_target - channel_std, min=0).mean()

        # 6. Gradient loss
        grad_loss = torch.tensor(0.0, device=x_current.device)
        if self.grad_weight > 0:
            x_recon_pixels = torch.sigmoid(x_recon_logits)
            x_pred_pixels = torch.sigmoid(x_pred_logits)
            grad_loss = (
                gradient_loss(x_recon_pixels, x_current)
                + gradient_loss(x_pred_pixels, x_next)
            ) * 0.5

        # 7. Total loss
        total_loss = (
            self.recon_weight * recon_loss
            + self.pred_weight * pred_loss
            + self.var_weight * var_loss
            + self.grad_weight * grad_loss
        )

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "pred_loss": pred_loss,
            "pde_alive_loss": torch.tensor(0.0, device=x_current.device),
            "var_loss": var_loss,
            "grad_loss": grad_loss,
            "deep_loss": torch.tensor(0.0, device=x_current.device),
            "rdm_loss": torch.tensor(0.0, device=x_current.device),
            "mean_turbulence": 0.0,
            "mean_step_energy": 0.0,
            "next_state": next_state.detach(),
            "x_recon": torch.sigmoid(x_recon_logits).detach(),
            "x_pred": torch.sigmoid(x_pred_logits).detach(),
            "bio_stats": {},
            "gate_mean": 1.0,
            "gate_std": 0.0,
        }

    @torch.no_grad()
    def rollout(
        self,
        x_init: torch.Tensor,
        stimulus: torch.Tensor,
        n_steps: int = 10,
    ) -> torch.Tensor:
        """
        Autoregressive rollout. Same API as FluidWorldModelV2.
        Returns (B, n_steps, C, H, W).
        """
        frames = []
        z = self.encode(x_init)["features"]
        state = torch.zeros_like(z)

        for _ in range(n_steps):
            # Merge + temporal predict
            z_tokens = self._to_tokens(z)
            s_tokens = self._to_tokens(state)
            merged = self.state_merge(
                torch.cat([z_tokens, s_tokens], dim=-1)
            )
            merged = merged + self.pos_embed

            for block in self.temporal_blocks:
                merged = block(merged)
            merged = self.temporal_norm(merged)

            state = self._to_spatial(merged)
            frame = self.decode_to_pixels(state)
            frames.append(frame)

            # Autoregressive: use prediction as next input
            z = state

        return torch.stack(frames, dim=1)

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_p = (
            sum(p.numel() for p in self.patch_embed.parameters())
            + sum(p.numel() for p in self.encoder_blocks.parameters())
            + sum(p.numel() for p in self.encoder_norm.parameters())
            + self.pos_embed.numel()
        )
        temporal_p = (
            sum(p.numel() for p in self.state_merge.parameters())
            + sum(p.numel() for p in self.temporal_blocks.parameters())
            + sum(p.numel() for p in self.temporal_norm.parameters())
        )
        decoder_p = sum(p.numel() for p in self.decoder.parameters())
        return {
            "total": total,
            "trainable": trainable,
            "encoder": encoder_p,
            "temporal": temporal_p,
            "decoder": decoder_p,
        }
