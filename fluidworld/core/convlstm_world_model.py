"""
convlstm_world_model.py -- ConvLSTM baseline for scaling experiments.
"""

from typing import Dict, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ConvLSTM Cell
# ---------------------------------------------------------------------------

class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell (Shi et al., 2015). Replaces LSTM matmuls with 2D convolutions."""

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        # Single conv for all 4 gates : i, f, o, g
        self.gates = nn.Conv2d(
            input_dim + hidden_dim, 4 * hidden_dim,
            kernel_size=kernel_size, padding=padding, bias=True,
        )
        self._init_weights()

    def _init_weights(self):
        # Xavier init for gates, bias forget gate = 1 (encourage remembering)
        nn.init.xavier_uniform_(self.gates.weight)
        if self.gates.bias is not None:
            nn.init.zeros_(self.gates.bias)
            # Set forget gate bias to 1
            n = self.hidden_dim
            self.gates.bias.data[n:2*n].fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, input_dim, H, W)
            state: (h, c) each (B, hidden_dim, H, W)
        Returns:
            (h_new, c_new) each (B, hidden_dim, H, W)
        """
        h, c = state
        combined = torch.cat([x, h], dim=1)  # (B, input+hidden, H, W)
        gates = self.gates(combined)          # (B, 4*hidden, H, W)

        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

    def init_state(
        self, batch_size: int, spatial_size: Tuple[int, int], device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        H, W = spatial_size
        return (
            torch.zeros(batch_size, self.hidden_dim, H, W, device=device),
            torch.zeros(batch_size, self.hidden_dim, H, W, device=device),
        )


# ---------------------------------------------------------------------------
# Simple PatchEmbed (same as FluidWorld / Transformer)
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
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x


# ---------------------------------------------------------------------------
# Pixel Decoder (identical to FluidWorld / Transformer)
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Residual block: Conv3x3 + GELU + GroupNorm + Conv3x3 + skip."""

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
    """Pixel decoder identical to FluidWorld/Transformer for fair comparison."""

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
# Gradient Loss (identical to FluidWorld / Transformer)
# ---------------------------------------------------------------------------

def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 loss on spatial gradients (edges/contours)."""
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


# ---------------------------------------------------------------------------
# ConvLSTMWorldModel
# ---------------------------------------------------------------------------

class ConvLSTMWorldModel(nn.Module):
    """
    ConvLSTM baseline for comparison with FluidWorldModelV2 and Transformer.
    Same encode/decode pipeline; temporal engine is ConvLSTM instead of PDE or attention.
    """

    def __init__(
        self,
        in_channels: int = 3,
        d_model: int = 128,
        convlstm_hidden: int = 64,
        convlstm_kernel: int = 3,
        patch_size: int = 4,
        spatial_hw: int = 16,
        recon_weight: float = 1.0,
        pred_weight: float = 1.0,
        var_weight: float = 0.0,
        var_target: float = 1.0,
        grad_weight: float = 0.0,
        decoder_mid_channels: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.spatial_hw = spatial_hw
        self.convlstm_hidden = convlstm_hidden
        self.recon_weight = recon_weight
        self.pred_weight = pred_weight
        self.var_weight = var_weight
        self.var_target = var_target
        self.grad_weight = grad_weight

        # --- Encoder ---
        self.patch_embed = PatchEmbed(in_channels, d_model, patch_size)

        # Spatial processing : bottleneck conv (128 -> enc_mid -> 128)
        # Dimension enc_mid is computed to match ~800K total params
        enc_mid = 88  # tuned for param matching (~800K total)
        self.encoder = nn.Sequential(
            nn.Conv2d(d_model, enc_mid, 3, 1, 1),
            nn.GELU(),
            nn.GroupNorm(8, enc_mid),
            nn.Conv2d(enc_mid, d_model, 1),
            nn.GELU(),
        )

        # --- Temporal : ConvLSTM ---
        self.convlstm = ConvLSTMCell(d_model, convlstm_hidden, convlstm_kernel)
        self.output_proj = nn.Sequential(
            nn.Conv2d(convlstm_hidden, d_model, 1),
            nn.GroupNorm(8, d_model),
        )

        # --- Decoder ---
        self.decoder = PixelDecoder(d_model, in_channels, decoder_mid_channels)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and not hasattr(m, '_custom_init'):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> Dict:
        """Encode frame -> spatial features (B, d_model, H, W)."""
        z = self.patch_embed(x)           # (B, 128, 16, 16)
        z = z + self.encoder(z)           # residual spatial processing
        return {"features": z}

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
        current_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        eq_weight: float = 0.0,
        eq_target: float = 1.2,
    ) -> Dict:
        """
        Forward : encode, reconstruct, predict, compute losses.
        Same API as FluidWorldModelV2 / TransformerWorldModel for drop-in comparison.

        current_state: (h, c) tuple for ConvLSTM, or None for first frame.
        """
        B = x_current.shape[0]

        # 1. Encode
        enc_out = self.encode(x_current)
        z_t = enc_out["features"]  # (B, 128, 16, 16)

        # 2. Reconstruct
        x_recon_logits = self.decode(z_t)
        recon_loss = F.mse_loss(torch.sigmoid(x_recon_logits), x_current)

        # 3. Temporal prediction via ConvLSTM
        if current_state is None:
            current_state = self.convlstm.init_state(
                B, (self.spatial_hw, self.spatial_hw), z_t.device,
            )

        h_new, c_new = self.convlstm(z_t, current_state)

        # Project hidden state back to d_model channels
        pred_features = self.output_proj(h_new)  # (B, 128, 16, 16)

        # 4. Decode prediction
        x_pred_logits = self.decode(pred_features)
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
            "next_state": (h_new.detach(), c_new.detach()),
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
        Autoregressive rollout. Same API as FluidWorldModelV2 / Transformer.
        Returns (B, n_steps, C, H, W).
        """
        frames = []
        z = self.encode(x_init)["features"]
        B = x_init.shape[0]
        state = self.convlstm.init_state(
            B, (self.spatial_hw, self.spatial_hw), x_init.device,
        )

        for _ in range(n_steps):
            h, c = self.convlstm(z, state)
            state = (h, c)

            pred_features = self.output_proj(h)
            frame = self.decode_to_pixels(pred_features)
            frames.append(frame)

            # Autoregressive: use prediction as next input
            z = pred_features

        return torch.stack(frames, dim=1)

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_p = (
            sum(p.numel() for p in self.patch_embed.parameters())
            + sum(p.numel() for p in self.encoder.parameters())
        )
        temporal_p = (
            sum(p.numel() for p in self.convlstm.parameters())
            + sum(p.numel() for p in self.output_proj.parameters())
        )
        decoder_p = sum(p.numel() for p in self.decoder.parameters())
        return {
            "total": total,
            "trainable": trainable,
            "encoder": encoder_p,
            "temporal": temporal_p,
            "decoder": decoder_p,
        }
