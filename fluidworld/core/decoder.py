"""
decoder.py -- Pixel decoder: upsamples latent features to pixel space.
"""

import torch
import torch.nn as nn


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
    """
    Decodes latent features to pixel frames via Upsample + Conv + ResBlocks.

    Args:
        d_model: latent channel dimension (default 128)
        out_channels: output channels (1=grayscale, 3=RGB)
        mid_channels: intermediate channels (default 64)
    """

    def __init__(
        self,
        d_model: int = 128,
        out_channels: int = 1,
        mid_channels: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.out_channels = out_channels

        half_mid = mid_channels // 2

        self.decoder = nn.Sequential(
            # (B, 128, 16, 16) -> (B, 64, 16, 16) : project
            nn.Conv2d(d_model, mid_channels, kernel_size=1),
            nn.GELU(),

            # ResBlock at 16x16: learns spatial patterns in latent
            ResBlock(mid_channels),

            # (B, 64, 16, 16) -> (B, 64, 32, 32) : upsample 1
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.GroupNorm(8, mid_channels),

            # ResBlock at 32x32: refine mid-resolution details
            ResBlock(mid_channels),

            # (B, 64, 32, 32) -> (B, 32, 64, 64) : upsample 2
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(mid_channels, half_mid, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.GroupNorm(8, half_mid),

            # ResBlock at 64x64: fine details (edges, textures)
            ResBlock(half_mid),

            # (B, 32, 64, 64) -> (B, out_channels, 64, 64) : output
            nn.Conv2d(half_mid, out_channels, kernel_size=3, stride=1, padding=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, d_model, Hf, Wf) latent features

        Returns:
            (B, out_channels, H, W) logits (no sigmoid)
        """
        return self.decoder(z)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
