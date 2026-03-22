"""
world_model_v2.py - FluidWorld v2: pixel-prediction world model (replaces JEPA/VICReg approach).
"""

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._fluidvla_imports import FluidLayer2D, PatchEmbed
from .fluid_world_layer import FluidWorldLayer2D, compute_pde_alive_loss
from .belief_field import BeliefField
from .decoder import PixelDecoder
from .bio_mechanisms import SynapticFatigue, LateralInhibition


def _image_gradients(x: torch.Tensor):
    """Finite-difference spatial gradients of an image tensor (B, C, H, W)."""
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dy = F.pad(dy, (0, 0, 0, 1))
    return dx, dy


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 loss on image gradients — forces edge/structure reconstruction.
    Breaks the 'mean color' local minimum that MSE converges to.
    """
    pred_dx, pred_dy = _image_gradients(pred)
    tgt_dx, tgt_dy = _image_gradients(target)
    return F.l1_loss(pred_dx, tgt_dx) + F.l1_loss(pred_dy, tgt_dy)


def rdm_reg_loss(
    features: torch.Tensor,
    n_projections: int = 64,
    mu: float = 0.0,
    sigma: float = 1.0,
    p: float = 2.0,
) -> torch.Tensor:
    """Rectified Distribution Matching Regularization (RDMReg).

    Inspired by Kuang et al. (arXiv:2602.01456, Feb 2026).
    Aligns features to a Rectified Generalized Gaussian distribution
    for sparse, non-negative, max-entropy representations.

    This is a simplified sliced Wasserstein variant:
    - Project features onto random directions
    - ReLU to ensure non-negativity (rectification)
    - Match sorted marginals to RGG quantiles

    Args:
        features: (B, C, H, W) or (B, C) — feature tensor
        n_projections: number of random projection directions
        mu, sigma: RGG distribution parameters
        p: shape parameter (2=Gaussian, 1=Laplace)
    """
    # Flatten features to (N, C)
    if features.dim() == 4:
        N = features.shape[0] * features.shape[2] * features.shape[3]
        C = features.shape[1]
        z = features.permute(0, 2, 3, 1).reshape(N, C)
    else:
        z = features
        N, C = z.shape

    # Rectify (non-negative representations like biological neurons)
    z_rect = F.relu(z)

    # Random projection directions (unit vectors on sphere)
    directions = torch.randn(n_projections, C, device=z.device, dtype=z.dtype)
    directions = F.normalize(directions, dim=1)

    # Project features onto random directions: (n_proj, N)
    projections = torch.mm(directions, z_rect.t())  # (n_proj, N)

    # Sort projected features
    proj_sorted, _ = torch.sort(projections, dim=1)

    # Generate target quantiles from Rectified Generalized Gaussian
    # For p=2 (Gaussian): sample from N(mu, sigma), then ReLU
    # Approximation: use uniform quantiles through the RGG CDF
    quantiles = torch.linspace(0.01, 0.99, N, device=z.device, dtype=z.dtype)

    if p == 2.0:
        # Rectified Gaussian quantiles: max(0, N(mu, sigma).icdf(q))
        # N(0,1).icdf(q) = sqrt(2) * erfinv(2q - 1)
        normal_quantiles = mu + sigma * torch.erfinv(2 * quantiles - 1) * (2 ** 0.5)
        target_quantiles = F.relu(normal_quantiles)
    else:
        # Fallback: Rectified Laplace (p=1) or general
        # Approximate with exponential quantiles (sparse)
        target_quantiles = F.relu(-sigma * torch.log(2 * (1 - quantiles)))

    # Expand for all projections
    target_quantiles = target_quantiles.unsqueeze(0).expand(n_projections, -1)

    # Sliced Wasserstein distance (L2 between sorted marginals)
    loss = F.mse_loss(proj_sorted, target_quantiles)
    return loss


class OnlineEncoder(nn.Module):
    """PDE encoder with optional deep self-supervision (inspired by V-JEPA 2.1).

    Deep supervision applies the loss at each intermediate PDE layer,
    not just the final output. This prevents intermediate layers from
    becoming global aggregators and ensures spatial information flows
    through all layers (Mur-Labadia et al., V-JEPA 2.1, Mar 2026).
    """

    def __init__(
        self,
        in_channels: int = 1,
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
        anisotropic_diffusion: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_embed = PatchEmbed(
            in_channels=in_channels, d_model=d_model,
            patch_size=patch_size, norm_type=norm_type,
        )

        # Use FluidWorldLayer2D when anisotropic diffusion is enabled
        # (it supports content-gated diffusion), otherwise use FluidLayer2D
        # (lighter, from FluidVLA). Both are compatible (action=None).
        LayerClass = FluidWorldLayer2D if anisotropic_diffusion else FluidLayer2D
        layer_kwargs = dict(
            channels=d_model, dilations=list(dilations),
            max_steps=max_steps, dt=dt, epsilon=epsilon,
            use_pde=True, norm_type=norm_type, norm_every=norm_every,
            local_memory_hw=local_memory_hw,
            signed_diffusion=signed_diffusion,
            diffusion_scale=diffusion_scale,
            stop_patience=stop_patience, min_steps=min_steps,
            stop_probe_hw=stop_probe_hw,
        )
        if anisotropic_diffusion:
            layer_kwargs["anisotropic_diffusion"] = True
            layer_kwargs["action_dim"] = 0  # no ActionForce in encoder

        self.fluid_layers = nn.ModuleList([
            LayerClass(**layer_kwargs) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, deep_supervision: bool = False) -> Dict:
        u = self.patch_embed(x)
        u_spatial = u  # preserve spatial detail before PDE diffusion
        info_list = []
        intermediate_features = []  # for deep supervision (V-JEPA 2.1)

        for layer in self.fluid_layers:
            u, info = layer(u)
            info_list.append(info)
            if deep_supervision:
                # Store intermediate features WITH spatial skip
                # Each layer's output + u_spatial gives a decodable representation
                intermediate_features.append(u + u_spatial)

        # Residual: PDE output (global context) + PatchEmbed (spatial detail)
        # Without this, PDE diffusion homogenizes all 16x16 positions → blobs
        u = u + u_spatial
        result = {"features": u, "info": info_list}
        if deep_supervision:
            result["intermediate_features"] = intermediate_features
        return result


class FluidWorldModelV2(nn.Module):
    """
    Pixel-prediction world model. Encoder PDE -> BeliefField -> Decoder.
    Losses: reconstruction (anchor), prediction (world model), PDE-Alive (regularizer).
    """

    def __init__(
        self,
        in_channels: int = 1,
        d_model: int = 128,
        stimulus_dim: int = 1,
        n_encoder_layers: int = 3,
        patch_size: int = 4,
        dilations: Sequence[int] = (1, 4, 16),
        max_steps_encoder: int = 6,
        dt: float = 0.1,
        epsilon: float = 0.08,
        norm_type: str = "rmsnorm",
        norm_every: int = 2,
        local_memory_hw: int = 4,
        signed_diffusion: bool = False,
        diffusion_scale: float = 0.25,
        belief_spatial_hw: int = 16,
        n_belief_evolve: int = 3,
        decoder_mid_channels: int = 64,
        recon_weight: float = 1.0,
        pred_weight: float = 1.0,
        loss_type: str = "auto",
        var_weight: float = 0.0,
        var_target: float = 1.0,
        grad_weight: float = 0.0,
        # Phase 1.5: JEPA-inspired improvements (Mar 2026)
        deep_supervision: bool = False,      # V-JEPA 2.1: loss on intermediate PDE layers
        deep_supervision_weight: float = 0.3,
        rdm_reg: bool = False,               # Rectified LpJEPA: sparse distribution matching
        rdm_weight: float = 0.1,
        rdm_n_projections: int = 64,
        input_masking: bool = False,          # C-JEPA inspired: mask input patches
        mask_ratio: float = 0.25,            # fraction of patches to mask
        # Phase 1.5+: Anisotropic diffusion (content-gated routing)
        anisotropic_diffusion: bool = False,  # Content-gated diffusion gate
        # Bio mechanisms (RESEARCH.md)
        use_fatigue: bool = True,
        fatigue_cost: float = 0.1,
        fatigue_recovery: float = 0.02,
        use_inhibition: bool = True,
        inhibition_strength: float = 0.3,
        use_memory_pump: bool = True,
        use_hebbian: bool = True,
        hebbian_lr: float = 0.01,
        hebbian_decay: float = 0.99,
        use_deltanet: bool = True,
        use_titans: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.in_channels = in_channels
        self.recon_weight = recon_weight
        self.pred_weight = pred_weight

        # Loss: "bce" for binary images (MNIST), "mse" for RGB video
        # "auto" picks based on in_channels
        if loss_type == "auto":
            self.loss_type = "bce" if in_channels == 1 else "mse"
        else:
            self.loss_type = loss_type

        # Variance loss (RESEARCH.md §7): force each feature dim to be active.
        # Only the variance term — NOT covariance (which destroyed spatial info in v4-v9).
        # Directly fights dead dims by penalizing std < var_target per channel.
        self.var_weight = var_weight
        self.var_target = var_target
        self.grad_weight = grad_weight

        # Phase 1.5: JEPA-inspired improvements
        self.deep_supervision = deep_supervision
        self.deep_supervision_weight = deep_supervision_weight
        self.rdm_reg = rdm_reg
        self.rdm_weight = rdm_weight
        self.rdm_n_projections = rdm_n_projections
        self.input_masking = input_masking
        self.mask_ratio = mask_ratio

        # ── PDE Encoder (proven at 91% on FluidVLA) ──
        self.encoder = OnlineEncoder(
            in_channels=in_channels, d_model=d_model,
            n_layers=n_encoder_layers, patch_size=patch_size,
            dilations=dilations, max_steps=max_steps_encoder,
            dt=dt, epsilon=epsilon, norm_type=norm_type,
            norm_every=norm_every, local_memory_hw=local_memory_hw,
            signed_diffusion=signed_diffusion,
            diffusion_scale=diffusion_scale,
            anisotropic_diffusion=anisotropic_diffusion,
        )

        # ── Bio: Synaptic Fatigue (RESEARCH.md #14) ──
        self.use_fatigue = use_fatigue
        if use_fatigue:
            self.fatigue = SynapticFatigue(
                channels=d_model,
                cost=fatigue_cost,
                recovery=fatigue_recovery,
            )

        # ── Bio: Lateral Inhibition (RESEARCH.md #16) ──
        self.use_inhibition = use_inhibition
        if use_inhibition:
            self.inhibition = LateralInhibition(
                strength=inhibition_strength,
            )

        # ── BeliefField (persistent world memory) ──
        self.belief_field = BeliefField(
            channels=d_model, stimulus_dim=stimulus_dim,
            spatial_hw=belief_spatial_hw, decay=0.95,
            n_evolve_steps=n_belief_evolve,
            dilations=list(dilations[:2]),
            use_memory_pump=use_memory_pump,
            use_hebbian=use_hebbian,
            hebbian_lr=hebbian_lr,
            hebbian_decay=hebbian_decay,
            use_deltanet=use_deltanet,
            use_titans=use_titans,
        )

        # ── Pixel Decoder ──
        self.decoder = PixelDecoder(
            d_model=d_model,
            out_channels=in_channels,
            mid_channels=decoder_mid_channels,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier for Linear, Kaiming for Conv."""
        for name, module in self.named_modules():
            # Skip decoder (has its own init)
            if name.startswith("decoder"):
                continue
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _apply_input_masking(self, x: torch.Tensor) -> torch.Tensor:
        """Mask random patches of the input (C-JEPA inspired).

        Forces the model to infer missing regions from context,
        testing true world model capability.
        """
        if not self.training or not self.input_masking:
            return x
        B, C, H, W = x.shape
        patch_size = 4  # match PatchEmbed
        nH, nW = H // patch_size, W // patch_size
        n_patches = nH * nW
        n_mask = int(n_patches * self.mask_ratio)

        # Random mask per sample
        x_masked = x.clone()
        for b in range(B):
            indices = torch.randperm(n_patches, device=x.device)[:n_mask]
            for idx in indices:
                ph, pw = idx // nW, idx % nW
                h_start, w_start = ph * patch_size, pw * patch_size
                x_masked[b, :, h_start:h_start+patch_size,
                         w_start:w_start+patch_size] = 0.0
        return x_masked

    def encode(self, x: torch.Tensor) -> Dict:
        """Encode a frame to latent features with bio-mechanisms."""
        # Optional input masking (C-JEPA inspired)
        x_enc = self._apply_input_masking(x)

        out = self.encoder(x_enc, deep_supervision=self.deep_supervision)
        z = out["features"]

        # Lateral Inhibition: strong channels suppress weak ones
        if self.use_inhibition:
            z = self.inhibition(z)

        # Synaptic Fatigue: overactive channels are depleted
        if self.use_fatigue:
            z = self.fatigue(z)

        out["features"] = z
        return out

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent features to logits (no sigmoid)."""
        return self.decoder(z)

    def decode_to_pixels(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent features to pixel frame [0, 1]."""
        return torch.sigmoid(self.decoder(z))

    def forward(
        self,
        x_current: torch.Tensor,
        stimulus: torch.Tensor,
        x_next: torch.Tensor,
        current_state: Optional[torch.Tensor] = None,
        eq_weight: float = 0.5,
        eq_target: float = 1.2,
    ) -> Dict:
        """
        Full forward: encode, predict, decode, compute losses.

        Args:
            x_current: (B, C, H, W) current frame
            stimulus: (B, D) stimulus/action
            x_next: (B, C, H, W) next frame (ground truth)
            current_state: (B, d, Hb, Wb) BeliefField state
            eq_weight: PDE-Alive loss weight
            eq_target: target PDE energy

        Returns:
            dict with loss, reconstructions, predictions, metrics
        """
        B = x_current.shape[0]

        # 1. Encode current frame
        enc_out = self.encode(x_current)
        z_t = enc_out["features"]       # (B, d, Hf, Wf)
        encoder_info = enc_out["info"]

        # 2. Reconstruction of frame_t (anchor)
        x_recon_logits = self.decode(z_t)      # (B, C, H, W) logits/values
        if self.loss_type == "bce":
            recon_loss = F.binary_cross_entropy_with_logits(x_recon_logits, x_current)
        else:
            recon_loss = F.mse_loss(torch.sigmoid(x_recon_logits), x_current)

        # 3. BeliefField: integrate z_t and evolve
        if current_state is None:
            current_state = self.belief_field.init_state(
                B, x_current.device, x_current.dtype)
        else:
            current_state = current_state.detach()

        state_updated = self.belief_field.write(current_state, z_t)
        next_state = self.belief_field.evolve(state_updated, stimulus=stimulus)

        # 4. Prediction: read future state and decode
        target_hw = (z_t.shape[2], z_t.shape[3])
        z_pred = self.belief_field.read_spatial(next_state, target_hw)
        x_pred_logits = self.decode(z_pred)    # (B, C, H, W) logits/values
        if self.loss_type == "bce":
            pred_loss = F.binary_cross_entropy_with_logits(x_pred_logits, x_next)
        else:
            pred_loss = F.mse_loss(torch.sigmoid(x_pred_logits), x_next)

        # 5. PDE-Alive regularizer
        pde_alive_out = compute_pde_alive_loss(encoder_info, target_eq=eq_target)
        pde_alive_loss = pde_alive_out["pde_alive_loss"]

        # 5b. Variance loss (RESEARCH.md §7 — variance term only)
        # Force each channel to have std >= var_target. Wakes up dead dims.
        var_loss = torch.tensor(0.0, device=x_current.device)
        if self.var_weight > 0:
            # z_t: (B, C, H, W) — compute std per channel across batch+spatial
            z_flat = z_t.permute(1, 0, 2, 3).reshape(z_t.shape[1], -1)  # (C, B*H*W)
            channel_std = z_flat.std(dim=1)  # (C,)
            var_loss = torch.clamp(self.var_target - channel_std, min=0).mean()

        # 5c. Gradient loss — forces edge/structure reconstruction
        # Breaks the 'predict mean color' local minimum of MSE
        grad_loss = torch.tensor(0.0, device=x_current.device)
        if self.grad_weight > 0:
            x_recon_pixels = torch.sigmoid(x_recon_logits)
            x_pred_pixels = torch.sigmoid(x_pred_logits)
            grad_loss = (
                gradient_loss(x_recon_pixels, x_current)
                + gradient_loss(x_pred_pixels, x_next)
            ) * 0.5  # average over recon + pred

        # 5d. Deep PDE Supervision (V-JEPA 2.1 inspired)
        # Supervise intermediate PDE layers, not just the final output.
        # Forces each PDE layer to maintain spatial information.
        deep_loss = torch.tensor(0.0, device=x_current.device)
        if self.deep_supervision and "intermediate_features" in enc_out:
            n_layers = len(enc_out["intermediate_features"])
            for i, inter_feat in enumerate(enc_out["intermediate_features"]):
                inter_logits = self.decode(inter_feat)
                if self.loss_type == "bce":
                    layer_loss = F.binary_cross_entropy_with_logits(
                        inter_logits, x_current)
                else:
                    layer_loss = F.mse_loss(
                        torch.sigmoid(inter_logits), x_current)
                # Weight: earlier layers get less weight (they are less refined)
                layer_weight = (i + 1) / n_layers
                deep_loss = deep_loss + layer_weight * layer_loss
            deep_loss = deep_loss / n_layers

        # 5e. RDMReg — Rectified Distribution Matching (Rectified LpJEPA)
        # Aligns features to a Rectified Generalized Gaussian distribution
        # for sparse, non-negative representations (biologically plausible).
        rdm_loss_val = torch.tensor(0.0, device=x_current.device)
        if self.rdm_reg and self.rdm_weight > 0:
            rdm_loss_val = rdm_reg_loss(
                z_t, n_projections=self.rdm_n_projections)

        # 6. Total loss
        total_loss = (
            self.recon_weight * recon_loss
            + self.pred_weight * pred_loss
            + eq_weight * pde_alive_loss
            + self.var_weight * var_loss
            + self.grad_weight * grad_loss
            + self.deep_supervision_weight * deep_loss
            + self.rdm_weight * rdm_loss_val
        )

        # Bio mechanism stats for monitoring
        bio_stats = {}
        if self.use_fatigue:
            bio_stats.update(self.fatigue.get_stats())
        if hasattr(self.belief_field, 'hebbian') and self.belief_field.use_hebbian:
            bio_stats.update(self.belief_field.hebbian.get_stats())

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "pred_loss": pred_loss,
            "pde_alive_loss": pde_alive_loss,
            "var_loss": var_loss,
            "grad_loss": grad_loss,
            "deep_loss": deep_loss,
            "rdm_loss": rdm_loss_val,
            "mean_turbulence": pde_alive_out["mean_turbulence"],
            "mean_step_energy": pde_alive_out["mean_step_energy"],
            "next_state": next_state.detach(),
            "encoder_steps": [i["steps_used"] for i in encoder_info],
            # Images for TensorBoard (sigmoid + detach for [0, 1] pixels)
            "x_recon": torch.sigmoid(x_recon_logits).detach(),
            "x_pred": torch.sigmoid(x_pred_logits).detach(),
            # Live (non-detached) pixel outputs for external loss computation
            "x_recon_live": torch.sigmoid(x_recon_logits),
            "x_pred_live": torch.sigmoid(x_pred_logits),
            # Bio mechanisms stats
            "bio_stats": bio_stats,
            # Anisotropic diffusion gate stats
            "gate_mean": sum(i.get("gate_mean", 1.0) for i in encoder_info) / max(len(encoder_info), 1),
            "gate_std": sum(i.get("gate_std", 0.0) for i in encoder_info) / max(len(encoder_info), 1),
        }

    @torch.no_grad()
    def rollout(
        self,
        x_init: torch.Tensor,
        stimulus: torch.Tensor,
        n_steps: int = 10,
    ) -> torch.Tensor:
        """Autoregressive rollout of n_steps future frames.

        Args:
            x_init: (B, C, H, W) initial frame
            stimulus: (B, D) constant stimulus
            n_steps: number of frames to generate

        Returns:
            (B, n_steps, C, H, W) predicted frame sequence
        """
        B = x_init.shape[0]
        frames = []

        # Encode initial frame
        z = self.encode(x_init)["features"]
        state = self.belief_field.init_state(B, x_init.device, x_init.dtype)
        state = self.belief_field.write(state, z)

        for _ in range(n_steps):
            state = self.belief_field.evolve(state, stimulus=stimulus)
            target_hw = (z.shape[2], z.shape[3])
            z_pred = self.belief_field.read_spatial(state, target_hw)
            frame = self.decode_to_pixels(z_pred)
            frames.append(frame)
            # Write prediction back into belief (autoregressive)
            state = self.belief_field.write(state, z_pred)

        return torch.stack(frames, dim=1)  # (B, n_steps, C, H, W)

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_p = sum(p.numel() for p in self.encoder.parameters())
        belief_p = sum(p.numel() for p in self.belief_field.parameters())
        decoder_p = sum(p.numel() for p in self.decoder.parameters())
        bio_p = 0
        if self.use_fatigue:
            bio_p += sum(p.numel() for p in self.fatigue.parameters())
        if self.use_inhibition:
            bio_p += sum(p.numel() for p in self.inhibition.parameters())
        return {
            "total": total,
            "trainable": trainable,
            "encoder": encoder_p,
            "belief_field": belief_p,
            "decoder": decoder_p,
            "bio_mechanisms": bio_p,
        }
