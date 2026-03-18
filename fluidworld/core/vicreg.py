"""
vicreg.py -- Anti-collapse losses (variance + covariance regularization). Ref: Bardes et al. VICReg, ICLR 2022.
"""

import torch
import torch.nn.functional as F


def variance_loss(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """Force each dimension to have variance >= gamma^2. Uses var (not std) to avoid 0/0 gradient at collapse."""
    z_f32 = z.float()
    var = z_f32.var(dim=0)  # (D,)
    return F.relu(gamma * gamma - var + eps).mean()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """Minimize cross-dimension correlations."""
    z_f32 = z.float()
    B, D = z_f32.shape
    if B < 2:
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)

    z_centered = z_f32 - z_f32.mean(dim=0, keepdim=True)
    cov = (z_centered.T @ z_centered) / (B - 1)

    # Off-diagonal of covariance matrix
    off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return off_diag / D


def vicreg_loss(
    z_pred: torch.Tensor,
    var_weight: float = 5.0,
    cov_weight: float = 0.04,
) -> dict:
    """VICReg on z_pred only (z_target is always detached, so excluded)."""
    var_loss = variance_loss(z_pred)
    cov_loss = covariance_loss(z_pred)

    var_total = var_weight * var_loss
    cov_total = cov_weight * cov_loss

    return {
        "var_loss": var_total,
        "cov_loss": cov_total,
        "vicreg_total": var_total + cov_total,
        "pred_std_mean": z_pred.float().std(dim=0).mean().detach(),
    }