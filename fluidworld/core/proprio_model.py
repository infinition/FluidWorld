"""
proprio_model.py -- Proprioceptive world model (Phase 0): predicts joint evolution without vision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProprioWorldModel(nn.Module):
    """
    Residual MLP: predicts proprio_{t+1} from (proprio_t, action_t).

    Args:
        proprio_dim: proprioceptive vector dimension (6 for SO-101)
        action_dim: action vector dimension (6 for SO-101)
        hidden_dim: hidden layer width
    """

    def __init__(
        self,
        proprio_dim: int = 6,
        action_dim: int = 6,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.proprio_dim = proprio_dim
        self.action_dim = action_dim

        in_dim = proprio_dim + action_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, proprio_dim),
        )

        self._init_small()

    def _init_small(self):
        """Small init for last layer so initial prediction ~ proprio_t via skip connection."""
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def forward(
        self, proprio: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict next-step proprioception.

        Args:
            proprio: (B, proprio_dim) current joint positions
            action: (B, action_dim) command sent

        Returns:
            (B, proprio_dim) predicted joint positions
        """
        x = torch.cat([proprio, action], dim=-1)  # (B, proprio_dim + action_dim)
        delta = self.net(x)  # (B, proprio_dim)
        return proprio + delta  # residual connection


class MultiStepProprioModel(nn.Module):
    """
    Multi-step extension: predicts N steps ahead via autoregression.

    Args:
        proprio_dim: proprioceptive dimension
        action_dim: action dimension
        hidden_dim: MLP width
    """

    def __init__(
        self,
        proprio_dim: int = 6,
        action_dim: int = 6,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.model = ProprioWorldModel(
            proprio_dim=proprio_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        proprio_init: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Autoregressive multi-step prediction.

        Args:
            proprio_init: (B, proprio_dim) initial position
            actions: (B, T, action_dim) action sequence

        Returns:
            (B, T, proprio_dim) predicted position sequence
        """
        B, T, _ = actions.shape
        predictions = []
        p = proprio_init

        for t in range(T):
            p = self.model(p, actions[:, t])
            predictions.append(p)

        return torch.stack(predictions, dim=1)  # (B, T, proprio_dim)

    def compute_loss(
        self,
        proprio_init: torch.Tensor,
        actions: torch.Tensor,
        proprio_targets: torch.Tensor,
    ) -> dict:
        """
        Compute multi-step loss with geometrically decaying weights (w_t = 0.9^t).

        Args:
            proprio_init: (B, proprio_dim) initial position
            actions: (B, T, action_dim) action sequence
            proprio_targets: (B, T, proprio_dim) ground truth positions

        Returns:
            Dict with total loss and per-step MSE
        """
        predictions = self.forward(proprio_init, actions)  # (B, T, proprio_dim)
        T = predictions.shape[1]

        # Per-step loss
        per_step_loss = F.mse_loss(
            predictions, proprio_targets, reduction="none"
        ).mean(dim=(0, 2))  # (T,)

        # Decaying weights
        weights = torch.tensor(
            [0.9 ** t for t in range(T)],
            device=predictions.device,
            dtype=predictions.dtype,
        )
        weights = weights / weights.sum()

        weighted_loss = (per_step_loss * weights).sum()

        return {
            "loss": weighted_loss,
            "per_step_mse": per_step_loss.detach(),
            "mean_mse": per_step_loss.mean().detach(),
            "final_step_mse": per_step_loss[-1].detach() if T > 0 else torch.tensor(0.0),
        }
