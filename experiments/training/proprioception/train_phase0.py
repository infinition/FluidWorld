"""
train_phase0.py -- Phase 0: proprioceptive world model training.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# ── Import FluidWorld ──
_here = Path(__file__).resolve().parent
_project = _here.parent.parent.parent
sys.path.insert(0, str(_project))

from fluidworld.core.proprio_model import ProprioWorldModel, MultiStepProprioModel


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────


class ProprioDataset(Dataset):
    """Triplet dataset (proprio_t, action_t, proprio_{t+1}) from .npz episodes."""

    def __init__(self, data_dir: str, horizon: int = 1):
        self.horizon = horizon
        self.proprios = []  # list of (proprio_t,)
        self.actions = []  # list of (action_t,) or (action_t:t+H,)
        self.targets = []  # list of (proprio_{t+1},) or (proprio_{t+1:t+H+1},)

        episode_files = sorted(Path(data_dir).glob("episode_*.npz"))
        if not episode_files:
            raise FileNotFoundError(
                f"No episode_*.npz files found in {data_dir}"
            )

        total_transitions = 0
        for ep_path in episode_files:
            data = np.load(str(ep_path))
            proprios = data["proprios"].astype(np.float32)  # (N, D)
            actions = data["actions"].astype(np.float32)  # (N, D)

            N = len(proprios)
            if N < horizon + 1:
                continue

            for t in range(N - horizon):
                self.proprios.append(proprios[t])
                if horizon == 1:
                    self.actions.append(actions[t])
                    self.targets.append(proprios[t + 1])
                else:
                    self.actions.append(actions[t : t + horizon])
                    self.targets.append(proprios[t + 1 : t + horizon + 1])
                total_transitions += 1

        print(
            f"[ProprioDataset] {len(episode_files)} episodes, "
            f"{total_transitions} transitions, horizon={horizon}"
        )

    def __len__(self):
        return len(self.proprios)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.proprios[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.targets[idx]),
        )


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────


def train_single_step(args):
    """Train ProprioWorldModel (single-step)."""
    device = torch.device(args.device)

    # ── Dataset ──
    dataset = ProprioDataset(args.data_dir, horizon=1)
    proprio_dim = dataset.proprios[0].shape[-1]
    action_dim = dataset.actions[0].shape[-1]

    # Split train/val (90/10)
    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # ── Model ──
    model = ProprioWorldModel(
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    print(f"[Model] ProprioWorldModel")
    print(f"  proprio_dim={proprio_dim}, action_dim={action_dim}")
    print(f"  hidden_dim={args.hidden_dim}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ── Training loop ──
    best_val_loss = float("inf")
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_losses = []
        for proprio, action, target in train_loader:
            proprio = proprio.to(device)
            action = action.to(device)
            target = target.to(device)

            pred = model(proprio, action)
            loss = F.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for proprio, action, target in val_loader:
                proprio = proprio.to(device)
                action = action.to(device)
                target = target.to(device)

                pred = model(proprio, action)
                loss = F.mse_loss(pred, target)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_mse={train_loss:.6f} | val_mse={val_loss:.6f} | lr={lr:.2e}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "proprio_dim": proprio_dim,
                    "action_dim": action_dim,
                    "hidden_dim": args.hidden_dim,
                },
                ckpt_dir / "best_proprio_model.pt",
            )
            print(f"  -> Best model saved (val_mse={val_loss:.6f})")

        scheduler.step()

    print(f"\n[Done] Best val_mse = {best_val_loss:.6f}")
    if best_val_loss < 0.01:
        print("  SUCCESS: MSE < 0.01, Phase 0 validated!")
    else:
        print("  WARNING: MSE >= 0.01, check data or hyperparameters.")


def train_multi_step(args):
    """Train MultiStepProprioModel (N steps)."""
    device = torch.device(args.device)

    # ── Dataset ──
    dataset = ProprioDataset(args.data_dir, horizon=args.horizon)
    proprio_dim = dataset.proprios[0].shape[-1]
    action_dim = dataset.actions[0].shape[-2] if len(dataset.actions[0].shape) > 1 else dataset.actions[0].shape[-1]
    # In multi-step, actions[0] is (horizon, action_dim)
    if len(dataset.actions[0].shape) > 1:
        action_dim = dataset.actions[0].shape[-1]

    # Split train/val
    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # ── Model ──
    model = MultiStepProprioModel(
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    print(f"[Model] MultiStepProprioModel (horizon={args.horizon})")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_val_loss = float("inf")
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_losses = []
        for proprio, actions, targets in train_loader:
            proprio = proprio.to(device)
            actions = actions.to(device)
            targets = targets.to(device)

            result = model.compute_loss(proprio, actions, targets)
            loss = result["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        val_per_step = []
        with torch.no_grad():
            for proprio, actions, targets in val_loader:
                proprio = proprio.to(device)
                actions = actions.to(device)
                targets = targets.to(device)

                result = model.compute_loss(proprio, actions, targets)
                val_losses.append(result["loss"].item())
                val_per_step.append(result["per_step_mse"].cpu())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        avg_per_step = torch.stack(val_per_step).mean(dim=0)

        per_step_str = " ".join(f"{v:.4f}" for v in avg_per_step.tolist())
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train={train_loss:.6f} | val={val_loss:.6f} | "
            f"per_step=[{per_step_str}]"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.model.state_dict(),
                    "val_loss": val_loss,
                    "horizon": args.horizon,
                },
                ckpt_dir / "best_proprio_multistep.pt",
            )
            print(f"  -> Best model saved")

        scheduler.step()

    print(f"\n[Done] Best val_loss = {best_val_loss:.6f}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FluidWorld Phase 0: proprioceptive training"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Directory containing episode_*.npz files"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints/phase0")
    parser.add_argument(
        "--mode", type=str, default="single",
        choices=["single", "multi"],
        help="single = 1-step, multi = N-step autoregressive"
    )
    parser.add_argument(
        "--horizon", type=int, default=10,
        help="Number of future steps (multi mode only)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("FluidWorld -- Phase 0: Proprioceptive World Model")
    print("=" * 60)
    print(f"  data_dir: {args.data_dir}")
    print(f"  mode:     {args.mode}")
    print(f"  device:   {args.device}")
    print()

    if args.mode == "single":
        train_single_step(args)
    else:
        train_multi_step(args)


if __name__ == "__main__":
    main()
