"""
train_hebbian.py -- Exp E : Training with Hebbian Gating

Fine-tunes a Phase 1 checkpoint with HebbianBeliefField and compares
metrics before/after.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_here = Path(__file__).resolve().parent
_project = _here.parent.parent.parent
sys.path.insert(0, str(_project))

from fluidworld.core.world_model import FluidWorldModel
from fluidworld.core.video_dataset import MovingMNISTDataset
from experiments.exp_e_hebbian.hebbian_belief_field import HebbianBeliefField


def load_model(ckpt_path, device, in_channels=1, d_model=128, stimulus_dim=1):
    model = FluidWorldModel(
        in_channels=in_channels, d_model=d_model, stimulus_dim=stimulus_dim,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    else:
        sd = ckpt
    model.load_state_dict(sd, strict=False)
    return model


def replace_belief_field(model, alpha, gamma, device):
    """Replace the standard BeliefField with a HebbianBeliefField."""
    old_bf = model.belief_field
    new_bf = HebbianBeliefField(
        channels=old_bf.channels,
        stimulus_dim=model.stimulus_dim,
        spatial_hw=old_bf.spatial_hw,
        n_evolve_steps=old_bf.n_evolve_steps,
        alpha=alpha,
        gamma=gamma,
    ).to(device)

    # Copy shared weights
    old_sd = old_bf.state_dict()
    new_sd = new_bf.state_dict()
    for k in old_sd:
        if k in new_sd and old_sd[k].shape == new_sd[k].shape:
            new_sd[k] = old_sd[k]
    new_bf.load_state_dict(new_sd)

    model.belief_field = new_bf
    return model


def train_one_epoch(model, loader, optimizer, device, bptt_steps=4):
    model.train()
    total_loss = 0.0
    total_pred = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        B, T_plus1 = batch.shape[0], batch.shape[1]
        T = T_plus1 - 1

        state = model.belief_field.init_state(B, device, batch.dtype)
        trace = model.belief_field.init_trace(B, device, batch.dtype)
        stimulus = torch.zeros(B, model.stimulus_dim, device=device)

        batch_loss = 0.0
        n_steps = 0

        for t in range(T):
            x_t = batch[:, t]
            x_next = batch[:, t + 1]

            # Encode
            z_t = model.encode(x_t)["features"]
            z_t_pooled = z_t.mean(dim=(-2, -1))

            # Write
            state = model.belief_field.write(state.detach(), z_t)

            # Evolve with Hebbian
            next_state, trace = model.belief_field.evolve_with_hebbian(
                state, trace.detach(), stimulus=stimulus,
            )

            # Predict
            z_pred_pooled = model.belief_field.read(next_state)
            z_pred = model.predictor(z_pred_pooled)

            # Target
            with torch.no_grad():
                z_target = model.target_encoder(x_next)["features"].mean(dim=(-2, -1))

            loss = F.mse_loss(z_pred, z_target)
            batch_loss = batch_loss + loss
            n_steps += 1
            state = next_state.detach()

        batch_loss = batch_loss / n_steps
        optimizer.zero_grad(set_to_none=True)
        batch_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.update_target()

        total_loss += batch_loss.item()
        total_pred += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1), total_pred / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Exp E : Hebbian Gating")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Hebbian reinforcement strength")
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="Trace decay rate")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bptt-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--tag", type=str, default="hebbian")
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--stimulus-dim", type=int, default=1)
    parser.add_argument("--save-dir", type=str, default="checkpoints/hebbian")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    print(f"Exp E -- Hebbian Gating")
    print(f"  alpha={args.alpha}, gamma={args.gamma}")
    print(f"  Checkpoint: {args.checkpoint}")

    # Dataset
    dataset = MovingMNISTDataset(args.data_dir, bptt_steps=args.bptt_steps)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, drop_last=True)

    # Model + Hebbian
    model = load_model(args.checkpoint, device, args.in_channels,
                       args.d_model, args.stimulus_dim)
    model = replace_belief_field(model, args.alpha, args.gamma, device)

    # Only optimize the HebbianBeliefField (fine-tune)
    optimizer = torch.optim.AdamW(model.belief_field.parameters(), lr=args.lr)

    print(f"\n{'Epoch':>5} | {'Loss':>8} | {'Pred MSE':>8}")
    print("-" * 30)

    for epoch in range(1, args.epochs + 1):
        avg_loss, avg_pred = train_one_epoch(
            model, loader, optimizer, device, args.bptt_steps,
        )
        print(f"{epoch:>5d} | {avg_loss:>8.4f} | {avg_pred:>8.4f}")

    # Save
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"hebbian_{args.tag}_a{args.alpha}_g{args.gamma}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "alpha": args.alpha,
        "gamma": args.gamma,
        "epochs": args.epochs,
    }, save_path)
    print(f"\nCheckpoint saved: {save_path}")
    print("Run Exp A/B/D with this checkpoint to compare.")


if __name__ == "__main__":
    main()
