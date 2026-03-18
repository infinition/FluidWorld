"""
train_curriculum.py -- Exp F : Curriculum Training

Trains the model through increasing levels of complexity.
Automatic promotion when the model has "mastered" a level
(surprise ratio < threshold).
"""

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_here = Path(__file__).resolve().parent
_project = _here.parent.parent.parent
sys.path.insert(0, str(_project))

from fluidworld.core.world_model import FluidWorldModel
from fluidworld.core.vicreg import vicreg_loss


# ── Dataset ────────────────────────────────────────────────────────────


class CurriculumDataset(Dataset):
    """Loads a .npy sequence file (N, T, H, W) as sliding windows."""

    def __init__(self, npy_path, bptt_steps=4):
        data = np.load(npy_path)
        # Handle (T, N, H, W) format like standard Moving MNIST
        if data.ndim == 4 and data.shape[0] == 20 and data.shape[1] == 10000:
            data = data.transpose(1, 0, 2, 3)
        self.data = data
        self.bptt_steps = bptt_steps
        N, T = data.shape[0], data.shape[1]
        self.samples = []
        max_start = T - (bptt_steps + 1)
        for i in range(N):
            for s in range(max(max_start + 1, 1)):
                self.samples.append((i, s))
        print(f"  Dataset: {npy_path} -- {len(self.samples)} windows")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_idx, start = self.samples[idx]
        clip = self.data[seq_idx, start:start + self.bptt_steps + 1]
        clip = np.expand_dims(clip, axis=1)  # (T, 1, H, W)
        t = torch.from_numpy(clip).float()
        if t.max() > 1.0:
            t = t / 255.0
        return t


# ── Utils ──────────────────────────────────────────────────────────────


def load_model(ckpt_path, device, in_channels=1, d_model=128, stimulus_dim=1):
    model = FluidWorldModel(
        in_channels=in_channels, d_model=d_model, stimulus_dim=stimulus_dim,
    ).to(device)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        else:
            sd = ckpt
        model.load_state_dict(sd, strict=False)
        print(f"  Checkpoint loaded: {ckpt_path}")
    return model


@torch.no_grad()
def compute_mean_surprise(model, loader, device, max_batches=50):
    """Compute mean surprise over a loader."""
    model.eval()
    surprises = []

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = batch.to(device)
        B, T_plus1 = batch.shape[0], batch.shape[1]
        T = T_plus1 - 1
        stimulus = torch.zeros(B, model.stimulus_dim, device=device)
        state = model.belief_field.init_state(B, device, batch.dtype)

        for t in range(T):
            x_t, x_next = batch[:, t], batch[:, t + 1]
            z_t = model.encode(x_t)["features"]
            state = model.belief_field.write(state.detach(), z_t)
            next_state = model.belief_field.evolve(state, stimulus=stimulus)
            z_pred = model.predictor(model.belief_field.read(next_state))
            z_actual = model.target_encoder(x_next)["features"].mean(dim=(-2, -1))
            surprise = (z_pred - z_actual).norm(dim=-1).mean().item()
            surprises.append(surprise)
            state = next_state.detach()

    return np.mean(surprises) if surprises else float("inf")


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        B, T_plus1 = batch.shape[0], batch.shape[1]
        T = T_plus1 - 1
        stimulus = torch.zeros(B, model.stimulus_dim, device=device)
        state = model.belief_field.init_state(B, device, batch.dtype)

        batch_loss = 0.0
        n_steps = 0

        for t in range(T):
            x_t, x_next = batch[:, t], batch[:, t + 1]

            enc_out = model.encode(x_t)
            z_t = enc_out["features"]
            z_t_pooled = z_t.mean(dim=(-2, -1))

            state = model.belief_field.write(state.detach(), z_t)
            next_state = model.belief_field.evolve(state, stimulus=stimulus)
            z_pred = model.predictor(model.belief_field.read(next_state))

            with torch.no_grad():
                z_target = model.target_encoder(x_next)["features"].mean(dim=(-2, -1))

            pred_loss = F.mse_loss(z_pred, z_target)
            vic = vicreg_loss(z_t_pooled, var_weight=5.0, cov_weight=0.04)
            step_loss = pred_loss + vic["vicreg_total"]

            batch_loss = batch_loss + step_loss
            n_steps += 1
            state = next_state.detach()

        batch_loss = batch_loss / n_steps
        optimizer.zero_grad(set_to_none=True)
        batch_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.update_target()

        total_loss += batch_loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ── Main ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Exp F : Curriculum Training")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Curriculum directory or single .npy file")
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3],
                        help="Levels to train (0-3 for curriculum, 4 for standard)")
    parser.add_argument("--epochs-per-level", type=int, default=10)
    parser.add_argument("--promotion-threshold", type=float, default=1.2,
                        help="Surprise ratio threshold for promotion")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bptt-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--tag", type=str, default="curriculum")
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--stimulus-dim", type=int, default=1)
    parser.add_argument("--save-dir", type=str, default="checkpoints/curriculum")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    # Level -> file mapping
    data_path = Path(args.data_dir)
    level_files = {
        0: data_path / "level_0_permanence.npy",
        1: data_path / "level_1_causality.npy",
        2: data_path / "level_2_interaction.npy",
        3: data_path / "level_3_multiobject.npy",
    }

    # If it's a single file (baseline or standard MNIST mode)
    if data_path.is_file():
        for lvl in args.levels:
            level_files[lvl] = data_path

    model = load_model(args.checkpoint, device, args.in_channels,
                       args.d_model, args.stimulus_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Exp F -- Curriculum Training")
    print(f"  Levels: {args.levels}")
    print(f"  Epochs per level: {args.epochs_per_level}")
    print(f"  Promotion threshold: {args.promotion_threshold}")

    total_epochs = 0

    for level in args.levels:
        npy_path = level_files.get(level)
        if npy_path is None or not npy_path.exists():
            print(f"\n  SKIP level {level}: {npy_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"LEVEL {level}")
        print(f"{'='*60}")

        dataset = CurriculumDataset(str(npy_path), bptt_steps=args.bptt_steps)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, drop_last=True)

        # Initial surprise
        surprise_init = compute_mean_surprise(model, loader, device)
        print(f"  Initial surprise: {surprise_init:.4f}")

        for epoch in range(1, args.epochs_per_level + 1):
            t0 = time.time()
            avg_loss = train_one_epoch(model, loader, optimizer, device)
            dt = time.time() - t0
            total_epochs += 1

            surprise = compute_mean_surprise(model, loader, device)
            ratio = surprise / max(surprise_init, 1e-8)

            print(f"  Epoch {epoch:>3d} | Loss {avg_loss:.4f} | "
                  f"Surprise {surprise:.4f} | Ratio {ratio:.3f} | "
                  f"{dt:.1f}s")

            if ratio < args.promotion_threshold and epoch >= 3:
                print(f"  >> PROMOTION: ratio {ratio:.3f} < {args.promotion_threshold}")
                break

    # Save final checkpoint
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{args.tag}_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "total_epochs": total_epochs,
        "levels": args.levels,
    }, save_path)
    print(f"\nFinal checkpoint: {save_path}")
    print(f"Total epochs: {total_epochs}")
    print("Run Exp A/B/D with this checkpoint to compare against baseline.")


if __name__ == "__main__":
    main()
