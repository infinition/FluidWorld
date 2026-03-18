"""
temporal_probe.py -- Exp A : Temporal Probing (v2 Pixel)

Demonstrates that the model encodes temporal dynamics (position, velocity)
and not just the static identity of the content.

A linear probe is trained to predict the FUTURE position of the digit
from the features encoded at time t.

Adapted for FluidWorldModelV2 (pixel prediction).

Usage :
    python experiments/exp_a_temporal_probe/temporal_probe.py \
        --checkpoint checkpoints/phase1_pixel/model_step_31000.pt \
        --data-dir data/mnist_test_seq.npy \
        --horizons 1 3 5 --n-sequences 2000
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_here = Path(__file__).resolve().parent
_project = _here.parent.parent.parent
sys.path.insert(0, str(_project))

from fluidworld.core.world_model_v2 import FluidWorldModelV2


# ── Utils ──────────────────────────────────────────────────────────────


def load_model_v2(ckpt_path, device, in_channels=1, d_model=128):
    """Load a FluidWorldModelV2 checkpoint (pixel prediction)."""
    model = FluidWorldModelV2(
        in_channels=in_channels,
        d_model=d_model,
        stimulus_dim=1,
        n_encoder_layers=3,
        max_steps_encoder=6,
        belief_spatial_hw=16,
        n_belief_evolve=3,
        use_fatigue=True,
        fatigue_cost=0.01,
        fatigue_recovery=0.05,
        use_inhibition=True,
        inhibition_strength=0.3,
        use_memory_pump=True,
        use_hebbian=True,
        hebbian_lr=0.01,
        hebbian_decay=0.99,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    else:
        sd = ckpt
    model.load_state_dict(sd, strict=False)
    model.requires_grad_(False)
    model.eval()
    return model


def extract_digit_positions(data):
    """
    Extract the center of mass of digits for each frame.

    Args:
        data: (N_seq, T, H, W) array uint8/float

    Returns:
        positions: (N_seq, T, 2) float array -- (y_center, x_center) normalized [0, 1]
    """
    N, T, H, W = data.shape
    positions = np.zeros((N, T, 2), dtype=np.float32)
    for i in range(N):
        for t in range(T):
            frame = data[i, t].astype(np.float32)
            if frame.max() > 1.0:
                frame = frame / 255.0
            total = frame.sum()
            if total < 1e-6:
                positions[i, t] = [0.5, 0.5]
            else:
                yy, xx = np.mgrid[:H, :W]
                positions[i, t, 0] = (yy * frame).sum() / total / H
                positions[i, t, 1] = (xx * frame).sum() / total / W
    return positions


@torch.no_grad()
def compute_activation_com(z):
    """
    Compute the center of mass of activations in the feature map.

    Args:
        z: (B, d, Hf, Wf) feature map

    Returns:
        com: (B, 2) normalized center of mass [0, 1] -- (y, x)
    """
    B, d, Hf, Wf = z.shape
    energy = z.abs().sum(dim=1)  # (B, Hf, Wf)
    energy = energy / (energy.sum(dim=(-2, -1), keepdim=True) + 1e-8)

    yy = torch.linspace(0, 1, Hf, device=z.device).view(1, Hf, 1).expand(B, Hf, Wf)
    xx = torch.linspace(0, 1, Wf, device=z.device).view(1, 1, Wf).expand(B, Hf, Wf)

    cy = (energy * yy).sum(dim=(-2, -1))
    cx = (energy * xx).sum(dim=(-2, -1))

    return torch.stack([cy, cx], dim=-1)  # (B, 2)


@torch.no_grad()
def encode_sequences(model, data, device, batch_size=64, spatial=False):
    """
    Encode all frames with the v2 model.

    Args:
        data: (N_seq, T, H, W) array
        spatial: 'com' = center of mass (2 dims)
                 'pool4' = adaptive pool 4x4 (2048 dims)
                 'flat' = full flatten (128*16*16 dims)
                 False = global pool (128 dims)

    Returns:
        features: (N_seq, T, feat_dim) tensor
        feat_dim: int
    """
    N, T, H, W = data.shape
    d = model.d_model

    # First pass to detect spatial size
    test_frame = torch.from_numpy(data[0, 0]).float().unsqueeze(0).unsqueeze(0).to(device)
    if test_frame.max() > 1.0:
        test_frame = test_frame / 255.0
    test_z = model.encode(test_frame)["features"]
    Hf, Wf = test_z.shape[2], test_z.shape[3]

    if spatial == "com":
        feat_dim = 2
        mode_label = "CoM(2)"
    elif spatial == "pool4":
        feat_dim = d * 4 * 4
        mode_label = f"pool4({feat_dim})"
    elif spatial == "flat":
        feat_dim = d * Hf * Wf
        mode_label = f"flat({feat_dim})"
    else:
        feat_dim = d
        mode_label = f"pooled({feat_dim})"

    print(f"  Feature map: ({d}, {Hf}, {Wf}) -> feat_dim={feat_dim} ({mode_label})")

    pool4 = torch.nn.AdaptiveAvgPool2d((4, 4)) if spatial == "pool4" else None
    all_feats = torch.zeros(N, T, feat_dim)

    for t in range(T):
        frames = torch.from_numpy(data[:, t]).float().unsqueeze(1)
        if frames.max() > 1.0:
            frames = frames / 255.0
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            x = frames[start:end].to(device)
            z = model.encode(x)["features"]

            if spatial == "com":
                feat = compute_activation_com(z)
            elif spatial == "pool4":
                feat = pool4(z).flatten(1)
            elif spatial == "flat":
                feat = z.flatten(1)
            else:
                feat = z.mean(dim=(-2, -1))

            all_feats[start:end, t] = feat.cpu()

    return all_feats, feat_dim


# ── Probe ──────────────────────────────────────────────────────────────


class TemporalProbe(nn.Module):
    def __init__(self, feat_dim, n_outputs=2):
        super().__init__()
        self.head = nn.Linear(feat_dim, n_outputs)

    def forward(self, x):
        return self.head(x)


def train_temporal_probe(features, positions, horizon, feat_dim=None, epochs=50, lr=1e-3):
    """
    Train a linear probe: z_t -> position_{t+horizon}.

    Returns:
        dict with mse, r2, baseline_mse
    """
    N, T, d = features.shape
    if feat_dim is None:
        feat_dim = d
    T_valid = T - horizon

    X = features[:, :T_valid].reshape(-1, d)
    Y = torch.from_numpy(positions[:, horizon:T_valid + horizon].reshape(-1, 2))

    # Baseline "copy current position"
    Y_current = torch.from_numpy(positions[:, :T_valid].reshape(-1, 2))
    baseline_mse = ((Y_current - Y) ** 2).mean().item()

    # Split train/test (80/20)
    n_total = X.shape[0]
    n_train = int(0.8 * n_total)
    perm = torch.randperm(n_total)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=256, shuffle=True,
    )

    probe = TemporalProbe(feat_dim, 2)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)

    best_mse = float("inf")
    for epoch in range(epochs):
        probe.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(probe(xb), yb)
            loss.backward()
            opt.step()

        probe.eval()
        with torch.no_grad():
            pred = probe(X_test)
            mse = nn.functional.mse_loss(pred, Y_test).item()
            best_mse = min(best_mse, mse)

    # R^2
    probe.eval()
    with torch.no_grad():
        pred = probe(X_test)
        ss_res = ((pred - Y_test) ** 2).sum().item()
        ss_tot = ((Y_test - Y_test.mean(dim=0)) ** 2).sum().item()
        r2 = 1.0 - ss_res / max(ss_tot, 1e-8)

    return {
        "horizon": horizon,
        "mse": best_mse,
        "r2": r2,
        "baseline_mse": baseline_mse,
        "ratio": best_mse / max(baseline_mse, 1e-8),
    }


# ── Main ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Exp A : Temporal Probe (v2 Pixel)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--checkpoint2", type=str, default=None,
                        help="Second checkpoint for comparison")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="mnist_test_seq.npy")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--n-sequences", type=int, default=2000,
                        help="Number of sequences to use")
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--spatial", type=str, default="",
                        choices=["", "com", "pool4", "flat"],
                        help="Feature mode: '' = pooled (128d), 'com' = center of mass (2d), "
                             "'pool4' = adaptive pool 4x4 (2048d), 'flat' = full flatten")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    # Load Moving MNIST
    print("Loading Moving MNIST...")
    raw = np.load(args.data_dir)
    if raw.shape[0] == 20 and raw.shape[1] == 10000:
        raw = raw.transpose(1, 0, 2, 3)
    data = raw[:args.n_sequences]
    print(f"  {data.shape[0]} sequences, {data.shape[1]} frames, {data.shape[2]}x{data.shape[3]}")

    # Positions by center of mass
    print("Extracting positions (center of mass)...")
    positions = extract_digit_positions(data)

    def run_one_checkpoint(ckpt_path, label):
        print(f"\n{'='*60}")
        print(f"Checkpoint: {label}")
        print(f"  {ckpt_path}")
        print(f"{'='*60}")

        model = load_model_v2(ckpt_path, device, args.in_channels, args.d_model)

        print("Encoding sequences...")
        features, feat_dim = encode_sequences(
            model, data, device, spatial=args.spatial or False)

        print(f"\n{'Horizon':>8} | {'MSE':>8} | {'Baseline':>8} | {'Ratio':>6} | {'R2':>6}")
        print("-" * 50)

        for h in args.horizons:
            res = train_temporal_probe(features, positions, h, feat_dim, args.epochs)
            print(f"{res['horizon']:>8d} | {res['mse']:>8.5f} | "
                  f"{res['baseline_mse']:>8.5f} | {res['ratio']:>6.3f} | "
                  f"{res['r2']:>6.3f}")

        return features

    run_one_checkpoint(args.checkpoint, Path(args.checkpoint).stem)

    if args.checkpoint2:
        run_one_checkpoint(args.checkpoint2, Path(args.checkpoint2).stem)

    print("\n--- Interpretation ---")
    print("  Ratio < 1.0 : model predicts BETTER than copying current position")
    print("  R2 > 0.6    : strong temporal information in features")
    print("  R2 < 0.0    : no temporal information (worse than average)")
    print("\n  Note: uses FluidWorldModelV2 (pixel prediction)")
    print("  Features are extracted from the PDE encoder (not the decoder)")


if __name__ == "__main__":
    main()
