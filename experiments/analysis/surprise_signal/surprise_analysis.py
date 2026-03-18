"""
surprise_analysis.py -- Exp B : Surprise Signal (v2 Pixel)

Measures whether the model develops a surprise signal in PIXEL SPACE:
  surprise_t = MSE(x_pred_{t+1}, x_actual_{t+1})

Should be high at bounces/collisions and low on linear trajectories.

Adapted for FluidWorldModelV2 (pixel prediction, not JEPA latent).

Usage :
    python experiments/exp_b_surprise/surprise_analysis.py \
        --checkpoint checkpoints/phase1_pixel/model_step_31000.pt \
        --data-dir data/mnist_test_seq.npy \
        --n-sequences 500
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

_here = Path(__file__).resolve().parent
_project = _here.parent.parent.parent
sys.path.insert(0, str(_project))

from fluidworld.core.world_model_v2 import FluidWorldModelV2


# ── Utils ──────────────────────────────────────────────────────────────


def load_model_v2(ckpt_path, device, in_channels=1, d_model=128, stimulus_dim=1):
    """Load a FluidWorldModelV2 checkpoint (pixel prediction)."""
    model = FluidWorldModelV2(
        in_channels=in_channels,
        d_model=d_model,
        stimulus_dim=stimulus_dim,
        n_encoder_layers=3,
        max_steps_encoder=6,
        belief_spatial_hw=16,
        n_belief_evolve=3,
        # Bio mechanisms (defaults)
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


def detect_bounces(data, threshold=2.0):
    """
    Detect bounces by direction change of the center of mass.

    Args:
        data: (N_seq, T, H, W)

    Returns:
        bounces: (N_seq, T) bool array -- True at bounce frames
    """
    N, T, H, W = data.shape
    bounces = np.zeros((N, T), dtype=bool)

    for i in range(N):
        positions = np.zeros((T, 2), dtype=np.float32)
        for t in range(T):
            frame = data[i, t].astype(np.float32)
            total = frame.sum()
            if total < 1e-6:
                positions[t] = positions[max(0, t - 1)]
            else:
                yy, xx = np.mgrid[:H, :W]
                positions[t, 0] = (yy * frame).sum() / total
                positions[t, 1] = (xx * frame).sum() / total

        # Velocity
        velocity = np.diff(positions, axis=0)  # (T-1, 2)

        # Direction change = velocity sign changes
        for t in range(1, len(velocity)):
            sign_change_y = velocity[t, 0] * velocity[t - 1, 0] < 0
            sign_change_x = velocity[t, 1] * velocity[t - 1, 1] < 0
            speed = np.linalg.norm(velocity[t])
            if (sign_change_y or sign_change_x) and speed > threshold:
                bounces[i, t + 1] = True

    return bounces


@torch.no_grad()
def compute_surprise_signals(model, data, device, batch_size=32):
    """
    Compute the surprise signal in PIXEL SPACE for each frame.

    surprise_t = MSE(decode(evolve(state)), x_{t+1})

    The model:
    1. Encodes frame_t -> z_t
    2. Writes z_t into the BeliefField
    3. Evolves the BeliefField (PDE prediction)
    4. Decodes the future state into pixels
    5. Compares with the actual frame_{t+1}

    Returns:
        surprises: (N_seq, T-1) float array -- per-frame pixel MSE
    """
    N, T, H, W = data.shape
    surprises = np.zeros((N, T - 1), dtype=np.float32)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B = end - start

        # Init belief field
        state = model.belief_field.init_state(B, device, torch.float32)
        stimulus = torch.zeros(B, 1, device=device)

        for t in range(T - 1):
            # Current frame
            x_t = torch.from_numpy(data[start:end, t]).float().unsqueeze(1)
            if x_t.max() > 1.0:
                x_t = x_t / 255.0
            x_t = x_t.to(device)

            # Next frame (ground truth)
            x_next = torch.from_numpy(data[start:end, t + 1]).float().unsqueeze(1)
            if x_next.max() > 1.0:
                x_next = x_next / 255.0
            x_next = x_next.to(device)

            # Encode frame_t
            z_t = model.encode(x_t)["features"]

            # Write + evolve (PDE prediction)
            state = model.belief_field.write(state.detach(), z_t)
            next_state = model.belief_field.evolve(state, stimulus=stimulus)

            # Read future state + decode to pixels
            target_hw = (z_t.shape[2], z_t.shape[3])
            z_pred = model.belief_field.read_spatial(next_state, target_hw)
            x_pred = model.decode_to_pixels(z_pred)  # (B, C, H, W) pixels [0,1]

            # Surprise = pixel MSE (per sample)
            surprise = F.mse_loss(x_pred, x_next, reduction="none")
            surprise = surprise.mean(dim=(1, 2, 3))  # (B,) MSE par sample
            surprises[start:end, t] = surprise.cpu().numpy()

            state = next_state.detach()

    return surprises


# ── Main ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Exp B : Surprise Signal (v2 Pixel)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Single checkpoint")
    parser.add_argument("--checkpoints", type=str, nargs="+", default=None,
                        help="Multiple checkpoints for temporal comparison")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--n-sequences", type=int, default=500)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    ckpts = args.checkpoints or ([args.checkpoint] if args.checkpoint else [])
    if not ckpts:
        parser.error("Provide --checkpoint or --checkpoints")

    # Load Moving MNIST
    print("Loading Moving MNIST...")
    raw = np.load(args.data_dir)
    if raw.shape[0] == 20 and raw.shape[1] == 10000:
        raw = raw.transpose(1, 0, 2, 3)
    data = raw[:args.n_sequences]
    N, T = data.shape[0], data.shape[1]
    print(f"  {N} sequences, {T} frames")

    # Bounce detection
    print("Detecting bounces...")
    bounces = detect_bounces(data)
    n_bounces = bounces[:, 1:].sum()
    n_linear = (~bounces[:, 1:]).sum()
    print(f"  {n_bounces} bounce frames, {n_linear} linear frames")

    # Per-checkpoint analysis
    print(f"\n{'Checkpoint':<40} | {'Mean':>8} | {'Bounce':>8} | "
          f"{'Linear':>8} | {'Ratio':>6} | {'Corr':>6}")
    print("-" * 90)

    for ckpt_path in ckpts:
        model = load_model_v2(ckpt_path, device, args.in_channels, args.d_model)

        surprises = compute_surprise_signals(model, data, device)

        # Global stats
        mean_surprise = surprises.mean()

        # Per frame-type stats
        bounce_mask = bounces[:, 1:]
        T_surp = surprises.shape[1]
        bounce_mask = bounce_mask[:, :T_surp]

        surprise_bounce = surprises[bounce_mask].mean() if bounce_mask.any() else 0.0
        surprise_linear = surprises[~bounce_mask].mean()
        ratio = surprise_bounce / max(surprise_linear, 1e-8)

        # Pearson correlation
        flat_surprise = surprises[:, :T_surp].flatten()
        flat_bounce = bounce_mask.flatten().astype(np.float32)
        if flat_surprise.std() > 1e-8 and flat_bounce.std() > 1e-8:
            corr = np.corrcoef(flat_surprise, flat_bounce)[0, 1]
        else:
            corr = 0.0

        label = Path(ckpt_path).stem[:38]
        print(f"{label:<40} | {mean_surprise:>8.5f} | {surprise_bounce:>8.5f} | "
              f"{surprise_linear:>8.5f} | {ratio:>6.2f} | {corr:>6.3f}")

        del model
        torch.cuda.empty_cache()

    print("\n--- Interpretation ---")
    print("  Surprise = pixel MSE between PDE prediction and actual frame")
    print("  Ratio > 1.5   : model is MORE surprised at bounces (good sign)")
    print("  Corr > 0.3    : significant surprise/event correlation")
    print("  Mean decreases: model is learning to predict (training progress)")
    print("\n  Note: unlike the old Exp B (latent JEPA),")
    print("  surprise is measured in PIXEL space (direct MSE),")
    print("  which is much more interpretable.")


if __name__ == "__main__":
    main()
