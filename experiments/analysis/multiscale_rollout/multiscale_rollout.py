"""
multiscale_rollout.py -- Exp D : Multi-Scale Rollout

Evaluates imagination stability across different horizons.
The BeliefField must be able to "dream" without observation for N steps
without diverging exponentially.
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

from fluidworld.core.world_model import FluidWorldModel


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
    model.requires_grad_(False)
    model.eval()
    return model


@torch.no_grad()
def evaluate_rollout(model, data, device, horizons, batch_size=32):
    """
    For each sequence:
    1. Encode x_0 and write into the BeliefField
    2. Evolve h steps WITHOUT new observations (pure imagination)
    3. Compare z_hat_h with z_h encoded from real x_h

    Returns:
        dict horizon -> {mse, cosine_sim, std}
    """
    N, T, H, W = data.shape
    max_h = max(horizons)
    assert max_h < T, f"Horizon max {max_h} >= T={T}"

    results = {h: {"mse": [], "cosine": []} for h in horizons}

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B = end - start
        stimulus = torch.zeros(B, model.stimulus_dim, device=device)

        # Encode x_0 and initialize the BeliefField
        x0 = torch.from_numpy(data[start:end, 0]).float().unsqueeze(1)
        if x0.max() > 1.0:
            x0 = x0 / 255.0
        x0 = x0.to(device)

        z0 = model.encode(x0)["features"]
        state = model.belief_field.init_state(B, device, torch.float32)
        state = model.belief_field.write(state, z0)

        # Pure imagination rollout
        for h in range(1, max_h + 1):
            state = model.belief_field.evolve(state, stimulus=stimulus)

            if h in horizons:
                # Prediction
                z_hat = model.belief_field.read(state)
                z_hat = model.predictor(z_hat)  # (B, d)

                # Real target
                x_h = torch.from_numpy(data[start:end, h]).float().unsqueeze(1)
                if x_h.max() > 1.0:
                    x_h = x_h / 255.0
                x_h = x_h.to(device)
                z_h = model.target_encoder(x_h)["features"].mean(dim=(-2, -1))

                # Metrics
                mse = ((z_hat - z_h) ** 2).mean(dim=-1)  # (B,)
                cos = F.cosine_similarity(z_hat, z_h, dim=-1)  # (B,)

                results[h]["mse"].extend(mse.cpu().tolist())
                results[h]["cosine"].extend(cos.cpu().tolist())

    # Aggregation
    summary = {}
    for h in horizons:
        mse_arr = np.array(results[h]["mse"])
        cos_arr = np.array(results[h]["cosine"])
        summary[h] = {
            "mse_mean": mse_arr.mean(),
            "mse_std": mse_arr.std(),
            "cosine_mean": cos_arr.mean(),
            "cosine_std": cos_arr.std(),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Exp D : Multiscale Rollout")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoints", type=str, nargs="+", default=None,
                        help="Multiple checkpoints for comparison")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 5, 10])
    parser.add_argument("--n-sequences", type=int, default=500)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--stimulus-dim", type=int, default=1)
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
    print(f"  {data.shape[0]} sequences, {data.shape[1]} frames")

    # Filter horizons to fit in the sequence length
    max_t = data.shape[1] - 1
    horizons = [h for h in args.horizons if h <= max_t]

    for ckpt_path in ckpts:
        label = Path(ckpt_path).stem
        print(f"\n{'='*60}")
        print(f"Checkpoint: {label}")
        print(f"{'='*60}")

        model = load_model(ckpt_path, device, args.in_channels,
                           args.d_model, args.stimulus_dim)

        summary = evaluate_rollout(model, data, device, horizons)

        print(f"\n{'Horizon':>8} | {'MSE':>10} | {'Cosine':>10}")
        print("-" * 35)
        for h in horizons:
            s = summary[h]
            print(f"{h:>8d} | {s['mse_mean']:>7.4f}+/-{s['mse_std']:.3f} | "
                  f"{s['cosine_mean']:>7.4f}+/-{s['cosine_std']:.3f}")

        # Growth rate
        if len(horizons) >= 2:
            h_min, h_max = horizons[0], horizons[-1]
            growth = summary[h_max]["mse_mean"] / max(summary[h_min]["mse_mean"], 1e-8)
            print(f"\n  Growth rate (h={h_max}/h={h_min}): {growth:.1f}x")
            if growth < 20:
                print(f"  PASS: sub-exponential growth")
            elif growth < 100:
                print(f"  WARNING: moderate growth")
            else:
                print(f"  FAIL: exponential divergence")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
