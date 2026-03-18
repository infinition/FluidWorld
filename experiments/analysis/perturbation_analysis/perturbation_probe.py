"""
perturbation_probe.py -- Exp C : Perturbation Probing

Tests whether the model has learned causal relationships in latent space.
By locally perturbing an object's features, the prediction should change
in a coherent and localized manner.
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


def find_object_region(features, top_k=4):
    """
    Identify the object region in the feature map by max activation.

    Args:
        features: (B, C, H, W)
        top_k: number of spatial positions to consider

    Returns:
        mask: (B, 1, H, W) -- 1 for object positions, 0 elsewhere
    """
    B, C, H, W = features.shape
    # Mean activation per spatial position
    activation = features.abs().mean(dim=1)  # (B, H, W)
    flat = activation.reshape(B, -1)
    _, top_idx = flat.topk(top_k, dim=1)

    mask = torch.zeros(B, H * W, device=features.device)
    mask.scatter_(1, top_idx, 1.0)
    mask = mask.reshape(B, 1, H, W)
    return mask


@torch.no_grad()
def run_perturbation_analysis(model, data, device, n_samples=200,
                               perturbation_scale=0.5):
    """
    For each frame:
    1. Encode -> z_t
    2. Identify the object region
    3. Perturb z_t in the object region
    4. Predict normal vs perturbed
    5. Measure locality and coherence of the change
    """
    N, T, H, W = data.shape
    n_samples = min(n_samples, N)
    stimulus = torch.zeros(1, model.stimulus_dim, device=device)

    locality_scores = []
    coherence_scores = []
    sensitivity_scores = []

    for i in range(n_samples):
        t = T // 2  # middle of the sequence
        x = torch.from_numpy(data[i, t]).float().unsqueeze(0).unsqueeze(0)
        if x.max() > 1.0:
            x = x / 255.0
        x = x.to(device)

        # Encode
        z_t = model.encode(x)["features"]  # (1, C, H', W')

        # Object region
        obj_mask = find_object_region(z_t)  # (1, 1, H', W')
        bg_mask = 1.0 - obj_mask

        # Localized perturbation
        delta = perturbation_scale * torch.randn_like(z_t) * obj_mask

        z_normal = z_t
        z_perturbed = z_t + delta

        # Predictions via BeliefField
        state = model.belief_field.init_state(1, device, torch.float32)

        # Normal
        state_n = model.belief_field.write(state, z_normal)
        state_n = model.belief_field.evolve(state_n, stimulus=stimulus)
        pred_n = model.belief_field.read(state_n)
        pred_n = model.predictor(pred_n)

        # Perturbed
        state_p = model.belief_field.write(state, z_perturbed)
        state_p = model.belief_field.evolve(state_p, stimulus=stimulus)
        pred_p = model.belief_field.read(state_p)
        pred_p = model.predictor(pred_p)

        # Measurements
        diff_pred = (pred_p - pred_n).abs()  # (1, C)

        # For locality: compare change in spatial state
        diff_state = (state_p - state_n).abs()  # (1, C, H_b, W_b)
        # Resize masks to belief field dimensions
        H_b, W_b = diff_state.shape[2], diff_state.shape[3]
        obj_mask_bf = F.interpolate(obj_mask, size=(H_b, W_b), mode="nearest")
        bg_mask_bf = 1.0 - obj_mask_bf

        change_obj = (diff_state * obj_mask_bf).mean().item()
        change_bg = (diff_state * bg_mask_bf).mean().item()
        locality = change_obj / max(change_bg, 1e-8)

        # Coherence: direction of change
        delta_flat = delta.reshape(1, -1)
        diff_state_flat = (state_p - state_n).reshape(1, -1)
        cos_sim = F.cosine_similarity(delta_flat, diff_state_flat, dim=-1).item()

        # Sensitivity
        delta_norm = delta.norm().item()
        diff_norm = diff_pred.norm().item()
        sensitivity = diff_norm / max(delta_norm, 1e-8)

        locality_scores.append(locality)
        coherence_scores.append(abs(cos_sim))
        sensitivity_scores.append(sensitivity)

    return {
        "locality_mean": np.mean(locality_scores),
        "locality_std": np.std(locality_scores),
        "coherence_mean": np.mean(coherence_scores),
        "coherence_std": np.std(coherence_scores),
        "sensitivity_mean": np.mean(sensitivity_scores),
        "sensitivity_std": np.std(sensitivity_scores),
    }


def main():
    parser = argparse.ArgumentParser(description="Exp C : Perturbation Probing")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--perturbation-scale", type=float, default=0.5)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--stimulus-dim", type=int, default=1)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    print("Exp C -- Perturbation Probing")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Scale     : {args.perturbation_scale}")

    # Load Moving MNIST
    raw = np.load(args.data_dir)
    if raw.shape[0] == 20 and raw.shape[1] == 10000:
        raw = raw.transpose(1, 0, 2, 3)
    data = raw[:args.n_samples]

    model = load_model(args.checkpoint, device, args.in_channels,
                       args.d_model, args.stimulus_dim)

    results = run_perturbation_analysis(
        model, data, device, args.n_samples, args.perturbation_scale,
    )

    print(f"\n--- Results ---")
    print(f"  Locality    : {results['locality_mean']:.3f} +/- {results['locality_std']:.3f}")
    print(f"  Coherence   : {results['coherence_mean']:.3f} +/- {results['coherence_std']:.3f}")
    print(f"  Sensitivity : {results['sensitivity_mean']:.4f} +/- {results['sensitivity_std']:.4f}")

    print(f"\n--- Interpretation ---")
    loc = results['locality_mean']
    coh = results['coherence_mean']
    if loc > 2.0:
        print(f"  Locality {loc:.1f} > 2.0: PASS -- localized changes")
    elif loc > 1.0:
        print(f"  Locality {loc:.1f}: PARTIAL -- some locality")
    else:
        print(f"  Locality {loc:.1f}: FAIL -- diffuse changes")

    if coh > 0.5:
        print(f"  Coherence {coh:.2f} > 0.5: PASS -- coherent direction")
    else:
        print(f"  Coherence {coh:.2f}: PARTIAL -- random direction")


if __name__ == "__main__":
    main()
