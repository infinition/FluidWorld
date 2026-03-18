"""
visualize_pca_features.py -- PCA visualization of encoder dense features.

Computes PCA on patch-level features from the FluidWorld encoder,
maps the top 3 components to RGB channels. This reveals whether the
learned representations preserve spatial structure (object boundaries,
semantic regions) despite the PDE-based architecture.

Inspired by V-JEPA 2.1 (Mur-Labadia et al., 2026) Figure 1/14.

Usage:
    python tools/visualize_pca_features.py --checkpoint training/best.pth --data-dir data/ucf101_64 --out-dir paper/figures/pca
    python tools/visualize_pca_features.py --checkpoint training/best.pth --data-dir data/ucf101_64 --out-dir paper/figures/pca --n-samples 16
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

_here = Path(__file__).resolve().parent
_project = _here.parent
sys.path.insert(0, str(_project))

from fluidworld.core.world_model_v2 import FluidWorldModelV2


def load_frames(data_dir: str, n_samples: int, device: torch.device) -> torch.Tensor:
    """Load random frames from the dataset."""
    is_mnist = data_dir.endswith(".npy")

    if is_mnist:
        from fluidworld.core.video_dataset import MovingMNISTDataset
        dataset = MovingMNISTDataset(npy_file=data_dir, bptt_steps=2)
    else:
        from fluidworld.core.video_dataset import PureVideoDataset
        dataset = PureVideoDataset(data_dir=data_dir, bptt_steps=2)

    # Sample random indices
    indices = torch.randperm(len(dataset))[:n_samples]
    frames = []
    for idx in indices:
        clip = dataset[idx.item()]
        frames.append(clip[0])  # first frame of the pair

    return torch.stack(frames).to(device)


def extract_features(model: FluidWorldModelV2, frames: torch.Tensor) -> torch.Tensor:
    """Run encoder and return patch features (B, C, H_p, W_p)."""
    model.eval()
    with torch.no_grad():
        out = model.encode(frames)
        features = out["features"]  # (B, C, H_p, W_p)
    return features


def pca_to_rgb(features: torch.Tensor, target_size: tuple = None) -> np.ndarray:
    """Map top-3 PCA components of patch features to RGB.

    Args:
        features: (B, C, H_p, W_p) encoder features
        target_size: (H, W) to upsample the PCA maps to original resolution

    Returns:
        (B, H, W, 3) numpy array of RGB images, values in [0, 1]
    """
    B, C, H_p, W_p = features.shape

    # Flatten spatial dims: (B*H_p*W_p, C)
    flat = features.permute(0, 2, 3, 1).reshape(-1, C).float().cpu()

    # Center the features
    mean = flat.mean(dim=0, keepdim=True)
    centered = flat - mean

    # PCA via SVD on centered features
    # Use a subset if too many patches (for numerical stability)
    max_patches = 50000
    if centered.shape[0] > max_patches:
        subset_idx = torch.randperm(centered.shape[0])[:max_patches]
        subset = centered[subset_idx]
    else:
        subset = centered

    U, S, Vt = torch.linalg.svd(subset, full_matrices=False)
    # Project all features onto top 3 components
    components = Vt[:3]  # (3, C)
    projected = centered @ components.T  # (B*H_p*W_p, 3)

    # Normalize each component to [0, 1]
    for i in range(3):
        col = projected[:, i]
        lo, hi = col.quantile(0.02), col.quantile(0.98)
        projected[:, i] = ((col - lo) / (hi - lo + 1e-8)).clamp(0, 1)

    # Reshape to spatial
    rgb_maps = projected.reshape(B, H_p, W_p, 3).numpy()

    # Upsample if needed
    if target_size is not None:
        H, W = target_size
        rgb_upsampled = []
        for i in range(B):
            img = torch.from_numpy(rgb_maps[i]).permute(2, 0, 1).unsqueeze(0)
            img_up = F.interpolate(img, size=(H, W), mode="bilinear", align_corners=False)
            rgb_upsampled.append(img_up.squeeze(0).permute(1, 2, 0).numpy())
        rgb_maps = np.stack(rgb_upsampled)

    return rgb_maps


def save_grid(images: np.ndarray, originals: np.ndarray, path: str, nrow: int = 8):
    """Save a side-by-side grid: original images on top, PCA maps below."""
    try:
        from PIL import Image
    except ImportError:
        print("PIL not available, saving raw numpy arrays instead")
        np.save(path.replace(".png", ".npy"), images)
        return

    B = min(images.shape[0], nrow)

    # Build grid rows
    cell_h, cell_w = images.shape[1], images.shape[2]
    grid_h = cell_h * 2 + 4  # 2 rows + gap
    grid_w = cell_w * B + (B - 1) * 2  # columns + gaps

    grid = np.ones((grid_h, grid_w, 3), dtype=np.float32)

    for i in range(B):
        x_offset = i * (cell_w + 2)
        # Top row: original image
        orig = originals[i]
        if orig.shape[0] in (1, 3):  # (C, H, W) -> (H, W, 3)
            orig = np.transpose(orig, (1, 2, 0))
        if orig.shape[-1] == 1:
            orig = np.repeat(orig, 3, axis=-1)
        # Resize original to match PCA map size
        orig_pil = Image.fromarray((np.clip(orig, 0, 1) * 255).astype(np.uint8))
        orig_pil = orig_pil.resize((cell_w, cell_h), Image.BILINEAR)
        orig_resized = np.array(orig_pil).astype(np.float32) / 255.0
        grid[:cell_h, x_offset:x_offset + cell_w] = orig_resized

        # Bottom row: PCA features
        grid[cell_h + 4:, x_offset:x_offset + cell_w] = images[i]

    # Save
    grid_uint8 = (np.clip(grid, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(grid_uint8).save(path)
    print(f"Saved: {path}")


def save_individual(images: np.ndarray, originals: np.ndarray, out_dir: str):
    """Save individual PCA maps and originals."""
    try:
        from PIL import Image
    except ImportError:
        return

    os.makedirs(os.path.join(out_dir, "originals"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "pca"), exist_ok=True)

    for i in range(images.shape[0]):
        # PCA map
        pca_uint8 = (np.clip(images[i], 0, 1) * 255).astype(np.uint8)
        Image.fromarray(pca_uint8).save(os.path.join(out_dir, "pca", f"pca_{i:04d}.png"))

        # Original
        orig = originals[i]
        if orig.shape[0] in (1, 3):
            orig = np.transpose(orig, (1, 2, 0))
        if orig.shape[-1] == 1:
            orig = np.repeat(orig, 3, axis=-1)
        orig_uint8 = (np.clip(orig, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(orig_uint8).save(os.path.join(out_dir, "originals", f"orig_{i:04d}.png"))


def main():
    parser = argparse.ArgumentParser(description="PCA visualization of FluidWorld dense features")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--data-dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--out-dir", type=str, default="paper/figures/pca", help="Output directory")
    parser.add_argument("--n-samples", type=int, default=16, help="Number of frames to visualize")
    parser.add_argument("--d-model", type=int, default=128, help="Model d_model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--upsample", type=int, default=256, help="Upsample PCA maps to this size (0=no upsample)")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    # Detect input type
    is_mnist = args.data_dir.endswith(".npy")
    in_channels = 1 if is_mnist else 3

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = FluidWorldModelV2(in_channels=in_channels, d_model=args.d_model).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"  Loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    # Load frames
    print(f"Loading {args.n_samples} frames from {args.data_dir}")
    frames = load_frames(args.data_dir, args.n_samples, device)
    print(f"  Frames shape: {frames.shape}")

    # Extract features
    print("Extracting encoder features...")
    features = extract_features(model, frames)
    print(f"  Features shape: {features.shape}")

    # PCA -> RGB
    target_size = (args.upsample, args.upsample) if args.upsample > 0 else None
    print(f"Computing PCA (top 3 components -> RGB)...")
    pca_maps = pca_to_rgb(features, target_size=target_size)
    print(f"  PCA maps shape: {pca_maps.shape}")

    # Save
    originals = frames.cpu().numpy()
    save_grid(pca_maps, originals, os.path.join(args.out_dir, "pca_grid.png"), nrow=min(8, args.n_samples))
    save_individual(pca_maps, originals, args.out_dir)

    # Summary stats
    print(f"\nFeature statistics:")
    flat = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1]).float().cpu()
    print(f"  Mean: {flat.mean():.4f}")
    print(f"  Std:  {flat.std():.4f}")
    print(f"  Dead dims (std < 0.01): {(flat.std(dim=0) < 0.01).sum().item()}/{flat.shape[1]}")

    # Explained variance
    centered = flat - flat.mean(dim=0)
    _, S, _ = torch.linalg.svd(centered[:10000] if centered.shape[0] > 10000 else centered, full_matrices=False)
    var_explained = S[:3].pow(2) / S.pow(2).sum()
    print(f"  Top-3 PCA variance explained: {var_explained[0]:.1%}, {var_explained[1]:.1%}, {var_explained[2]:.1%}")
    print(f"  Total top-3: {var_explained.sum():.1%}")

    print(f"\nDone. Results in {args.out_dir}/")


if __name__ == "__main__":
    main()
