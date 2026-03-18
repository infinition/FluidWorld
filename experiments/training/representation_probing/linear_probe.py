"""
linear_probe.py -- Linear probing on frozen FluidWorld encoder representations.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

_here = Path(__file__).resolve().parent
_project = _here.parent.parent.parent
sys.path.insert(0, str(_project))

from fluidworld.core.world_model import FluidWorldModel


class LinearProbe(nn.Module):
    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class NpzLabeledDataset(Dataset):
    """Labeled image dataset from .npz with 'images' and 'labels' keys."""

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        if "images" not in data or "labels" not in data:
            raise ValueError("The .npz must contain 'images' and 'labels' keys.")

        images = data["images"]
        labels = data["labels"]

        # Handle 3D images (N, H, W) -> (N, 1, H, W)
        if images.ndim == 3:
            images = images[:, np.newaxis, :, :]
        elif images.ndim == 4:
            # (N, H, W, C) -> (N, C, H, W) if C is the last dim
            if images.shape[-1] in (1, 3):
                images = images.transpose(0, 3, 1, 2)
        else:
            raise ValueError(f"images must be 3D or 4D, got {images.shape}")

        if labels.ndim != 1:
            raise ValueError(f"labels must be 1D, got {labels.shape}")
        if len(images) != len(labels):
            raise ValueError("Number of images != number of labels")

        self.images = torch.from_numpy(images).float()
        if self.images.max() > 1.0:
            self.images = self.images / 255.0
        self.labels = torch.from_numpy(labels).long()

        print(f"  Dataset loaded: {len(self)} images, shape={tuple(self.images.shape[1:])}, "
              f"{len(set(labels.tolist()))} classes")

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]


def build_probe_loader(data_dir: str, batch_size: int, num_workers: int):
    p = Path(data_dir)
    if p.is_file() and p.suffix.lower() == ".npz":
        dataset = NpzLabeledDataset(str(p))
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    # Fallback: class directory (ImageFolder)
    try:
        from torchvision import datasets, transforms
    except Exception as exc:
        raise RuntimeError(
            "Cannot build dataset. Provide a .npz (images/labels) "
            "or install torchvision for ImageFolder."
        ) from exc

    tfm = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]
    )
    dataset = datasets.ImageFolder(root=data_dir, transform=tfm)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def load_world_model(args, device: torch.device) -> FluidWorldModel:
    """Load model with matching hyperparameters."""
    world_model = FluidWorldModel(
        in_channels=args.in_channels,
        d_model=args.d_model,
        stimulus_dim=args.stimulus_dim,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    else:
        state_dict = ckpt
    world_model.load_state_dict(state_dict, strict=False)
    world_model.requires_grad_(False)
    world_model.eval()

    n_params = sum(p.numel() for p in world_model.parameters())
    print(f"  Model loaded: {n_params:,} params, in_channels={args.in_channels}")
    return world_model


def train_probe(args):
    device = torch.device(args.device)

    print(f"Phase 2 -- Linear Probing")
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  Data:       {args.data_dir}")
    print(f"  Classes:    {args.num_classes}")

    world_model = load_world_model(args, device)
    probe = LinearProbe(d_model=args.d_model, num_classes=args.num_classes).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()

    loader = build_probe_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        probe.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                features = world_model.encode(images)["features"]
                z = features.mean(dim=(-2, -1))

            logits = probe(z)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

        epoch_loss = running_loss / max(running_total, 1)
        epoch_acc = running_correct / max(running_total, 1)
        best_acc = max(best_acc, epoch_acc)
        print(f"  Epoch {epoch:2d}/{args.epochs} | Loss: {epoch_loss:.4f} | "
              f"Acc: {epoch_acc:.4f} | Best: {best_acc:.4f}")

    print(f"\n--- Final Result ---")
    print(f"  Best Accuracy: {best_acc:.4f} ({best_acc*100:.1f}%)")
    if best_acc > 0.8:
        print(f"  SUCCESS: representations are linearly separable")
    elif best_acc > 0.5:
        print(f"  PARTIAL: features are useful but not yet optimal")
    else:
        print(f"  FAILURE: features likely collapsed or non-informative")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Linear probing on FluidWorld representations"
    )
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Checkpoint Phase 1 (.pt)")
    parser.add_argument("--data-dir", type=str, required=True,
                        help=".npz (images+labels) or ImageFolder directory")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--in-channels", type=int, default=1,
                        help="Input channels (1=grayscale, 3=RGB)")
    parser.add_argument("--d-model", type=int, default=128,
                        help="Model dimension (must match checkpoint)")
    parser.add_argument("--stimulus-dim", type=int, default=1,
                        help="Stimulus dimension (must match the checkpoint)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    train_probe(args)
