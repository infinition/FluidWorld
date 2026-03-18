"""
train_transformer.py -- Transformer baseline for PDE vs attention scaling comparison.
"""

import argparse
import math
import os
import random
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision.utils as vutils

_here = Path(__file__).resolve().parent
_project = _here.parent.parent.parent
sys.path.insert(0, str(_project))

from fluidworld.core.transformer_world_model import TransformerWorldModel


def make_image_grid(frames: torch.Tensor, nrow: int = 8) -> torch.Tensor:
    return vutils.make_grid(frames[:nrow], nrow=nrow, normalize=False, pad_value=1.0)


def train_transformer(args):
    device = torch.device(args.device)

    log_dir = os.path.join("runs", "phase2_transformer")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard : tensorboard --logdir=runs")

    # ── 1. Data ──
    print(f"Loading data: {args.data_dir}")

    is_mnist = args.data_dir.endswith(".npy")
    in_channels = 1 if is_mnist else 3

    if is_mnist:
        from fluidworld.core.video_dataset import MovingMNISTDataset
        dataset = MovingMNISTDataset(
            npy_file=args.data_dir,
            bptt_steps=args.bptt_steps
        )
    else:
        from fluidworld.core.video_dataset import PureVideoDataset
        dataset = PureVideoDataset(
            data_dir=args.data_dir,
            bptt_steps=args.bptt_steps
        )

    n_workers = 0 if sys.platform == "win32" else min(4, os.cpu_count() or 0)

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # ── 2. Transformer model ──
    print(f"Init TransformerWorldModel (in_channels={in_channels}, d_model={args.d_model})")
    model = TransformerWorldModel(
        in_channels=in_channels,
        d_model=args.d_model,
        n_encoder_layers=args.n_encoder_layers,
        n_temporal_layers=args.n_temporal_layers,
        n_heads=args.n_heads,
        ffn_dim=args.ffn_dim,
        patch_size=4,
        spatial_hw=16,
        recon_weight=args.recon_weight,
        pred_weight=args.pred_weight,
        var_weight=args.var_weight,
        var_target=args.var_target,
        grad_weight=args.grad_weight,
    ).to(device)

    params = model.count_parameters()
    print(f"  Parameters: {params['total']:,} total ({params['trainable']:,} trainable)")
    print(f"    Encoder: {params['encoder']:,}  Temporal: {params['temporal']:,}  Decoder: {params['decoder']:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd
    )
    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    # LR warmup + cosine decay
    batches_per_epoch = args.max_batches_per_epoch or len(train_loader)
    total_steps = args.epochs * batches_per_epoch
    warmup_steps = min(500, total_steps // 20)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume
    start_epoch = 1
    global_step = 0

    if args.resume:
        print(f"\nResuming from: {args.resume}")
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch']
                global_step = checkpoint['global_step']
                print(f"-> Resumed: Epoch {start_epoch}, Step {global_step}")
            else:
                state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
                model.load_state_dict(state_dict, strict=False)
                print("-> Legacy format: weights loaded, optimizer reset")

    # ── 3. Training loop ──
    print(f"\nTraining: {args.epochs} epochs, {len(train_loader)} batches/epoch")
    print(f"  batch_size={args.batch_size}, bptt={args.bptt_steps}")
    print(f"  recon_weight={args.recon_weight}, pred_weight={args.pred_weight}")
    print(f"  Architecture: {args.n_encoder_layers} encoder blocks + {args.n_temporal_layers} temporal blocks")
    print(f"  d_model={args.d_model}, heads={args.n_heads}, ffn_dim={args.ffn_dim}")
    model.train()

    viz_batch = None

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_metrics = {
            "total_loss": 0.0, "recon_loss": 0.0, "pred_loss": 0.0,
            "var_loss": 0.0, "grad_loss": 0.0,
        }
        epoch_feature_std = 0.0

        max_batches = args.max_batches_per_epoch or len(train_loader)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", total=max_batches)
        batch_count = 0

        for images in pbar:
            if batch_count >= max_batches:
                break
            batch_count += 1
            images = images.to(device, non_blocking=True)
            B, T_total, C, H, W = images.shape

            if viz_batch is None:
                viz_batch = images[:8].clone()

            stim_t = torch.zeros(B, 1, device=device)
            optimizer.zero_grad()

            batch_metrics = {
                "total_loss": 0.0, "recon_loss": 0.0, "pred_loss": 0.0,
                "var_loss": 0.0, "grad_loss": 0.0,
            }
            current_state = None
            n_steps = T_total - 1

            for t in range(n_steps):
                x_current = images[:, t]
                x_next = images[:, t + 1]

                with autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    out = model(
                        x_current=x_current,
                        stimulus=stim_t,
                        x_next=x_next,
                        current_state=current_state,
                    )
                    step_loss = out["loss"] / n_steps

                scaler.scale(step_loss).backward()

                batch_metrics["total_loss"] += step_loss.item()
                batch_metrics["recon_loss"] += out["recon_loss"].item() / n_steps
                batch_metrics["pred_loss"] += out["pred_loss"].item() / n_steps
                batch_metrics["var_loss"] = out.get("var_loss", torch.tensor(0.0)).item()
                batch_metrics["grad_loss"] = out.get("grad_loss", torch.tensor(0.0)).item()

                current_state = out["next_state"]

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # Feature monitoring
            with torch.no_grad():
                sample_features = model.encode(images[:, 0])["features"]
                feature_std = sample_features.std(dim=0).mean().item()
                epoch_feature_std += feature_std

                if global_step % 100 == 0:
                    centered = sample_features - sample_features.mean(dim=0)
                    try:
                        s = torch.linalg.svdvals(centered)
                        p = s / s.sum()
                        entropy = -(p * torch.log(p + 1e-10)).sum().item()
                        effective_rank = math.exp(entropy)
                        max_rank = sample_features.shape[1]
                        rank_ratio = effective_rank / max_rank

                        writer.add_scalar("Monitor/Effective_Rank", effective_rank, global_step)
                        writer.add_scalar("Monitor/Rank_Ratio", rank_ratio, global_step)

                        dim_stds = sample_features.std(dim=0)
                        writer.add_scalar("Monitor/Dim_Std_Min", dim_stds.min().item(), global_step)
                        writer.add_scalar("Monitor/Dim_Std_Max", dim_stds.max().item(), global_step)
                        writer.add_scalar("Monitor/Dim_Std_Median", dim_stds.median().item(), global_step)
                        dead_dims = (dim_stds < 0.1).sum().item()
                        writer.add_scalar("Monitor/Dead_Dims", dead_dims, global_step)

                        spatial_std = sample_features.std(dim=(2, 3)).mean().item()
                        writer.add_scalar("Monitor/Spatial_Std", spatial_std, global_step)
                    except Exception:
                        pass

            pbar.set_postfix({
                "Loss": f"{batch_metrics['total_loss']:.3f}",
                "Recon": f"{batch_metrics['recon_loss']:.3f}",
                "Pred": f"{batch_metrics['pred_loss']:.3f}",
                "Std": f"{feature_std:.3f}",
            })

            # TensorBoard scalars
            writer.add_scalar("Train/Total_Loss", batch_metrics["total_loss"], global_step)
            writer.add_scalar("Train/Recon_Loss", batch_metrics["recon_loss"], global_step)
            writer.add_scalar("Train/Pred_Loss", batch_metrics["pred_loss"], global_step)
            writer.add_scalar("Train/Var_Loss", batch_metrics.get("var_loss", 0.0), global_step)
            writer.add_scalar("Train/Grad_Loss", batch_metrics.get("grad_loss", 0.0), global_step)
            writer.add_scalar("Monitor/Feature_Std", feature_std, global_step)
            writer.add_scalar("Monitor/LR", optimizer.param_groups[0]["lr"], global_step)

            # TensorBoard images (every 200 steps)
            if global_step % 200 == 0 and viz_batch is not None:
                with torch.no_grad():
                    model.eval()
                    vb = viz_batch.to(device)
                    B_viz = vb.shape[0]

                    enc = model.encode(vb[:, 0])
                    recon = model.decode_to_pixels(enc["features"])

                    # Rollout 5 frames
                    stim_viz = torch.zeros(B_viz, 1, device=device)
                    rollout = model.rollout(vb[:, 0], stim_viz, n_steps=5)

                    # Log images
                    writer.add_image("Viz/0_Input",
                                     make_image_grid(vb[:, 0]), global_step)
                    writer.add_image("Viz/1_Reconstruction",
                                     make_image_grid(recon), global_step)
                    writer.add_image("Viz/2_GT_Next",
                                     make_image_grid(vb[:, 1]), global_step)

                    # Rollout
                    for h in range(min(5, rollout.shape[1])):
                        writer.add_image(f"Rollout/h{h+1}",
                                         make_image_grid(rollout[:, h]), global_step)
                        if h < vb.shape[1] - 1:
                            writer.add_image(f"Rollout/GT_h{h+1}",
                                             make_image_grid(vb[:, h + 1]), global_step)

                    model.train()

            # LR scheduler
            if global_step > 0:
                scheduler.step()

            for k in epoch_metrics:
                epoch_metrics[k] += batch_metrics[k]
            global_step += 1

            # Checkpoint
            if global_step % 500 == 0:
                ckpt_dir = Path("checkpoints/phase2_transformer")
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                save_path = ckpt_dir / f"model_step_{global_step}.pt"
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                }
                torch.save(checkpoint, save_path)
                pbar.write(f"  [ckpt] {save_path.name}")

        # Epoch summary
        n = batch_count
        for k in epoch_metrics:
            epoch_metrics[k] /= max(n, 1)
        avg_std = epoch_feature_std / max(n, 1)

        print(f"--- Epoch {epoch} ---")
        print(f"  Loss={epoch_metrics['total_loss']:.4f}"
              f"  Recon={epoch_metrics['recon_loss']:.4f}"
              f"  Pred={epoch_metrics['pred_loss']:.4f}"
              f"  Feature Std={avg_std:.4f}")

        # Save epoch checkpoint
        ckpt_dir = Path("checkpoints/phase2_transformer")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_epoch = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch + 1,
            "global_step": global_step,
        }
        torch.save(checkpoint_epoch, ckpt_dir / f"model_epoch_{epoch}.pt")

    writer.close()
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer Baseline - Scaling Experiment")

    # Data
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bptt-steps", type=int, default=4,
                        help="Temporal window length")

    # Architecture
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-encoder-layers", type=int, default=2,
                        help="Number of Transformer encoder blocks")
    parser.add_argument("--n-temporal-layers", type=int, default=1,
                        help="Number of Transformer temporal blocks")
    parser.add_argument("--n-heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--ffn-dim", type=int, default=384,
                        help="FFN hidden dimension (adjust to match PDE param count)")

    # Losses
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--pred-weight", type=float, default=1.0)
    parser.add_argument("--var-weight", type=float, default=0.5)
    parser.add_argument("--var-target", type=float, default=1.0)
    parser.add_argument("--grad-weight", type=float, default=1.0)

    # Training
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=0.04)
    parser.add_argument("--max-batches-per-epoch", type=int, default=None)

    # Resume
    parser.add_argument("--resume", type=str, default="")

    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    train_transformer(parser.parse_args())
