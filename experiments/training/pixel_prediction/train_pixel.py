"""
train_pixel.py -- FluidWorld v2: pixel prediction training with TensorBoard visualization.
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

from fluidworld.core.world_model_v2 import FluidWorldModelV2


def make_image_grid(frames: torch.Tensor, nrow: int = 8) -> torch.Tensor:
    """Create an image grid for TensorBoard."""
    return vutils.make_grid(frames[:nrow], nrow=nrow, normalize=False, pad_value=1.0)


def train_phase1(args):
    device = torch.device(args.device)

    log_dir = os.path.join("runs", "phase1_pixel")
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

    # ── 2. Model v2 ──
    loss_type = args.loss_type  # "auto", "bce", or "mse"
    print(f"Init FluidWorld v2 (in_channels={in_channels}, d_model={args.d_model}, loss={loss_type})")
    model = FluidWorldModelV2(
        in_channels=in_channels,
        d_model=args.d_model,
        stimulus_dim=1,
        n_encoder_layers=3,
        max_steps_encoder=args.max_steps,
        belief_spatial_hw=args.belief_hw,
        n_belief_evolve=3,
        recon_weight=args.recon_weight,
        pred_weight=args.pred_weight,
        loss_type=loss_type,
        var_weight=args.var_weight,
        var_target=args.var_target,
        grad_weight=args.grad_weight,
        # Phase 1.5: JEPA-inspired improvements
        deep_supervision=args.deep_supervision,
        deep_supervision_weight=args.deep_supervision_weight,
        rdm_reg=args.rdm_reg,
        rdm_weight=args.rdm_weight,
        input_masking=args.input_masking,
        mask_ratio=args.mask_ratio,
        # Anisotropic diffusion (content-gated routing)
        anisotropic_diffusion=args.anisotropic_diffusion,
        # Bio mechanisms
        use_fatigue=args.use_fatigue,
        fatigue_cost=args.fatigue_cost,
        fatigue_recovery=args.fatigue_recovery,
        use_inhibition=args.use_inhibition,
        inhibition_strength=args.inhibition_strength,
        use_memory_pump=args.use_memory_pump,
        use_hebbian=args.use_hebbian,
        hebbian_lr=args.hebbian_lr,
        hebbian_decay=args.hebbian_decay,
    ).to(device)

    params = model.count_parameters()
    print(f"  Parameters: {params['total']:,} total ({params['trainable']:,} trainable)")
    print(f"    Encoder: {params['encoder']:,}  BeliefField: {params['belief_field']:,}  Decoder: {params['decoder']:,}")

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
    print(f"  batch_size={args.batch_size}, bptt={args.bptt_steps}, max_steps={args.max_steps}, max_batches/epoch={batches_per_epoch}")
    print(f"  recon_weight={args.recon_weight}, pred_weight={args.pred_weight}")
    print(f"  eq_weight={args.eq_weight}, eq_target={args.eq_target}")
    bio_flags = []
    if args.use_fatigue:
        bio_flags.append(f"Fatigue(cost={args.fatigue_cost})")
    if args.use_inhibition:
        bio_flags.append(f"Inhibition(str={args.inhibition_strength})")
    if args.use_memory_pump:
        bio_flags.append("MemoryPump")
    if args.use_hebbian:
        bio_flags.append(f"Hebbian(lr={args.hebbian_lr})")
    print(f"  Bio: {', '.join(bio_flags) if bio_flags else 'None'}")
    jepa_flags = []
    if args.deep_supervision:
        jepa_flags.append(f"DeepSupervision(w={args.deep_supervision_weight})")
    if args.rdm_reg:
        jepa_flags.append(f"RDMReg(w={args.rdm_weight})")
    if args.input_masking:
        jepa_flags.append(f"InputMasking(ratio={args.mask_ratio})")
    if args.anisotropic_diffusion:
        jepa_flags.append("AnisotropicDiffusion(content-gated)")
    if args.scheduled_sampling:
        jepa_flags.append(f"ScheduledSampling(start={args.ss_start}, end={args.ss_end}, warmup={args.ss_warmup_steps})")
    if jepa_flags:
        print(f"  JEPA-inspired: {', '.join(jepa_flags)}")
    model.train()

    # Keep a reference batch for visualization
    viz_batch = None

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_metrics = {
            "total_loss": 0.0, "recon_loss": 0.0, "pred_loss": 0.0,
            "pde_alive": 0.0, "var_loss": 0.0, "grad_loss": 0.0,
            "deep_loss": 0.0, "rdm_loss": 0.0, "step_energy": 0.0,
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

            # Keep first batch for visualization
            if viz_batch is None:
                viz_batch = images[:8].clone()

            stim_t = torch.zeros(B, 1, device=device)
            optimizer.zero_grad()

            batch_metrics = {
                "total_loss": 0.0, "recon_loss": 0.0, "pred_loss": 0.0,
                "pde_alive": 0.0, "var_loss": 0.0, "grad_loss": 0.0, "step_energy": 0.0,
            }
            current_state = None
            n_steps = T_total - 1

            # Phase 2: Scheduled Sampling -- teacher forcing ratio
            if args.scheduled_sampling:
                ss_progress = min(global_step / args.ss_warmup_steps, 1.0)
                tf_ratio = args.ss_start + (args.ss_end - args.ss_start) * ss_progress
            else:
                tf_ratio = 1.0  # full teacher forcing

            prev_pred_frame = None  # previous step prediction
            n_autoreg_steps = 0
            autoreg_pred_loss = 0.0

            for t in range(n_steps):
                x_next = images[:, t + 1]

                # Scheduled sampling: GT or previous prediction?
                if t == 0 or prev_pred_frame is None or not args.scheduled_sampling:
                    x_current = images[:, t]
                    is_autoreg = False
                else:
                    use_teacher = random.random() < tf_ratio
                    if use_teacher:
                        x_current = images[:, t]
                        is_autoreg = False
                    else:
                        x_current = prev_pred_frame.detach()
                        is_autoreg = True

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
                        eq_weight=args.eq_weight,
                        eq_target=args.eq_target,
                    )
                    step_loss = out["loss"] / n_steps

                scaler.scale(step_loss).backward()

                # Keep prediction for next step (scheduled sampling)
                prev_pred_frame = out["x_pred"]

                batch_metrics["total_loss"] += step_loss.item()
                batch_metrics["recon_loss"] += out["recon_loss"].item() / n_steps
                batch_metrics["pred_loss"] += out["pred_loss"].item() / n_steps
                batch_metrics["pde_alive"] += out["pde_alive_loss"].item() / n_steps
                batch_metrics["var_loss"] = out.get("var_loss", torch.tensor(0.0)).item()
                batch_metrics["grad_loss"] = out.get("grad_loss", torch.tensor(0.0)).item()
                batch_metrics["deep_loss"] = out.get("deep_loss", torch.tensor(0.0)).item()
                batch_metrics["rdm_loss"] = out.get("rdm_loss", torch.tensor(0.0)).item()
                batch_metrics["step_energy"] += out.get("mean_step_energy", 0.0) / n_steps

                # Track autoregressive prediction loss separately
                if is_autoreg:
                    n_autoreg_steps += 1
                    autoreg_pred_loss += out["pred_loss"].item()

                current_state = out["next_state"]

            # Store autoreg metrics for logging
            batch_metrics["tf_ratio"] = tf_ratio
            batch_metrics["autoreg_pred_loss"] = (
                autoreg_pred_loss / max(n_autoreg_steps, 1)
            )
            batch_metrics["n_autoreg_steps"] = n_autoreg_steps

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # Feature monitoring
            with torch.no_grad():
                sample_features = model.encode(images[:, 0])["features"]
                feature_std = sample_features.std(dim=0).mean().item()
                epoch_feature_std += feature_std

                # Effective rank
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

                        # Spatial variance: are positions different from each other?
                        # High = good (spatial info preserved), Low = blob (PDE homogenized)
                        # sample_features: (B, C, H, W) — compute std across H,W per sample per channel
                        spatial_std = sample_features.std(dim=(2, 3)).mean().item()
                        writer.add_scalar("Monitor/Spatial_Std", spatial_std, global_step)

                        if rank_ratio < 0.1:
                            pbar.write(f"  COLLAPSE: rank_eff={effective_rank:.1f}/{max_rank} ({rank_ratio:.1%})")
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
            writer.add_scalar("PDE/Alive_Loss", batch_metrics["pde_alive"], global_step)
            writer.add_scalar("Train/Var_Loss", batch_metrics.get("var_loss", 0.0), global_step)
            writer.add_scalar("Train/Grad_Loss", batch_metrics.get("grad_loss", 0.0), global_step)
            writer.add_scalar("Train/Deep_Loss", batch_metrics.get("deep_loss", 0.0), global_step)
            writer.add_scalar("Train/RDM_Loss", batch_metrics.get("rdm_loss", 0.0), global_step)
            writer.add_scalar("PDE/Step_Energy", batch_metrics["step_energy"], global_step)
            writer.add_scalar("Monitor/Feature_Std", feature_std, global_step)
            writer.add_scalar("Monitor/LR", optimizer.param_groups[0]["lr"], global_step)

            # Anisotropic diffusion gate monitoring
            if args.anisotropic_diffusion:
                writer.add_scalar("PDE/Gate_Mean", out.get("gate_mean", 1.0), global_step)
                writer.add_scalar("PDE/Gate_Std", out.get("gate_std", 0.0), global_step)

            # Phase 2: Scheduled Sampling monitoring
            if args.scheduled_sampling:
                writer.add_scalar("Phase2/TF_Ratio", batch_metrics["tf_ratio"], global_step)
                writer.add_scalar("Phase2/Autoreg_Pred_Loss", batch_metrics["autoreg_pred_loss"], global_step)
                writer.add_scalar("Phase2/N_Autoreg_Steps", batch_metrics["n_autoreg_steps"], global_step)

            # Bio mechanisms monitoring
            bio_stats = out.get("bio_stats", {})
            for key, val in bio_stats.items():
                writer.add_scalar(f"Bio/{key}", val, global_step)

            # TensorBoard images (every 200 steps)
            if global_step % 200 == 0 and viz_batch is not None:
                with torch.no_grad():
                    model.eval()
                    vb = viz_batch.to(device)
                    B_viz = vb.shape[0]

                    # Reconstruct frame 0
                    enc = model.encode(vb[:, 0])
                    recon = model.decode_to_pixels(enc["features"])

                    # Predict frame 1 from frame 0
                    stim_viz = torch.zeros(B_viz, 1, device=device)
                    state = model.belief_field.init_state(B_viz, device, vb.dtype)
                    state = model.belief_field.write(state, enc["features"])
                    state = model.belief_field.evolve(state, stimulus=stim_viz)
                    z_pred = model.belief_field.read_spatial(
                        state, (enc["features"].shape[2], enc["features"].shape[3]))
                    pred_frame = model.decode_to_pixels(z_pred)

                    # Rollout 5 frames
                    rollout = model.rollout(vb[:, 0], stim_viz, n_steps=5)

                    # Log images
                    writer.add_image("Viz/0_Input",
                                     make_image_grid(vb[:, 0]), global_step)
                    writer.add_image("Viz/1_Reconstruction",
                                     make_image_grid(recon), global_step)
                    writer.add_image("Viz/2_GT_Next",
                                     make_image_grid(vb[:, 1]), global_step)
                    writer.add_image("Viz/3_Predicted_Next",
                                     make_image_grid(pred_frame), global_step)

                    # Rollout: each future frame
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
                ckpt_dir = Path("checkpoints/phase1_pixel")
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                save_path = ckpt_dir / f"model_step_{global_step}.pt"
                checkpoint = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }
                torch.save(checkpoint, save_path)
                pbar.write(f"  [ckpt] {save_path.name}")

        # End of epoch
        num_batches = batch_count if batch_count > 0 else 1
        for k in epoch_metrics:
            epoch_metrics[k] /= num_batches
        avg_std = epoch_feature_std / num_batches

        print(f"\n--- Epoch {epoch} ---")
        print(f"  Loss={epoch_metrics['total_loss']:.4f}  "
              f"Recon={epoch_metrics['recon_loss']:.4f}  "
              f"Pred={epoch_metrics['pred_loss']:.4f}  "
              f"PDE={epoch_metrics['pde_alive']:.4f}  "
              f"Energy={epoch_metrics['step_energy']:.4f}")
        print(f"  Feature Std={avg_std:.4f}", end="")
        if avg_std < 0.01:
            print("  WARNING: COLLAPSE!")
        elif avg_std < 0.1:
            print("  WARNING: low variance")
        else:
            print("  OK")

        # Epoch checkpoint
        ckpt_dir = Path("checkpoints/phase1_pixel")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_epoch = {
            'epoch': epoch + 1,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        torch.save(checkpoint_epoch, ckpt_dir / f"model_epoch_{epoch}.pt")

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FluidWorld v2 -- Pixel Prediction Training")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bptt-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.04)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--belief-hw", type=int, default=16)
    parser.add_argument("--recon-weight", type=float, default=1.0,
                        help="Reconstruction loss weight")
    parser.add_argument("--pred-weight", type=float, default=1.0,
                        help="Prediction loss weight")
    parser.add_argument("--eq-weight", type=float, default=0.5,
                        help="PDE-Alive loss weight")
    parser.add_argument("--eq-target", type=float, default=1.2,
                        help="Target step energy for PDE")
    parser.add_argument("--loss-type", type=str, default="auto",
                        choices=["auto", "bce", "mse"],
                        help="Loss type: auto (bce for 1ch, mse for 3ch), bce, or mse")
    parser.add_argument("--var-weight", type=float, default=0.0,
                        help="Variance loss weight (RESEARCH.md #7). 0=off, 0.5=moderate, 1.0=strong")
    parser.add_argument("--var-target", type=float, default=1.0,
                        help="Target std per channel for variance loss")
    parser.add_argument("--grad-weight", type=float, default=0.0,
                        help="Gradient/edge loss weight. Forces spatial structure. "
                             "0=off, 1.0=moderate, 2.0=strong")
    parser.add_argument("--max-batches-per-epoch", type=int, default=0,
                        help="Cap batches per epoch (0 = full dataset). "
                             "Ex: 2000 = ~25min/epoch on RTX 4070 Ti")

    # Phase 1.5: JEPA-inspired improvements (Mar 2026)
    parser.add_argument("--deep-supervision", action="store_true", default=False,
                        help="V-JEPA 2.1: supervise intermediate PDE layers")
    parser.add_argument("--deep-supervision-weight", type=float, default=0.3,
                        help="Weight for deep supervision loss")
    parser.add_argument("--rdm-reg", action="store_true", default=False,
                        help="Rectified LpJEPA: RDMReg sparse distribution matching")
    parser.add_argument("--rdm-weight", type=float, default=0.1,
                        help="Weight for RDMReg loss")
    parser.add_argument("--input-masking", action="store_true", default=False,
                        help="C-JEPA inspired: mask input patches (world model test)")
    parser.add_argument("--mask-ratio", type=float, default=0.25,
                        help="Fraction of input patches to mask")
    parser.add_argument("--anisotropic-diffusion", action="store_true", default=False,
                        help="Content-gated anisotropic diffusion (attention-free routing)")

    # Phase 2: Scheduled Sampling
    parser.add_argument("--scheduled-sampling", action="store_true", default=False,
                        help="Phase 2: progressively replace teacher forcing with autoregressive")
    parser.add_argument("--ss-start", type=float, default=1.0,
                        help="Teacher forcing ratio at start (1.0=full teacher)")
    parser.add_argument("--ss-end", type=float, default=0.2,
                        help="Teacher forcing ratio at end (0.0=full autoregressive)")
    parser.add_argument("--ss-warmup-steps", type=int, default=8000,
                        help="Steps over which TF ratio decays from ss-start to ss-end")

    parser.add_argument("--resume", type=str, default="",
                        help="Checkpoint to resume from")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    # Bio mechanisms (RESEARCH.md)
    parser.add_argument("--use-fatigue", action="store_true", default=True,
                        help="Synaptic Fatigue (#14)")
    parser.add_argument("--no-fatigue", dest="use_fatigue", action="store_false")
    parser.add_argument("--fatigue-cost", type=float, default=0.1)
    parser.add_argument("--fatigue-recovery", type=float, default=0.02)
    parser.add_argument("--use-inhibition", action="store_true", default=True,
                        help="Lateral Inhibition (#16)")
    parser.add_argument("--no-inhibition", dest="use_inhibition", action="store_false")
    parser.add_argument("--inhibition-strength", type=float, default=0.3)
    parser.add_argument("--use-memory-pump", action="store_true", default=True,
                        help="Global Memory Pump (#3)")
    parser.add_argument("--no-memory-pump", dest="use_memory_pump", action="store_false")
    parser.add_argument("--use-hebbian", action="store_true", default=True,
                        help="Hebbian Diffusion (#14)")
    parser.add_argument("--no-hebbian", dest="use_hebbian", action="store_false")
    parser.add_argument("--hebbian-lr", type=float, default=0.01)
    parser.add_argument("--hebbian-decay", type=float, default=0.99)

    train_phase1(parser.parse_args())
