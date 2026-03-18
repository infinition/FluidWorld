# Phase 1 -- Pixel-Space PDE World Model

Core experiment. Trains FluidWorldModelV2 (reaction-diffusion PDE) on UCF-101 video
sequences at 64x64 resolution. The model learns to reconstruct the current frame and
predict the next frame through latent PDE dynamics.

## Status

Done. Results at step 8000: Recon Loss = 0.001, Pred Loss = 0.003.

## Prerequisites

Preprocessed UCF-101 data (see `scripts/prepare_ucf101.py`).

## Run

```bash
python experiments/training/pixel_prediction/train_pixel.py --data-dir data/ucf101_64 --epochs 200 --batch-size 16 --bptt-steps 4 --max-steps 6 --lr 3e-4 --max-batches-per-epoch 2000
```

## Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | required | Path to preprocessed .npy video directory |
| `--epochs` | 200 | Training epochs |
| `--batch-size` | 16 | Batch size |
| `--bptt-steps` | 4 | Backprop-through-time steps |
| `--max-steps` | 6 | Max PDE integration steps per layer |
| `--lr` | 3e-4 | Learning rate |
| `--max-batches-per-epoch` | 2000 | Batches per epoch (caps long datasets) |

## Outputs

- Checkpoints: `checkpoints/phase1_pixel/`
- TensorBoard: `runs/phase1_pixel/`

## TensorBoard tags

`Train/Recon_Loss`, `Train/Pred_Loss`, `Train/Grad_Loss`, `Monitor/Spatial_Std`,
`Monitor/Feature_Std`, `Monitor/Effective_Rank`
