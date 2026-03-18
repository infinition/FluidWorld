# Exp D -- Multiscale Rollout Stability

Evaluates imagination stability across rollout horizons (1, 2, 5, 10, 15 steps).
Measures pixel-space reconstruction quality to characterize error accumulation rate.

## Priority: P1

## Status

Planned. Needs adaptation to FluidWorldModelV2.

## Run

```bash
python experiments/analysis/multiscale_rollout/multiscale_rollout.py --checkpoint checkpoints/phase1_pixel/best.pt --data-path data/mnist_moving.npy --horizons 1 2 5 10 15
```

## Metrics

- Pixel MSE at each horizon
- SSIM at each horizon
- Growth rate (should be sub-exponential, target < 20)
