# Phase 3 -- Multi-Step Rollout Evaluation

Tests PDE stability in pure autoregression (no new observations). Initializes the
BeliefField from frame x_0, then evolves the latent state forward without visual input.
Compares imagined vs real embeddings at each horizon step.

## Status

Planned. Requires Phase 1 + Exp A/B complete.

## Prerequisites

- Trained FluidWorld checkpoint
- Test video sequences

## Run

```bash
python experiments/training/rollout_evaluation/eval_rollout.py --checkpoint checkpoints/phase1_pixel/best.pt --data-dir data/ucf101_64 --horizons 1 2 5 10 20
```

## Metrics

- MSE between predicted and actual latent states at each horizon
- Cosine similarity of feature maps
- Effective rank evolution across horizons
- Growth rate (should be sub-exponential)

## Success criteria

- MSE(h=5) < 3.0
- MSE(h=20) < 10.0
- Growth rate < 20
- Cosine similarity at h=10 > 0.3

## TensorBoard tags

`Rollout/MSE_h{N}`, `Rollout/CosineSim_h{N}`, `Rollout/EffRank_h{N}`
