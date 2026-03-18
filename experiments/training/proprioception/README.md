# Phase 0 -- Proprioceptive World Model

Validates the training pipeline on the simplest case: predicting future proprioceptive
state from current state + action, using a residual MLP.

## Status

Planned. Awaiting SO-101 robot data collection.

## Architecture

- `ProprioWorldModel`: single-step residual MLP (proprio + action -> delta_proprio)
- `MultiStepProprioModel`: autoregressive wrapper for multi-step prediction

## Run

```bash
python experiments/training/proprioception/train_phase0.py --data-path data/proprio_trajectories.npz --epochs 100 --lr 1e-3
```

## Success criteria

- Single-step MSE < 0.01
- Multi-step MSE < 0.05 at horizon h=10

## TensorBoard tags

`Train/Loss`, `Val/MSE_1step`, `Val/MSE_multistep`
