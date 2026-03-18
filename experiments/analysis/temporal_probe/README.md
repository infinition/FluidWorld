# Exp A -- Temporal Dynamics Probing

Proves the model encodes temporal dynamics (position, velocity, direction of motion),
not just static appearance. A linear probe is trained to predict object position at
time t+k from features z_t.

## Priority: P0

## Status

Partially done on Moving MNIST (step 32K, pool4 features). Needs adaptation to UCF-101.

## Run

```bash
python experiments/analysis/temporal_probe/temporal_probe.py --checkpoint checkpoints/phase1_pixel/best.pt --data-path data/mnist_moving.npy --feature-mode pool4 --horizon 5
```

## Feature modes

- `pooled`: global average pooling
- `pool4`: 4x4 spatial pooling (recommended)
- `com`: center-of-mass weighted
- `flat`: full flattened features

## Metrics

- R2 score for position prediction at each horizon
- Velocity prediction accuracy
- Comparison: position-only vs position+velocity features
