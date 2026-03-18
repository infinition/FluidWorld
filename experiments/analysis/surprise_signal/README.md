# Exp B -- Surprise Signal Analysis

Measures whether the model develops an implicit surprise signal: higher prediction
error at unexpected events (bounces, collisions, scene changes).

## Priority: P0

## Run

```bash
python experiments/analysis/surprise_signal/surprise_analysis.py --checkpoint checkpoints/phase1_pixel/best.pt --data-path data/mnist_moving.npy
```

## Metrics

- Correlation between prediction error and annotated event times
- Surprise ratio at bounces (pred_error_bounce / pred_error_normal > 1.5)
- Mean surprise evolution during training (should decrease)
