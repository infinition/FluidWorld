# Exp C -- Causal Perturbation Analysis

Tests causal relationships in latent space. Applies localized perturbations to features
in one object's spatial region and measures whether prediction changes remain spatially
localized and semantically coherent.

## Priority: P1

## Status

Planned. Needs adaptation to FluidWorldModelV2.

## Run

```bash
python experiments/analysis/perturbation_analysis/perturbation_probe.py --checkpoint checkpoints/phase1_pixel/best.pt --data-path data/mnist_moving.npy
```

## Metrics

- Locality score (perturbed vs unperturbed region response ratio > 2.0)
- Coherence (cosine similarity of perturbed predictions > 0.5)
