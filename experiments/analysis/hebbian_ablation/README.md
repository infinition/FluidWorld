# Exp E -- Hebbian Plasticity Ablation

Evaluates the impact of the HebbianDiffusion mechanism in FluidWorldModelV2. This
module modifies PDE diffusion coefficients based on past activation patterns,
implementing a form of local synaptic plasticity.

## Priority: P2

## Status

Planned. Needs adaptation to FluidWorldModelV2.

## Run

Train with Hebbian enabled (default):

```bash
python experiments/analysis/hebbian_ablation/train_hebbian.py --data-dir data/ucf101_64 --epochs 50 --batch-size 16
```

## Ablation

Compare metrics (Exp A temporal probe, Exp B surprise, Exp D multiscale rollout)
between models trained with and without Hebbian plasticity.

## Files

- `hebbian_belief_field.py`: standalone Hebbian belief field module
- `train_hebbian.py`: training script with Hebbian-specific logging
