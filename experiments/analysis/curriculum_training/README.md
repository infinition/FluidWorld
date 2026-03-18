# Exp F -- Curriculum Training

Piaget-inspired curriculum with 5 levels of increasing visual complexity.
Tests whether structured training order improves final performance or accelerates
convergence compared to direct training on the full dataset.

## Priority: P2

## Status

Planned. Needs adaptation to FluidWorldModelV2.

## Curriculum levels

1. Single digit, linear motion
2. Single digit with bounces
3. Two digits with collisions
4. Three digits, complex dynamics
5. Full Moving MNIST

Promotion criterion: surprise ratio < 1.2 (model has "mastered" current level).

## Run

Generate curriculum data:

```bash
python experiments/analysis/curriculum_training/generate_curriculum_data.py --out-dir data/curriculum/
```

Train with curriculum:

```bash
python experiments/analysis/curriculum_training/train_curriculum.py --data-dir data/curriculum/ --epochs 100 --batch-size 16
```

## Targets

- Better final performance or 2x faster convergence vs direct training baseline
