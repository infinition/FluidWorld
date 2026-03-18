# Phase 4 -- Gradient Planning

Optimizes actions via backpropagation through the differentiable PDE dynamics, instead of
sampling-based methods (CEM, MPPI). The PDE forward pass is fully differentiable, allowing
direct gradient descent on action sequences to minimize a user-defined cost function.

## Status

Planned. Requires Phase 3 + Exp D complete.

## Modes

1. **Single-step**: optimize one action to reach a target latent state
2. **Trajectory**: optimize a sequence of actions over horizon h
3. **MPC**: receding-horizon planning with re-planning at each step

## Run

```bash
python experiments/training/gradient_planning/gradient_plan.py --checkpoint checkpoints/phase1_pixel/best.pt --data-dir data/ucf101_64 --horizon 10 --optim-steps 50
```

## Success criteria

- Cost reduction > 30% over random actions
- Gradient norm > 1e-4 (gradients flow through PDE)
- Planning time < 100ms for h=10
- Competitive with CEM baseline

## TensorBoard tags

`Plan/Cost`, `Plan/GradNorm`, `Plan/Time_ms`
