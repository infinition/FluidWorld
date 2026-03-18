# Phase 5 -- Real Robot Deployment

Full pipeline from data collection to closed-loop control on a SO-101 robot arm
using the LeRobot framework.

## Status

Planned. Requires Phase 4 complete.

## Pipeline

1. Collect demonstrations via LeRobot teleoperation
2. Convert trajectories to `.npz` (images + proprio + actions)
3. Train FluidWorld with vision + proprioception
4. Gradient planning for action selection
5. Closed-loop deployment with receding-horizon MPC

## Integration targets

- FluidVLA features: spatial-aware pooling, action chunking, delta-action normalization
- Action MSE < 5.0
- Cosine similarity > 0.7
- Inference latency < 50ms
- Pick success rate > 50%

## Dependencies

- [LeRobot](https://github.com/huggingface/lerobot)
- SO-101 robot arm + camera setup
