"""
gradient_plan.py -- Gradient-based planning through the differentiable PDE.

FluidWorld Phase 4: instead of sampling actions (MPC), we optimize
directly via backprop through the BeliefField.

Modes:
  - single: optimize a single action to minimize the cost at t+1
  - trajectory: optimize a sequence of actions over a horizon
"""

import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

_here = Path(__file__).resolve().parent
_project = _here.parent.parent.parent
sys.path.insert(0, str(_project))

from fluidworld.core.world_model import FluidWorldModel


def gradient_planning(args):
    device = torch.device(args.device)
    is_mnist = args.data_dir.endswith(".npy")
    in_channels = 1 if is_mnist else 3

    # Load the model
    model = FluidWorldModel(
        in_channels=in_channels,
        d_model=128,
        stimulus_dim=args.stimulus_dim,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Load a batch of data
    if is_mnist:
        from fluidworld.core.video_dataset import MovingMNISTDataset
        dataset = MovingMNISTDataset(args.data_dir, bptt_steps=args.horizon + 1)
    else:
        from fluidworld.core.video_dataset import PureVideoDataset
        dataset = PureVideoDataset(args.data_dir, bptt_steps=args.horizon + 1)

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, drop_last=True)
    images = next(iter(loader)).to(device)
    B = images.shape[0]

    # Target: the future state encoded by the target encoder
    with torch.no_grad():
        x_0 = images[:, 0]
        z_0 = model.encode(x_0)["features"]
        state = model.belief_field.init_state(B, device)
        state = model.belief_field.write(state, z_0)

        x_target = images[:, args.horizon]
        z_target = model.target_encoder(x_target)["features"].mean(dim=(-2, -1))

    # Actions to optimize (initialized to zero)
    actions = torch.zeros(
        args.horizon, B, args.stimulus_dim,
        device=device, requires_grad=True,
    )
    action_optimizer = torch.optim.Adam([actions], lr=args.lr_action)

    print(f"Gradient planning: horizon={args.horizon}, "
          f"optim_steps={args.optim_steps}")

    for step in range(args.optim_steps):
        action_optimizer.zero_grad()

        # Imaginary rollout
        current_state = state.detach().clone()
        for t in range(args.horizon):
            stim_t = actions[t]
            current_state = model.belief_field.evolve(current_state, stimulus=stim_t)

        z_pred = model.belief_field.read(current_state)
        z_pred = model.predictor(z_pred)

        # Cost: MSE between prediction and target
        cost = F.mse_loss(z_pred, z_target.detach())
        cost.backward()

        grad_norm = actions.grad.norm().item()
        action_optimizer.step()

        if step % 10 == 0 or step == args.optim_steps - 1:
            print(f"  Step {step:3d} | Cost: {cost.item():.4f} | "
                  f"Grad norm: {grad_norm:.6f} | "
                  f"Action norm: {actions.detach().norm().item():.4f}")

    # Baseline: cost with zero actions
    with torch.no_grad():
        current_state = state.detach().clone()
        for t in range(args.horizon):
            stim_t = torch.zeros(B, args.stimulus_dim, device=device)
            current_state = model.belief_field.evolve(current_state, stimulus=stim_t)
        z_pred_zero = model.predictor(model.belief_field.read(current_state))
        cost_zero = F.mse_loss(z_pred_zero, z_target).item()

    print(f"\n--- Results ---")
    print(f"  Cost (zero actions)      : {cost_zero:.4f}")
    print(f"  Cost (optimized actions) : {cost.item():.4f}")
    print(f"  Reduction                : {(1 - cost.item()/cost_zero)*100:.1f}%")
    print(f"  Non-zero gradient        : {'YES' if grad_norm > 1e-8 else 'NO'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--optim-steps", type=int, default=50)
    parser.add_argument("--lr-action", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--stimulus-dim", type=int, default=1)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    gradient_planning(parser.parse_args())
