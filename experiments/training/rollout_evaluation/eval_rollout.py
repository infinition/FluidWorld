import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_here = Path(__file__).resolve().parent
_project = _here.parent.parent.parent
sys.path.insert(0, str(_project))

from fluidworld.core.world_model import FluidWorldModel


@torch.no_grad()
def eval_rollout(args):
    device = torch.device(args.device)
    is_mnist = args.data_dir.endswith(".npy")
    in_channels = 1 if is_mnist else 3

    world_model = FluidWorldModel(
        in_channels=in_channels,
        d_model=128,
        stimulus_dim=args.stimulus_dim,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    else:
        state_dict = ckpt
    world_model.load_state_dict(state_dict, strict=False)
    world_model.eval()

    if is_mnist:
        from fluidworld.core.video_dataset import MovingMNISTDataset

        dataset = MovingMNISTDataset(args.data_dir, bptt_steps=args.horizon)
    else:
        from fluidworld.core.video_dataset import PureVideoDataset

        dataset = PureVideoDataset(args.data_dir, bptt_steps=args.horizon)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    images = next(iter(loader)).to(device, non_blocking=True)
    # In phase 3, we test "blind" imagination: no exogenous stimulus.
    stimulus = torch.zeros(
        images.shape[0], args.horizon, args.stimulus_dim, device=device
    )
    B = images.shape[0]

    x_0 = images[:, 0]
    z_0 = world_model.encode(x_0)["features"]
    state = world_model.belief_field.init_state(B, device)
    state = world_model.belief_field.write(state, z_0)

    mse_per_step = []
    for t in range(args.horizon):
        stim_t = stimulus[:, t]
        state = world_model.belief_field.evolve(state, stimulus=stim_t)

        z_pred_pooled = world_model.belief_field.read(state)
        z_pred_projected = world_model.predictor(z_pred_pooled)

        x_target = images[:, t + 1]
        z_target_features = world_model.target_encoder(x_target)["features"]
        z_target = z_target_features.mean(dim=(-2, -1))

        mse = F.mse_loss(z_pred_projected, z_target).item()
        mse_per_step.append(mse)
        print(f"Step T+{t+1} | MSE: {mse:.4f}")

    print(
        "\nRollout test complete. Error trend:",
        ["{:.4f}".format(m) for m in mse_per_step],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--stimulus-dim", type=int, default=6)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    eval_rollout(parser.parse_args())
