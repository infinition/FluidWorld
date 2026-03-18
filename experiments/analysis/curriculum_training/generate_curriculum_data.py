"""
generate_curriculum_data.py -- Exp F : Curriculum dataset generator

Generates complexity-controlled Moving MNIST datasets:
  Level 0: 1 digit, linear trajectory, no bouncing
  Level 1: 1 digit, wall bouncing
  Level 2: 2 digits, possible collisions
  Level 3: 3 digits, crossing dynamics
"""

import argparse
from pathlib import Path

import numpy as np


def generate_moving_digits(
    n_sequences: int,
    n_frames: int,
    img_size: int,
    n_digits: int,
    bounce: bool,
    digit_size: int = 28,
    speed_range: tuple = (2, 5),
    seed: int = 42,
):
    """
    Generate sequences of moving digits.

    Returns:
        data: (n_sequences, n_frames, img_size, img_size) uint8
    """
    rng = np.random.RandomState(seed)

    # Load MNIST digits (generate blobs as fallback if unavailable)
    try:
        from torchvision import datasets
        mnist = datasets.MNIST(root="/tmp/mnist", train=True, download=True)
        digit_images = mnist.data.numpy()  # (60000, 28, 28)
    except Exception:
        # Fallback: generate Gaussian blobs
        print("  MNIST unavailable, using Gaussian blobs")
        digit_images = np.zeros((1000, 28, 28), dtype=np.uint8)
        for i in range(1000):
            cx, cy = rng.randint(8, 20, size=2)
            yy, xx = np.mgrid[:28, :28]
            blob = np.exp(-((yy - cy)**2 + (xx - cx)**2) / 20.0)
            digit_images[i] = (blob * 255).astype(np.uint8)

    data = np.zeros((n_sequences, n_frames, img_size, img_size), dtype=np.uint8)

    for seq in range(n_sequences):
        # Initialiser les digits
        positions = []
        velocities = []
        sprites = []

        for d in range(n_digits):
            idx = rng.randint(len(digit_images))
            sprite = digit_images[idx].astype(np.float32) / 255.0

            # Resize if needed
            if digit_size != 28:
                from PIL import Image
                sprite_pil = Image.fromarray((sprite * 255).astype(np.uint8))
                sprite_pil = sprite_pil.resize((digit_size, digit_size))
                sprite = np.array(sprite_pil).astype(np.float32) / 255.0

            max_pos = img_size - digit_size
            x = rng.randint(0, max(max_pos, 1))
            y = rng.randint(0, max(max_pos, 1))
            speed = rng.uniform(*speed_range)
            angle = rng.uniform(0, 2 * np.pi)
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)

            positions.append([float(x), float(y)])
            velocities.append([vx, vy])
            sprites.append(sprite)

        positions = np.array(positions)
        velocities = np.array(velocities)

        for t in range(n_frames):
            frame = np.zeros((img_size, img_size), dtype=np.float32)

            for d in range(n_digits):
                x, y = positions[d]
                ix, iy = int(round(x)), int(round(y))

                # Placer le sprite
                x1 = max(0, ix)
                y1 = max(0, iy)
                x2 = min(img_size, ix + digit_size)
                y2 = min(img_size, iy + digit_size)
                sx1 = x1 - ix
                sy1 = y1 - iy
                sx2 = sx1 + (x2 - x1)
                sy2 = sy1 + (y2 - y1)

                if x2 > x1 and y2 > y1 and sx2 <= digit_size and sy2 <= digit_size:
                    frame[y1:y2, x1:x2] = np.maximum(
                        frame[y1:y2, x1:x2],
                        sprites[d][sy1:sy2, sx1:sx2],
                    )

                # Update position
                positions[d][0] += velocities[d][0]
                positions[d][1] += velocities[d][1]

                # Bouncing
                max_pos = img_size - digit_size
                if bounce:
                    if positions[d][0] < 0:
                        positions[d][0] = -positions[d][0]
                        velocities[d][0] = -velocities[d][0]
                    if positions[d][0] > max_pos:
                        positions[d][0] = 2 * max_pos - positions[d][0]
                        velocities[d][0] = -velocities[d][0]
                    if positions[d][1] < 0:
                        positions[d][1] = -positions[d][1]
                        velocities[d][1] = -velocities[d][1]
                    if positions[d][1] > max_pos:
                        positions[d][1] = 2 * max_pos - positions[d][1]
                        velocities[d][1] = -velocities[d][1]
                else:
                    # No bouncing: wrap around (digit reappears on the other side)
                    positions[d][0] = positions[d][0] % max(max_pos, 1)
                    positions[d][1] = positions[d][1] % max(max_pos, 1)

            data[seq, t] = (frame * 255).clip(0, 255).astype(np.uint8)

    return data


def main():
    parser = argparse.ArgumentParser(description="Generate curriculum datasets")
    parser.add_argument("--output-dir", type=str, default="data/curriculum")
    parser.add_argument("--n-sequences", type=int, default=5000)
    parser.add_argument("--n-frames", type=int, default=20)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    levels = [
        {"name": "level_0_permanence", "n_digits": 1, "bounce": False,
         "speed": (2, 4), "desc": "1 digit, linear, no bouncing"},
        {"name": "level_1_causality", "n_digits": 1, "bounce": True,
         "speed": (2, 5), "desc": "1 digit, bouncing"},
        {"name": "level_2_interaction", "n_digits": 2, "bounce": True,
         "speed": (2, 5), "desc": "2 digits, bouncing"},
        {"name": "level_3_multiobject", "n_digits": 3, "bounce": True,
         "speed": (1, 4), "desc": "3 digits, bouncing"},
    ]

    for i, lvl in enumerate(levels):
        print(f"\nLevel {i}: {lvl['desc']}")
        data = generate_moving_digits(
            n_sequences=args.n_sequences,
            n_frames=args.n_frames,
            img_size=args.img_size,
            n_digits=lvl["n_digits"],
            bounce=lvl["bounce"],
            speed_range=lvl["speed"],
            seed=args.seed + i,
        )
        path = out / f"{lvl['name']}.npy"
        np.save(path, data)
        print(f"  Saved: {path} -- shape {data.shape}")

    print(f"\nCurriculum datasets generated in {out}")
    print("Format: (N_seq, T, H, W) with values [0, 255]")


if __name__ == "__main__":
    main()
