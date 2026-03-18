"""
prepare_ucf101.py -- Convert UCF-101 (.avi/.mp4) to .npy for FluidWorld v2

Download UCF-101:
    https://www.crcv.ucf.edu/data/UCF101.php
    -> UCF101.rar (~6.5 GB, 13320 videos, 101 classes)

Extract:
    # Windows: use 7-Zip or WinRAR
    # Linux:   unrar x UCF101.rar

Usage:
    python scripts/prepare_ucf101.py \
        --source-dir "C:/path/to/UCF-101" \
        --out-dir data/ucf101_64 \
        --size 64 \
        --max-frames 150

Result: ~13K .npy files in data/ucf101_64/
    Each file: (T, 64, 64, 3) uint8

Then, train:
    python experiments/phase1_pixel/train_pixel.py \
        --data-dir data/ucf101_64 \
        --epochs 50 --batch-size 16 --bptt-steps 4 --max-steps 6
"""

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm


def process_video(args):
    video_path, out_dir, size, max_frames, min_frames = args
    base_name = os.path.basename(video_path).split(".")[0]
    out_path = os.path.join(out_dir, f"{base_name}.npy")

    if os.path.exists(out_path):
        return True

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()

    if len(frames) < min_frames:
        return False

    video_array = np.array(frames, dtype=np.uint8)
    np.save(out_path, video_array)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert UCF-101 to .npy for FluidWorld v2")
    parser.add_argument("--source-dir", type=str, required=True,
                        help="UCF-101 folder (contains per-class subfolders)")
    parser.add_argument("--out-dir", type=str, default="data/ucf101_64")
    parser.add_argument("--size", type=int, default=64,
                        help="Target square resolution (default: 64)")
    parser.add_argument("--max-frames", type=int, default=150)
    parser.add_argument("--min-frames", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Searching for videos in {args.source_dir}...")
    video_files = []
    for ext in ("*.avi", "*.mp4", "*.mkv"):
        video_files.extend(
            glob.glob(os.path.join(args.source_dir, "**", ext), recursive=True)
        )
    video_files.sort()
    print(f"{len(video_files)} videos found.")

    if not video_files:
        print("ERROR: No videos found. Check the --source-dir path.")
        print(f"  Expected: {args.source_dir}/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi")
        return

    tasks = [
        (vf, args.out_dir, (args.size, args.size), args.max_frames, args.min_frames)
        for vf in video_files
    ]

    print(f"Converting to .npy {args.size}x{args.size} (multiprocessing)...")
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_video, tasks), total=len(tasks)))

    success = sum(1 for r in results if r)
    failed = len(results) - success
    print(f"\nDone: {success} videos converted, {failed} failed.")
    print(f"Output: {args.out_dir}/")

    # Total size
    total_bytes = sum(
        os.path.getsize(os.path.join(args.out_dir, f))
        for f in os.listdir(args.out_dir) if f.endswith(".npy")
    )
    print(f"Total size: {total_bytes / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
