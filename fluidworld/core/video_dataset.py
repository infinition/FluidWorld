"""
video_dataset.py -- Pure video data loaders: sliding windows from video sequences (no actions/labels).
"""

import glob
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class PureVideoDataset(Dataset):
    """
    Loads .npy video files from a directory. Expected shape: (T, H, W, C).
    Optimized for large datasets via cached index file.
    """
    def __init__(
        self,
        data_dir: str,
        bptt_steps: int = 8,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.bptt_steps = bptt_steps

        # Find all .npy video files (exclude _index)
        self.video_files = sorted([
            f for f in glob.glob(str(self.data_dir / "*.npy"))
            if not Path(f).name.startswith("_")
        ])
        if not self.video_files:
            raise FileNotFoundError(f"No .npy video files found in {data_dir}")

        # Try loading cached index (avoids scanning 13K files)
        index_path = self.data_dir / "_index.npz"
        if index_path.exists():
            cached = np.load(str(index_path))
            if (int(cached["n_files"]) == len(self.video_files)
                    and int(cached["bptt"]) == bptt_steps):
                self.video_lengths = cached["lengths"].tolist()
                # samples stored as (N, 2) int array
                samples_arr = cached["samples"]
                self.samples = [(int(r[0]), int(r[1])) for r in samples_arr]
                print(f"PureVideoDataset: {len(self.samples)} windows (cached index).")
                return

        # Scan: read shapes via mmap (header only)
        self.video_lengths = []
        self.samples = []

        print(f"Indexing {len(self.video_files)} videos...")
        for vid_idx, file_path in enumerate(self.video_files):
            video_data = np.load(file_path, mmap_mode='r')
            ep_length = video_data.shape[0]
            self.video_lengths.append(ep_length)

            max_start_idx = ep_length - (self.bptt_steps + 1)
            for start_idx in range(max(0, max_start_idx + 1)):
                self.samples.append((vid_idx, start_idx))

        # Save index as .npz
        np.savez(
            str(index_path),
            n_files=np.array(len(self.video_files)),
            bptt=np.array(bptt_steps),
            lengths=np.array(self.video_lengths, dtype=np.int32),
            samples=np.array(self.samples, dtype=np.int32),
        )
        print(f"PureVideoDataset: {len(self.samples)} temporal windows (index saved).")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        vid_idx, start_idx = self.samples[idx]
        file_path = self.video_files[vid_idx]

        # Load image sequence
        video_data = np.load(file_path, mmap_mode='r')
        img_seq = video_data[start_idx : start_idx + self.bptt_steps + 1]
        img_seq = np.array(img_seq)

        # Format (T, H, W, C) -> (T, C, H, W) and normalize to [0, 1]
        if img_seq.ndim == 4 and img_seq.shape[-1] in [1, 3]: 
            img_seq = img_seq.transpose(0, 3, 1, 2)
            
        img_tensor = torch.from_numpy(img_seq).float()
        if img_tensor.max() > 1.0:
            img_tensor = img_tensor / 255.0

        return img_tensor


class MovingMNISTDataset(Dataset):
    """Loader for raw 'mnist_test_seq.npy' (~800 MB). Handles dimension transposition and single-channel format."""
    def __init__(self, npy_file: str, bptt_steps: int = 8):
        super().__init__()
        self.bptt_steps = bptt_steps
        
        print(f"Loading raw Moving MNIST dataset into RAM: {npy_file}...")
        data = np.load(npy_file)
        
        # Official format is (20, 10000, 64, 64) = (T, B, H, W)
        # Transpose to (10000, 20, 64, 64) = (B, T, H, W)
        if data.shape[0] == 20 and data.shape[1] == 10000:
            data = data.transpose(1, 0, 2, 3)
            
        self.data = data
        self.num_sequences = self.data.shape[0]
        self.total_time = self.data.shape[1]
        
        # Create sliding windows
        self.samples = []
        max_start = self.total_time - (self.bptt_steps + 1)
        
        for seq_idx in range(self.num_sequences):
            for start_idx in range(max_start + 1):
                self.samples.append((seq_idx, start_idx))
                
        print(f"Moving MNIST ready: {len(self.samples)} windows of {bptt_steps+1} frames extracted.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        seq_idx, start_idx = self.samples[idx]
        
        # Extract sequence: (T, H, W)
        img_seq = self.data[seq_idx, start_idx : start_idx + self.bptt_steps + 1]
        
        # Add channel dim C=1 for grayscale: (T, H, W) -> (T, 1, H, W)
        img_seq = np.expand_dims(img_seq, axis=1) # -> (T, 1, H, W)
        
        img_tensor = torch.from_numpy(img_seq).float()
        
        # Normalize to [0, 1]
        if img_tensor.max() > 1.0:
            img_tensor = img_tensor / 255.0
            
        return img_tensor