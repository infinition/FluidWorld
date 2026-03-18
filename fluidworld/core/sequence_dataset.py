"""
sequence_dataset.py -- Temporal data loader: sliding windows from .npz episodes for TBPTT.
"""

import glob
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class FluidSequenceDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        bptt_steps: int = 8,
        image_key: str = "images",    # Key in .npz for video frames
        stimulus_key: str = "actions" # Key in .npz for control signals
    ):
        """
        Args:
            data_dir: Directory containing `episode_*.npz` files.
            bptt_steps: Imagination sequence length.
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.bptt_steps = bptt_steps
        self.image_key = image_key
        self.stimulus_key = stimulus_key
        
        # Find all episodes
        self.episode_files = sorted(glob.glob(str(self.data_dir / "episode_*.npz")))
        if not self.episode_files:
            raise FileNotFoundError(f"No 'episode_*.npz' found in {data_dir}")

        # Build global index mapping idx -> (file, frame_start)
        self.samples = []
        print(f"Indexing episodes in {data_dir}...")
        
        for file_path in self.episode_files:
            # Load only metadata (shape) without loading into RAM
            with np.load(file_path, mmap_mode='r') as data:
                if image_key not in data or stimulus_key not in data:
                    print(f"Skipped: {file_path} (missing keys)")
                    continue
                    
                ep_length = data[image_key].shape[0]
                
                # Need bptt_steps + 1 frames per window
                max_start_idx = ep_length - (self.bptt_steps + 1)
                
                for start_idx in range(max_start_idx + 1):
                    self.samples.append((file_path, start_idx))
                    
        print(f"Dataset ready: {len(self.samples)} temporal sequences extracted.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path, start_idx = self.samples[idx]
        
        # Lazy load the specific sequence
        with np.load(file_path, mmap_mode='r') as data:
            # Extract temporal sequences
            # Images: need T + 1 frames
            img_seq = data[self.image_key][start_idx : start_idx + self.bptt_steps + 1]
            # Stimulus: need T frames
            stim_seq = data[self.stimulus_key][start_idx : start_idx + self.bptt_steps]
            
            # Convert to standard numpy array (leave mmap)
            img_seq = np.array(img_seq)
            stim_seq = np.array(stim_seq)

        # ── Image formatting ──
        # Assume (T, H, W, C) uint8 [0, 255] -> (T, C, H, W) float32 [0, 1]
        if img_seq.ndim == 4 and img_seq.shape[-1] in [1, 3, 4]: 
            img_seq = img_seq.transpose(0, 3, 1, 2)
            
        img_tensor = torch.from_numpy(img_seq).float()
        if img_tensor.max() > 1.0:
            img_tensor = img_tensor / 255.0

        # ── Stimulus formatting ──
        # (T, D) float32
        stim_tensor = torch.from_numpy(stim_seq).float()

        return img_tensor, stim_tensor