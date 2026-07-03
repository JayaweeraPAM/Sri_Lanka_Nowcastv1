"""
PyTorch Dataset: turns preprocessed .npy grids + dataset_info.json into
(input_sequence, target_sequence) tensor pairs. Splits by contiguous date
blocks (not random) to avoid train/val leakage.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config


class CTHSequenceDataset(Dataset):
    def __init__(self, cfg: dict, split: str = "train", val_fraction: float = 0.15):
        self.processed_dir = cfg["paths"]["processed_dir"]
        self.data_dir = self.processed_dir / "data"
        self.input_frames = cfg["sequence"]["input_frames"]
        self.target_frames = cfg["sequence"]["target_frames"]
        window = self.input_frames + self.target_frames

        with open(self.processed_dir / "dataset_info.json") as f:
            info = json.load(f)

        filenames = [t["filename"] for t in info["timestamps"]]
        split_idx = int(len(filenames) * (1 - val_fraction))
        if split == "train":
            filenames = filenames[:split_idx]
        elif split == "val":
            filenames = filenames[split_idx:]
        else:
            raise ValueError("split must be 'train' or 'val'")

        self.windows = [
            filenames[i : i + window]
            for i in range(len(filenames) - window + 1)
        ]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window_files = self.windows[idx]
        frames = [np.load(self.data_dir / fn) for fn in window_files]
        frames = np.stack(frames, axis=0)
        frames = frames[:, None, :, :]
        x = frames[: self.input_frames]
        y = frames[self.input_frames :]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


if __name__ == "__main__":
    cfg = load_config()
    train_ds = CTHSequenceDataset(cfg, split="train")
    val_ds = CTHSequenceDataset(cfg, split="val")
    print(f"Train sequences: {len(train_ds)}")
    print(f"Val sequences:   {len(val_ds)}")
    if len(train_ds) > 0:
        x, y = train_ds[0]
        print(f"Sample input shape:  {x.shape}")
        print(f"Sample target shape: {y.shape}")
