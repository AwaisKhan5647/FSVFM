"""
Dataset utilities for experiment_image_kutub.

  scan_image_dir(root)  — recursive (path, label) collector using 0_real/1_fake path logic
  KutubDataset          — PIL image dataset with optional transform
  BalancedSampler       — epoch-level balanced real/fake sampler for single-GPU training
"""

import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, Sampler

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def label_from_path(path: str) -> int:
    """
    Infer label from any component in the path tree.
    Returns 0 (real), 1 (fake), or -1 (unknown).
    Handles: 0_real/1_fake dirs at any depth, and common alias names.
    """
    parts = Path(path).parts
    for part in parts:
        p = part.lower()
        if p == "0_real" or p in {"real", "original", "youtube", "actors", "raw"}:
            return 0
        if p == "1_fake" or p in {"fake", "manipulated", "tampered", "forgery"}:
            return 1
    return -1


def scan_image_dir(root: str, skip_unknown: bool = True) -> List[Tuple[str, int]]:
    """
    Recursively scan `root` and return list of (abs_path, label).
    Files with label == -1 are skipped when skip_unknown=True.
    """
    root_path = Path(root)
    samples: List[Tuple[str, int]] = []
    for p in root_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            lbl = label_from_path(str(p))
            if skip_unknown and lbl < 0:
                continue
            samples.append((str(p), lbl))
    return samples


class KutubDataset(Dataset):
    """Image dataset from a list of (path, label) pairs."""

    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
        except Exception:
            img = torch.zeros(3, 224, 224)
        return img, label, path


class BalancedSampler(Sampler):
    """
    Epoch-level balanced sampler for single-GPU training.

    Each epoch draws min(n_real, n_fake) from each class, interleaves pairs,
    then shuffles the full sequence.  Call set_epoch(e) before each epoch.
    """

    def __init__(self, samples: List[Tuple[str, int]], seed: int = 0):
        self.real_idx = [i for i, (_, l) in enumerate(samples) if l == 0]
        self.fake_idx = [i for i, (_, l) in enumerate(samples) if l == 1]
        self.seed     = seed
        self._epoch   = 0
        if not self.real_idx:
            raise ValueError("BalancedSampler: no real samples found")
        if not self.fake_idx:
            raise ValueError("BalancedSampler: no fake samples found")

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return 2 * min(len(self.real_idx), len(self.fake_idx))

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        n   = min(len(self.real_idx), len(self.fake_idx))
        real_sample = rng.sample(self.real_idx, n)
        fake_sample = rng.sample(self.fake_idx, n)
        merged = []
        for r, f in zip(real_sample, fake_sample):
            merged.append(r)
            merged.append(f)
        rng.shuffle(merged)
        return iter(merged)
