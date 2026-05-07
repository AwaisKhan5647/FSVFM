# -*- coding: utf-8 -*-
# Composite dataset loader for FSVFM robust training
# Supports: multi-root datasets, mixed structures, corrupted-file skipping,
#           class balancing, dataset statistics logging.

import os
import glob
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError
import torch
from torch.utils.data import Dataset, ConcatDataset, WeightedRandomSampler

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

# Folder names whose content is treated as real (label = 0)
REAL_KEYWORDS = {"real", "original", "youtube", "actors", "raw"}
# Everything else → fake (label = 1); you can also specify via label files


def _label_from_folder(folder_name: str) -> int:
    """Infer binary label (0=real, 1=fake) from a folder name."""
    return 0 if any(kw in folder_name.lower() for kw in REAL_KEYWORDS) else 1


def _safe_open(path: str) -> Optional[Image.Image]:
    try:
        img = Image.open(path).convert("RGB")
        img.verify()          # detect truncated headers
        img = Image.open(path).convert("RGB")  # re-open after verify
        return img
    except (UnidentifiedImageError, OSError, Exception):
        return None


# ---------------------------------------------------------------------------
# Single-root dataset  (ImageFolder-like, arbitrary depth)
# ---------------------------------------------------------------------------

class SingleRootDataset(Dataset):
    """
    Recursively scans `root` for image files.

    Label logic:
      - If `label_file` is given: reads "path label\\n" pairs (0=real, 1=fake).
      - Otherwise: inspects the *first-level subdirectory* name under root
        (real/fake keywords → label 0/1).

    Supported image extensions: jpg, jpeg, png, bmp, webp.
    Corrupted files are skipped silently.
    """

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        label_file: Optional[str] = None,
        delimiter: str = " ",
        skip_corrupted: bool = True,
        max_samples: Optional[int] = None,
        dataset_name: Optional[str] = None,
    ):
        self.root = root
        self.transform = transform
        self.skip_corrupted = skip_corrupted
        self.dataset_name = dataset_name or Path(root).name

        self.samples: List[Tuple[str, int]] = []

        if label_file is not None:
            self._load_from_label_file(label_file, delimiter)
        else:
            self._scan_root()

        if max_samples and len(self.samples) > max_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

        self._log_stats()

    def _load_from_label_file(self, label_file: str, delimiter: str):
        with open(label_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(delimiter)
                if len(parts) < 2:
                    continue
                rel_path, label = parts[0], int(parts[1])
                abs_path = rel_path if os.path.isabs(rel_path) else os.path.join(self.root, rel_path)
                if os.path.isfile(abs_path):
                    self.samples.append((abs_path, label))

    def _scan_root(self):
        root = Path(self.root)
        if not root.is_dir():
            logger.warning(f"[{self.dataset_name}] Root not found: {self.root}")
            return

        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in self.IMG_EXTS:
                continue
            # Determine label from the first-level subfolder under root
            try:
                rel = path.relative_to(root)
                first_folder = rel.parts[0] if len(rel.parts) > 1 else path.parent.name
            except ValueError:
                first_folder = path.parent.name
            label = _label_from_folder(first_folder)
            self.samples.append((str(path), label))

    def _log_stats(self):
        counts = defaultdict(int)
        for _, lbl in self.samples:
            counts[lbl] += 1
        logger.info(
            f"[{self.dataset_name}] Loaded {len(self.samples)} samples — "
            f"real={counts[0]}, fake={counts[1]}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = _safe_open(path)

        if img is None:
            if self.skip_corrupted:
                # Return a random valid sample instead
                alt_idx = random.randint(0, len(self.samples) - 1)
                return self.__getitem__(alt_idx)
            raise RuntimeError(f"Cannot open image: {path}")

        if self.transform is not None:
            img = self.transform(img)

        return img, label


# ---------------------------------------------------------------------------
# Composite dataset  (wraps multiple SingleRootDatasets)
# ---------------------------------------------------------------------------

class CompositeDeepfakeDataset(Dataset):
    """
    Merges multiple SingleRootDatasets into one flat dataset.

    Args:
        dataset_roots:  list of root directories, or list of (root, label_file) tuples.
        transform:      torchvision transform applied to every sample.
        balance:        if True, oversample the minority class to 1:1 ratio.
        max_per_dataset: cap samples per sub-dataset (None = no cap).
    """

    def __init__(
        self,
        dataset_roots: List[Union[str, Tuple[str, str]]],
        transform: Optional[Callable] = None,
        balance: bool = False,
        max_per_dataset: Optional[int] = None,
        skip_corrupted: bool = True,
    ):
        self.transform = transform
        self.all_samples: List[Tuple[str, int]] = []
        self.dataset_sizes: dict = {}

        for entry in dataset_roots:
            if isinstance(entry, (list, tuple)) and len(entry) == 2 and isinstance(entry[1], str) and entry[1].endswith(".txt"):
                root, label_file = entry
            else:
                root, label_file = (entry, None)

            ds = SingleRootDataset(
                root=str(root),
                transform=None,   # transform applied in __getitem__ of this class
                label_file=label_file,
                skip_corrupted=skip_corrupted,
                max_samples=max_per_dataset,
                dataset_name=Path(str(root)).name,
            )
            self.dataset_sizes[Path(str(root)).name] = {
                "total": len(ds),
                "real": sum(1 for _, l in ds.samples if l == 0),
                "fake": sum(1 for _, l in ds.samples if l == 1),
            }
            self.all_samples.extend(ds.samples)

        if balance:
            self.all_samples = self._balance(self.all_samples)

        self._print_summary()

    def _balance(self, samples: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        real = [s for s in samples if s[1] == 0]
        fake = [s for s in samples if s[1] == 1]
        min_count = min(len(real), len(fake))
        random.shuffle(real)
        random.shuffle(fake)
        balanced = real[:min_count] + fake[:min_count]
        random.shuffle(balanced)
        logger.info(f"[Composite] After balancing: {min_count} real + {min_count} fake = {len(balanced)} total")
        return balanced

    def _print_summary(self):
        total_real = sum(1 for _, l in self.all_samples if l == 0)
        total_fake = sum(1 for _, l in self.all_samples if l == 1)
        print("\n" + "=" * 60)
        print("COMPOSITE DATASET SUMMARY")
        print("=" * 60)
        for name, stats in self.dataset_sizes.items():
            print(f"  {name:40s}  total={stats['total']:7d}  real={stats['real']:7d}  fake={stats['fake']:7d}")
        print("-" * 60)
        print(f"  {'TOTAL':40s}  total={len(self.all_samples):7d}  real={total_real:7d}  fake={total_fake:7d}")
        print("=" * 60 + "\n")

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """Returns a WeightedRandomSampler for balanced batch sampling without discarding data."""
        labels = np.array([s[1] for s in self.all_samples])
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]
        return WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).float(),
            num_samples=len(self.all_samples),
            replacement=True,
        )

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.all_samples[idx]
        img = _safe_open(path)

        if img is None:
            alt_idx = random.randint(0, len(self.all_samples) - 1)
            return self.__getitem__(alt_idx)

        if self.transform is not None:
            img = self.transform(img)

        return img, label


# ---------------------------------------------------------------------------
# Helper: build composite dataset from a config dict or list of roots
# ---------------------------------------------------------------------------

def build_composite_dataset(
    roots: List[Union[str, Tuple[str, str]]],
    is_train: bool,
    args,
    balance: bool = False,
    max_per_dataset: Optional[int] = None,
) -> CompositeDeepfakeDataset:
    """
    Build a CompositeDeepfakeDataset using args-derived transforms.
    `roots` is a list of directory paths or (root, label_file) tuples.
    """
    from util.datasets import build_transform  # reuse existing transform builder
    transform = build_transform(is_train, args)

    dataset = CompositeDeepfakeDataset(
        dataset_roots=roots,
        transform=transform,
        balance=balance,
        max_per_dataset=max_per_dataset,
        skip_corrupted=True,
    )
    return dataset
