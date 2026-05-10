# -*- coding: utf-8 -*-
# Composite dataset loader for FSVFM robust training
# Supports: multi-root datasets, mixed structures, corrupted-file skipping,
#           class balancing, dataset statistics logging, balanced DDP sampling.

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
import torch.distributed as dist
from torch.utils.data import Dataset, ConcatDataset, WeightedRandomSampler, Sampler

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


# ---------------------------------------------------------------------------
# Build dataset from separate real / fake path-list txt files
# (one absolute path per line, no label column needed — label comes from file)
# ---------------------------------------------------------------------------

def build_from_txt_pair(
    real_txt: str,
    fake_txt: str,
    is_train: bool,
    args,
    balance: bool = False,
    max_per_class: Optional[int] = None,
    path_remap_from: str = "",
    path_remap_to: str = "",
) -> "CompositeDeepfakeDataset":
    """Build a dataset from two separate path-list files: one for real, one for fake.

    path_remap_from/to: replace a path prefix in every line of the txt files,
    e.g. from='/data/saad/datasets/gend_unified' to='/mnt/h200_dataset/gend_unified'.
    """
    from util.datasets import build_transform
    transform = build_transform(is_train, args)

    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    do_remap = bool(path_remap_from and path_remap_to)
    samples: List[Tuple[str, int]] = []
    skipped = 0

    for txt_path, label in [(real_txt, 0), (fake_txt, 1)]:
        tag = "real" if label == 0 else "fake"
        with open(txt_path, "r") as fh:
            raw_lines = [ln.strip() for ln in fh if ln.strip()]

        if do_remap:
            raw_lines = [p.replace(path_remap_from, path_remap_to, 1) for p in raw_lines]
        if max_per_class is not None:
            random.shuffle(raw_lines)
            raw_lines = raw_lines[:max_per_class]

        # Spot-check 5 paths to confirm the mount is accessible before bulk-loading
        spot = [raw_lines[i] for i in range(0, min(len(raw_lines), 5000), 1000)]
        bad_spot = sum(1 for p in spot if not os.path.isfile(p))
        if bad_spot == len(spot) and len(spot) > 0:
            raise RuntimeError(
                f"[build_from_txt_pair] Spot-check FAILED for '{txt_path}': "
                f"none of {len(spot)} sampled paths exist. "
                f"Check mount and --txt_path_remap_to. Sample: {spot[0]}"
            )
        if bad_spot > 0:
            logger.warning(f"[build_from_txt_pair] {tag}: {bad_spot}/{len(spot)} spot-checked paths missing.")

        # Load all paths directly — no per-file stat (too slow over network mounts).
        # Missing files are handled gracefully in __getitem__ via _safe_open.
        it = (
            _tqdm(raw_lines, desc=f"  Indexing {tag} ({len(raw_lines):,})", unit="path",
                  dynamic_ncols=True, leave=True,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]")
            if _tqdm is not None else raw_lines
        )
        for p in it:
            samples.append((p, label))

    logger.info(f"[build_from_txt_pair] Indexed {len(samples):,} paths total "
                f"({sum(1 for _,l in samples if l==0):,} real  "
                f"{sum(1 for _,l in samples if l==1):,} fake)")

    # Re-use CompositeDeepfakeDataset but inject samples directly
    ds = CompositeDeepfakeDataset.__new__(CompositeDeepfakeDataset)
    ds.transform = transform
    ds.all_samples = samples
    ds.dataset_sizes = {
        Path(real_txt).stem: {
            "total": sum(1 for _, l in samples if l == 0),
            "real": sum(1 for _, l in samples if l == 0),
            "fake": 0,
        },
        Path(fake_txt).stem: {
            "total": sum(1 for _, l in samples if l == 1),
            "real": 0,
            "fake": sum(1 for _, l in samples if l == 1),
        },
    }

    if balance:
        ds.all_samples = ds._balance(ds.all_samples)

    ds._print_summary()
    return ds


# ---------------------------------------------------------------------------
# DistributedBalancedSampler — strict 50/50 real/fake per batch, DDP-safe
# ---------------------------------------------------------------------------

class DistributedBalancedSampler(Sampler):
    """
    Emits indices in alternating [half_batch real, half_batch fake] blocks so
    that every DataLoader batch (batch_size = 2 * half_batch) is exactly 50 %
    real and 50 % fake.  DDP-safe: each rank receives a non-overlapping,
    balanced slice of the dataset.

    Usage:
        sampler = DistributedBalancedSampler(
            dataset,
            batch_size_per_gpu=64,
            num_replicas=world_size,
            rank=global_rank,
        )
        loader = DataLoader(dataset, batch_size=64, sampler=sampler, ...)
        # Each batch on every GPU: 32 real + 32 fake frames.
    """

    def __init__(
        self,
        dataset: "CompositeDeepfakeDataset",
        batch_size_per_gpu: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
    ):
        if batch_size_per_gpu % 2 != 0:
            raise ValueError("batch_size_per_gpu must be even for balanced batching.")

        if num_replicas is None:
            num_replicas = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
        if rank is None:
            rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0

        self.real_indices = [i for i, (_, l) in enumerate(dataset.all_samples) if l == 0]
        self.fake_indices = [i for i, (_, l) in enumerate(dataset.all_samples) if l == 1]

        if not self.real_indices or not self.fake_indices:
            raise ValueError("Dataset must contain both real and fake samples for balanced sampling.")

        self.half = batch_size_per_gpu // 2   # half-batch per rank
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Pad each class up to a multiple of (half * num_replicas)
        step = self.half * num_replicas
        n = max(len(self.real_indices), len(self.fake_indices))
        self.n_per_class = ((n + step - 1) // step) * step
        # Each rank yields (n_per_class // num_replicas) real + same fake indices
        self.num_samples = (self.n_per_class // num_replicas) * 2

        logger.info(
            f"[DistributedBalancedSampler] rank={rank}/{num_replicas}  "
            f"real={len(self.real_indices)}  fake={len(self.fake_indices)}  "
            f"n_per_class={self.n_per_class}  num_samples/rank={self.num_samples}  "
            f"half_batch={self.half}"
        )

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        real = list(self.real_indices)
        fake = list(self.fake_indices)

        if self.shuffle:
            real = [real[i] for i in torch.randperm(len(real), generator=g).tolist()]
            fake = [fake[i] for i in torch.randperm(len(fake), generator=g).tolist()]

        n = self.n_per_class
        # Tile to reach target size
        real = (real * ((n // len(real)) + 1))[:n]
        fake = (fake * ((n // len(fake)) + 1))[:n]

        half = self.half
        total_half_blocks = n // half  # total half-blocks per class across all ranks

        # Round-robin half-blocks to ranks so indices never overlap between ranks.
        # Rank r owns half-blocks: r, r+num_replicas, r+2*num_replicas, ...
        # Each pair (real_block_b, fake_block_b) → one DataLoader batch on this rank.
        result = []
        for blk in range(self.rank, total_half_blocks, self.num_replicas):
            s = blk * half
            result.extend(real[s: s + half])
            result.extend(fake[s: s + half])

        return iter(result)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
