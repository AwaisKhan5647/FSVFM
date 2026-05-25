#!/usr/bin/env python3
"""
Kutub FOUNDATIONS training — per-epoch multi-dataset evaluation.

Single-GPU training on LSUN/FOUNDATIONS with two variants:
  Variant A: --no_compression_aug   (GPU 3)
  Variant B: --use_compression_aug  (GPU 4)

Per-epoch evaluation on image_eval24, ReWIND_images, AIGI_TEST.
Best checkpoint = epoch with highest average AUC across those 3 datasets.

Usage (no torchrun — single GPU):
  CUDA_VISIBLE_DEVICES=3 python experiment_image_kutub/train_kutub.py \\
      --output_dir experiment_image_kutub/runs/variant_a_noaug \\
      --no_compression_aug

  CUDA_VISIBLE_DEVICES=4 python experiment_image_kutub/train_kutub.py \\
      --output_dir experiment_image_kutub/runs/variant_b_aug \\
      --use_compression_aug

Resume (auto-detects checkpoint-latest.pth in --output_dir):
  Add --resume to the command above.
"""

import argparse
import datetime
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from sklearn.metrics import roc_auc_score
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    from timm.layers import trunc_normal_
except ImportError:
    from timm.models.layers import trunc_normal_

# ---- FSVFM paths ----
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "fsvfm"))
sys.path.insert(0, os.path.join(REPO_ROOT, "fsvfm", "finetune", "cross_dataset_DFD_and_DiFF"))
sys.path.insert(0, os.path.join(REPO_ROOT, "experiment_image"))

import models_vit
import util.lr_decay as lrd
from util.pos_embed import interpolate_pos_embed

from augmentations.compression_augment import (
    CompressionAugment,
    CompressionAwareTransform,
    load_augment_config,
)
from dataset_kutub import BalancedSampler, KutubDataset, scan_image_dir

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Dataset paths
FOUNDATIONS_ROOT  = "/data/Kutub/DEEPFAKES/DATASETS/LSUN/FOUNDATIONS"
IMAGE_EVAL24_ROOT = "/mnt/h200_dataset/saad/datasets/images/image_eval24"
REWIND_ROOT       = "/mnt/h200_dataset/saad/datasets/images/ancestree/ReWIND"
AIGI_ROOT         = "/mnt/h200_dataset/kutub/datasets/AIGI/TEST"

DEFAULT_AUG_CFG = os.path.join(
    os.path.dirname(__file__), "configs", "augment_default.json"
)
DEFAULT_FINETUNE = os.path.join(
    REPO_ROOT,
    "experiments/fsvfm_vitl_composite_20260508_160444/FSFM_best_checkpoint.pth",
)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser("Kutub FOUNDATIONS training", add_help=True)

    # Training
    p.add_argument("--epochs",        default=20,  type=int)
    p.add_argument("--batch_size",    default=64,  type=int)
    p.add_argument("--warmup_epochs", default=2,   type=int)
    p.add_argument("--accum_iter",    default=1,   type=int)

    # Model
    p.add_argument("--model",      default="vit_large_patch16", type=str)
    p.add_argument("--input_size", default=224, type=int)
    p.add_argument("--drop_path",  default=0.1, type=float)
    p.add_argument("--nb_classes", default=2,   type=int)

    # Optimizer
    p.add_argument("--blr",          default=5e-5, type=float)
    p.add_argument("--lr",           default=None, type=float)
    p.add_argument("--layer_decay",  default=0.90, type=float)
    p.add_argument("--weight_decay", default=0.01, type=float)
    p.add_argument("--min_lr",       default=1e-6, type=float)
    p.add_argument("--clip_grad",    default=None, type=float)
    p.add_argument("--smoothing",    default=0.1,  type=float)

    # Mixup
    p.add_argument("--mixup",  default=0.8, type=float)
    p.add_argument("--cutmix", default=1.0, type=float)

    # Augmentation
    p.add_argument("--use_compression_aug",  action="store_true",  default=False)
    p.add_argument("--no_compression_aug",   action="store_false", dest="use_compression_aug")
    p.add_argument("--augment_config",       default=DEFAULT_AUG_CFG, type=str)
    p.add_argument("--aug_prob_scale",       default=1.0, type=float)

    # Training data
    p.add_argument("--train_root", default=FOUNDATIONS_ROOT, type=str)

    # Test datasets
    p.add_argument("--image_eval24_root", default=IMAGE_EVAL24_ROOT, type=str)
    p.add_argument("--rewind_root",       default=REWIND_ROOT,       type=str)
    p.add_argument("--aigi_root",         default=AIGI_ROOT,         type=str)

    # Paths
    p.add_argument("--finetune",   default=DEFAULT_FINETUNE, type=str)
    p.add_argument("--output_dir", default="", type=str)
    p.add_argument("--resume",     action="store_true",
                   help="Auto-resume from checkpoint-latest.pth in output_dir")

    # Training data cap (0 = use all)
    p.add_argument("--max_per_class", default=0, type=int,
                   help="Cap training samples per class (0=use all). "
                        "E.g. 100000 gives 200K balanced samples/epoch.")

    # Misc
    p.add_argument("--seed",        default=0,  type=int)
    p.add_argument("--num_workers", default=8,  type=int)
    p.add_argument("--eval_batch",  default=128, type=int)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def make_train_transform(input_size: int = 224, aa: str = "rand-m9-mstd0.5-inc1",
                          reprob: float = 0.25):
    return create_transform(
        input_size=input_size,
        is_training=True,
        color_jitter=None,
        auto_augment=aa,
        re_prob=reprob,
        re_mode="pixel",
        re_count=1,
        interpolation="bicubic",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )


def make_val_transform(input_size: int = 224):
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def adjust_lr(optimizer, epoch: int, args) -> float:
    """Linear warmup then cosine decay."""
    if epoch < args.warmup_epochs:
        lr = args.lr * max(epoch, 1) / args.warmup_epochs
    else:
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr * pg.get("lr_scale", 1.0)
    return lr


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scaler, criterion, mixup_fn,
                    device, epoch: int, args) -> Dict:
    model.train()
    total_loss = 0.0
    n_batches  = 0
    sampler    = loader.sampler
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)

    for step, (samples, targets, _paths) in enumerate(loader):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss    = criterion(outputs, targets)

        loss_val = loss.item()
        if not math.isfinite(loss_val):
            log.error(f"Loss is {loss_val} at epoch {epoch+1} step {step} — stopping.")
            sys.exit(1)

        scaler.scale(loss).backward()
        if args.clip_grad is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss_val
        n_batches  += 1

        if step % 100 == 0:
            log.info(
                f"Epoch {epoch+1}/{args.epochs} [{step}/{len(loader)}] "
                f"loss={loss_val:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}"
            )

    return {"loss": total_loss / max(n_batches, 1)}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_folder(root: str, model, device, args, name: str = "") -> Dict:
    """
    Evaluate model on a folder dataset (0_real/1_fake label from path).
    Returns dict with accuracy, auc, real_accuracy, fake_accuracy.
    """
    samples = scan_image_dir(root)
    if not samples:
        log.warning(f"  {name}: no labeled images found in {root}")
        return {"accuracy": 0, "auc": 0, "real_accuracy": 0, "fake_accuracy": 0, "n": 0}

    val_tf  = make_val_transform(args.input_size)
    dataset = KutubDataset(samples, transform=val_tf)
    loader  = DataLoader(
        dataset, batch_size=args.eval_batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    model.eval()
    all_probs  = []
    all_labels = []

    for imgs, labels, _paths in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)[:, 1]
        all_probs.extend(probs.cpu().float().numpy())
        all_labels.extend(labels.numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    preds    = (all_probs >= 0.5).astype(int)
    accuracy = float(np.mean(preds == all_labels)) * 100.0

    real_mask    = all_labels == 0
    fake_mask    = all_labels == 1
    real_accuracy = float(np.mean(preds[real_mask] == 0)) * 100.0 if real_mask.any() else 0.0
    fake_accuracy = float(np.mean(preds[fake_mask] == 1)) * 100.0 if fake_mask.any() else 0.0

    try:
        auc = float(roc_auc_score(all_labels, all_probs)) * 100.0
    except Exception:
        auc = 0.0

    n_real = int(real_mask.sum())
    n_fake = int(fake_mask.sum())
    log.info(
        f"  {name:20s}  n={len(all_labels):6d} ({n_real}R/{n_fake}F) | "
        f"Acc={accuracy:.2f}%  AUC={auc:.2f}%  "
        f"Real={real_accuracy:.2f}%  Fake={fake_accuracy:.2f}%"
    )
    return {
        "accuracy":      accuracy,
        "auc":           auc,
        "real_accuracy": real_accuracy,
        "fake_accuracy": fake_accuracy,
        "n": len(all_labels),
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(output_dir: str, tag: str, epoch: int,
                    model, optimizer, scaler, best_avg_auc: float,
                    best_per_ds: Dict) -> None:
    path = Path(output_dir) / f"checkpoint-{tag}.pth"
    payload = {
        "model":        model.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "scaler":       scaler.state_dict(),
        "epoch":        epoch,
        "best_avg_auc": best_avg_auc,
        "best_per_ds":  best_per_ds,
    }
    torch.save(payload, path)
    log.info(f"  Checkpoint saved: {path}")


def load_checkpoint(path: str, model, optimizer, scaler):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    epoch        = int(ckpt.get("epoch", -1)) + 1
    best_avg_auc = float(ckpt.get("best_avg_auc", 0.0))
    best_per_ds  = ckpt.get("best_per_ds", {})
    return epoch, best_avg_auc, best_per_ds


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _append(path: str, text: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)


def write_epoch_results(result_path: str, progress_path: str, epoch: int,
                        train_stats: Dict, eval_stats: Dict,
                        avg_auc: float, is_best: bool) -> None:
    tag = " [BEST_AVG_AUC]" if is_best else ""
    ds_names = ["image_eval24", "ReWIND_images", "AIGI_TEST"]

    # result.txt — tabular row per epoch
    row = (
        f"Epoch {epoch+1:3d}{tag}\n"
        f"  train_loss  : {train_stats['loss']:.4f}\n"
        f"  avg_AUC     : {avg_auc:.2f}%\n"
    )
    for ds in ds_names:
        s = eval_stats.get(ds, {})
        row += (
            f"  {ds:20s}: Acc={s.get('accuracy',0):.2f}%  "
            f"AUC={s.get('auc',0):.2f}%  "
            f"Real={s.get('real_accuracy',0):.2f}%  "
            f"Fake={s.get('fake_accuracy',0):.2f}%\n"
        )
    row += "\n"
    _append(result_path, row)

    # progress.txt — single summary line per epoch
    ds_short = "  ".join(
        f"{ds.split('_')[0]}={eval_stats.get(ds, {}).get('auc', 0):.1f}%"
        for ds in ds_names
    )
    _append(
        progress_path,
        f"Epoch {epoch+1:3d} | train={train_stats['loss']:.4f} | "
        f"avgAUC={avg_auc:.2f}%{' *BEST*' if is_best else ''} | "
        f"{ds_short}\n"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    if not args.output_dir:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        variant = "aug" if args.use_compression_aug else "noaug"
        args.output_dir = os.path.join(
            os.path.dirname(__file__), "runs",
            f"kutub_vitl_{variant}_{ts}",
        )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ---- Redirect stdout/stderr to log file ----
    log_file_path = os.path.join(args.output_dir, "log_detail.txt")
    if not os.environ.get("NO_LOG_REDIRECT"):
        log_fh = open(log_file_path, "a")
        sys.stdout = log_fh
        sys.stderr = log_fh

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    log.info(f"GPU: {gpu_ids}  |  Output: {args.output_dir}")
    log.info(f"Compression aug: {'ENABLED' if args.use_compression_aug else 'DISABLED'}")

    # ---- Training dataset ----
    log.info(f"Scanning training data: {args.train_root}")
    train_samples = scan_image_dir(args.train_root)
    n_real = sum(1 for _, l in train_samples if l == 0)
    n_fake = sum(1 for _, l in train_samples if l == 1)
    log.info(f"Training samples: {len(train_samples):,} ({n_real:,} real / {n_fake:,} fake)")

    if args.max_per_class and args.max_per_class > 0:
        import random as _rng
        rng = _rng.Random(args.seed)
        real_s = [s for s in train_samples if s[1] == 0]
        fake_s = [s for s in train_samples if s[1] == 1]
        rng.shuffle(real_s); rng.shuffle(fake_s)
        real_s = real_s[:args.max_per_class]
        fake_s = fake_s[:args.max_per_class]
        train_samples = real_s + fake_s
        n_real = len(real_s); n_fake = len(fake_s)
        log.info(f"Capped to {args.max_per_class}/class → {len(train_samples):,} samples")

    base_train_tf = make_train_transform(args.input_size)
    if args.use_compression_aug:
        aug_cfg      = load_augment_config(args.augment_config)
        comp_aug     = CompressionAugment(aug_cfg, prob_scale=args.aug_prob_scale)
        train_tf     = CompressionAwareTransform(comp_aug, base_train_tf)
        log.info(f"CompressionAugment active: {comp_aug}")
    else:
        train_tf = base_train_tf

    train_dataset = KutubDataset(train_samples, transform=train_tf)
    sampler       = BalancedSampler(train_samples, seed=args.seed)
    loader_train  = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    log.info(f"Balanced sampler: {len(sampler):,} samples/epoch (50/50 real/fake)")

    # ---- Model ----
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=True,
    )

    resume_path = Path(args.output_dir) / "checkpoint-latest.pth"
    if not args.resume or not resume_path.exists():
        # Load pretrained weights
        if args.finetune and Path(args.finetune).exists():
            log.info(f"Loading pretrained: {args.finetune}")
            ckpt       = torch.load(args.finetune, map_location="cpu", weights_only=False)
            ckpt_model = ckpt.get("model", ckpt)
            state_dict = model.state_dict()
            for k in ["head.weight", "head.bias"]:
                if k in ckpt_model and ckpt_model[k].shape != state_dict[k].shape:
                    del ckpt_model[k]
            interpolate_pos_embed(model, ckpt_model)
            msg = model.load_state_dict(ckpt_model, strict=False)
            log.info(f"Pretrained load: {msg}")
            trunc_normal_(model.head.weight, std=2e-5)
        else:
            log.warning(f"No pretrained checkpoint found at {args.finetune}")

    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable params: {n_params/1e6:.1f}M")

    # ---- Optimizer ----
    eff_batch = args.batch_size * args.accum_iter
    if args.lr is None:
        args.lr = args.blr * eff_batch / 256
    log.info(f"Effective batch: {eff_batch}  |  LR: {args.lr:.2e}")

    param_groups = lrd.param_groups_lrd(
        model, args.weight_decay,
        no_weight_decay_list=model.no_weight_decay(),
        layer_decay=args.layer_decay,
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    scaler    = GradScaler()

    # ---- Loss ----
    mixup_fn = None
    if args.mixup > 0 or args.cutmix > 0:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
            prob=1.0, switch_prob=0.5, mode="batch",
            label_smoothing=args.smoothing, num_classes=args.nb_classes,
        )
    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # ---- Resume ----
    start_epoch  = 0
    best_avg_auc = 0.0
    best_per_ds  = {}

    if args.resume and resume_path.exists():
        log.info(f"Resuming from {resume_path}")
        start_epoch, best_avg_auc, best_per_ds = load_checkpoint(
            str(resume_path), model, optimizer, scaler
        )
        log.info(f"Resumed at epoch {start_epoch}  best_avg_auc={best_avg_auc:.2f}%")

    # ---- Save config snapshot ----
    config_snap = Path(args.output_dir) / "config.json"
    if not config_snap.exists():
        with open(config_snap, "w") as f:
            json.dump(vars(args), f, indent=2)

    aug_dst = Path(args.output_dir) / "augment_config_used.json"
    if not aug_dst.exists() and args.use_compression_aug and Path(args.augment_config).exists():
        import shutil
        shutil.copy2(args.augment_config, aug_dst)

    # ---- Init progress / result files ----
    result_path   = str(Path(args.output_dir) / "result.txt")
    progress_path = str(Path(args.output_dir) / "progress.txt")

    if start_epoch == 0:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = (
            "=" * 70 + "\n"
            f"KUTUB FOUNDATIONS TRAINING — {now}\n"
            f"Output dir  : {args.output_dir}\n"
            f"GPU         : CUDA_VISIBLE_DEVICES={gpu_ids}\n"
            f"Train root  : {args.train_root}\n"
            f"Train data  : {len(train_samples):,} ({n_real:,} real / {n_fake:,} fake)\n"
            f"Model       : {args.model}\n"
            f"Batch/GPU   : {args.batch_size}  (balanced 50/50)\n"
            f"Epochs      : {args.epochs}  (warmup {args.warmup_epochs})\n"
            f"Base LR     : {args.blr}  →  {args.lr:.2e}\n"
            f"Comp aug    : {'ENABLED' if args.use_compression_aug else 'DISABLED'}\n"
            f"Aug config  : {args.augment_config}\n"
            "=" * 70 + "\n\n"
        )
        _append(progress_path, header)
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(f"# Kutub FOUNDATIONS Training Results — {now}\n")
            f.write(f"# Compression aug: {'ENABLED' if args.use_compression_aug else 'DISABLED'}\n")
            f.write(f"# Best model = highest average AUC across image_eval24 + ReWIND_images + AIGI_TEST\n")
            f.write("-" * 70 + "\n\n")

    # ---- Test dataset roots ----
    test_datasets = {
        "image_eval24":  args.image_eval24_root,
        "ReWIND_images": args.rewind_root,
        "AIGI_TEST":     args.aigi_root,
    }
    for name, root in test_datasets.items():
        if not Path(root).exists():
            log.warning(f"Test dataset not found: {name} → {root}")

    # ---- Training loop ----
    log.info(f"Starting training from epoch {start_epoch+1} to {args.epochs}")
    train_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        lr = adjust_lr(optimizer, epoch, args)
        log.info(f"\n{'='*60}")
        log.info(f"Epoch {epoch+1}/{args.epochs}  LR={lr:.2e}")

        train_stats = train_one_epoch(
            model, loader_train, optimizer, scaler, criterion, mixup_fn,
            device, epoch, args,
        )
        log.info(f"Epoch {epoch+1} train_loss={train_stats['loss']:.4f}")

        # Per-epoch evaluation on all 3 test datasets
        log.info(f"Evaluating on {len(test_datasets)} test datasets...")
        eval_stats: Dict[str, Dict] = {}
        for ds_name, ds_root in test_datasets.items():
            if Path(ds_root).exists():
                eval_stats[ds_name] = evaluate_folder(ds_root, model, device, args, name=ds_name)
            else:
                eval_stats[ds_name] = {"accuracy": 0, "auc": 0,
                                        "real_accuracy": 0, "fake_accuracy": 0, "n": 0}

        # Average AUC across 3 datasets
        aucs = [eval_stats[ds]["auc"] for ds in test_datasets if eval_stats[ds]["n"] > 0]
        avg_auc = float(np.mean(aucs)) if aucs else 0.0
        is_best = avg_auc > best_avg_auc

        log.info(
            f"  avg_AUC={avg_auc:.2f}%  best={best_avg_auc:.2f}%  "
            f"{'→ NEW BEST' if is_best else ''}"
        )

        # Save checkpoints
        save_checkpoint(args.output_dir, "latest", epoch, model, optimizer, scaler,
                        best_avg_auc if not is_best else avg_auc, best_per_ds)
        if is_best:
            best_avg_auc = avg_auc
            best_per_ds  = {ds: eval_stats[ds]["auc"] for ds in test_datasets}
            save_checkpoint(args.output_dir, "best_avg_auc", epoch, model, optimizer, scaler,
                            best_avg_auc, best_per_ds)
            log.info(f"  New best_avg_auc checkpoint: {best_avg_auc:.2f}%")

        # Log results
        write_epoch_results(result_path, progress_path, epoch,
                            train_stats, eval_stats, avg_auc, is_best)

        epoch_secs = time.time() - epoch_start
        remaining  = (args.epochs - epoch - 1) * epoch_secs
        log.info(
            f"  Epoch time: {epoch_secs/60:.1f} min  "
            f"ETA: {str(datetime.timedelta(seconds=int(remaining)))}"
        )

        # Also write per-epoch JSON log
        log_row = {
            "epoch":        epoch + 1,
            "train_loss":   train_stats["loss"],
            "avg_auc":      avg_auc,
            "best_avg_auc": best_avg_auc,
            **{f"{ds}_auc": eval_stats[ds]["auc"] for ds in test_datasets},
            **{f"{ds}_acc": eval_stats[ds]["accuracy"] for ds in test_datasets},
        }
        with open(os.path.join(args.output_dir, "train_log.txt"), "a") as f:
            f.write(json.dumps(log_row) + "\n")

    # ---- Final summary ----
    total_time = str(datetime.timedelta(seconds=int(time.time() - train_start)))
    log.info(f"\nTraining complete in {total_time}")
    log.info(f"Best avg AUC: {best_avg_auc:.2f}%")
    log.info(f"Per-dataset: {best_per_ds}")

    _append(
        progress_path,
        f"\n{'='*70}\n"
        f"Training complete in {total_time}\n"
        f"Best avg AUC: {best_avg_auc:.2f}%\n"
        + "".join(f"  {ds}: {auc:.2f}%\n" for ds, auc in best_per_ds.items())
        + "=" * 70 + "\n"
    )
    _append(
        result_path,
        f"\n# FINAL BEST (avg AUC = {best_avg_auc:.2f}%)\n"
        + "".join(f"#   {ds}: AUC={auc:.2f}%\n" for ds, auc in best_per_ds.items())
    )


if __name__ == "__main__":
    main()
