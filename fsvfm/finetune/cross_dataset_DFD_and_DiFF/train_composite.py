#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Robust FSVFM fine-tuning on composite deepfake datasets
# Fixes: timm 1.0.20 compat, DDP via torchrun, crash recovery, best+latest ckpt saving
#
# Launch:  CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 train_composite.py [args]
# Resume:  add --resume path/to/checkpoint-latest.pth

import argparse
import datetime
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

# timm 1.0.x compatibility: trunc_normal_ moved from timm.models.layers → timm.layers
try:
    from timm.layers import trunc_normal_
except ImportError:
    from timm.models.layers import trunc_normal_

from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

# Add fsvfm root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_transform
from util.composite_dataset import (
    CompositeDeepfakeDataset, build_composite_dataset,
    build_from_txt_pair, DistributedBalancedSampler,
)
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import models_vit

# Patch engine_finetune path
sys.path.insert(0, os.path.dirname(__file__))
from engine_finetune import train_one_epoch, evaluate_full

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful shutdown on SIGTERM/SIGINT — saves a crash checkpoint
# ---------------------------------------------------------------------------
_CRASH_SAVE_PATH: str = ""


def _signal_handler(signum, frame):
    if _CRASH_SAVE_PATH:
        log.warning(f"Received signal {signum}. Crash checkpoint should already be at: {_CRASH_SAVE_PATH}")
    sys.exit(0)


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_args_parser():
    p = argparse.ArgumentParser("FSVFM composite-dataset fine-tuning", add_help=False)

    # Training schedule
    p.add_argument("--batch_size", default=64, type=int, help="Batch size per GPU (must be even for --balanced_batch)")
    p.add_argument("--epochs", default=50, type=int)
    p.add_argument("--accum_iter", default=1, type=int)
    p.add_argument("--warmup_epochs", default=5, type=int)
    p.add_argument("--save_every", default=1, type=int, help="Save latest checkpoint every N epochs")

    # Model
    p.add_argument("--model", default="vit_large_patch16", type=str)
    p.add_argument("--input_size", default=224, type=int)
    p.add_argument("--drop_path", default=0.1, type=float)
    p.add_argument("--global_pool", action="store_true", default=True)
    p.add_argument("--cls_token", action="store_false", dest="global_pool")
    p.add_argument("--nb_classes", default=2, type=int)

    # Optimizer
    p.add_argument("--lr", default=None, type=float)
    p.add_argument("--blr", default=5e-5, type=float, help="Base LR; actual = blr * batch / 256")
    p.add_argument("--layer_decay", default=0.90, type=float)
    p.add_argument("--weight_decay", default=0.01, type=float)
    p.add_argument("--min_lr", default=1e-6, type=float)
    p.add_argument("--clip_grad", default=None, type=float)

    # Augmentation
    p.add_argument("--apply_simple_augment", action="store_true", default=True)
    p.add_argument("--normalize_from_IMN", action="store_true", default=True,
                   help="Use ImageNet normalization (avoids dependency on pretrain stats file)")
    p.add_argument("--color_jitter", default=None, type=float)
    p.add_argument("--aa", default="rand-m9-mstd0.5-inc1", type=str)
    p.add_argument("--smoothing", default=0.1, type=float)
    p.add_argument("--reprob", default=0.25, type=float)
    p.add_argument("--remode", default="pixel", type=str)
    p.add_argument("--recount", default=1, type=int)
    p.add_argument("--resplit", action="store_true", default=False)
    p.add_argument("--mixup", default=0.8, type=float)
    p.add_argument("--cutmix", default=1.0, type=float)
    p.add_argument("--cutmix_minmax", nargs="+", default=None, type=float)
    p.add_argument("--mixup_prob", default=1.0, type=float)
    p.add_argument("--mixup_switch_prob", default=0.5, type=float)
    p.add_argument("--mixup_mode", default="batch", type=str)

    # Dataset  — pass multiple --train_root / --val_root for composite loading
    p.add_argument("--train_root", action="append", default=[],
                   help="Root dir(s) for training. Repeat for multiple datasets.")
    p.add_argument("--val_root", action="append", default=[],
                   help="Root dir(s) for validation.")
    p.add_argument("--train_label_file", action="append", default=[],
                   help="Optional label .txt file(s) paired with --train_root (same order).")
    p.add_argument("--val_label_file", action="append", default=[],
                   help="Optional label .txt file(s) paired with --val_root (same order).")
    p.add_argument("--balance", action="store_true", default=False,
                   help="Oversample minority class to 1:1 real/fake ratio.")
    p.add_argument("--max_per_dataset", default=None, type=int,
                   help="Cap samples per sub-dataset (None = no cap).")

    # Separate real/fake txt-file dataset inputs (one absolute path per line)
    p.add_argument("--real_train_txt", default="", type=str,
                   help="Txt file: one real-frame path per line (training).")
    p.add_argument("--fake_train_txt", default="", type=str,
                   help="Txt file: one fake-frame path per line (training).")
    p.add_argument("--real_test_txt", default="", type=str,
                   help="Txt file: one real-frame path per line (test/eval).")
    p.add_argument("--fake_test_txt", default="", type=str,
                   help="Txt file: one fake-frame path per line (test/eval).")

    # Path prefix remapping for txt files (e.g. stored paths use a different mount point)
    p.add_argument("--txt_path_remap_from", default="", type=str,
                   help="Prefix to replace in txt file paths (e.g. /data/saad/datasets/gend_unified).")
    p.add_argument("--txt_path_remap_to", default="", type=str,
                   help="Replacement prefix (e.g. /mnt/h200_dataset/gend_unified).")

    # Balanced batch sampling
    p.add_argument("--balanced_batch", action="store_true", default=True,
                   help="Guarantee equal real/fake samples in every batch (DistributedBalancedSampler).")
    p.add_argument("--no_balanced_batch", action="store_false", dest="balanced_batch")

    # Paths
    p.add_argument("--finetune", default="", help="Pre-trained checkpoint to fine-tune from")
    p.add_argument("--output_dir", default="", help="Output directory for checkpoints and logs")
    p.add_argument("--log_dir", default="", help="TensorBoard log directory (defaults to output_dir)")
    p.add_argument("--resume", default="", help="Resume training from checkpoint")

    # Misc
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--num_workers", default=10, type=int)
    p.add_argument("--pin_mem", action="store_true", default=True)
    p.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    p.add_argument("--start_epoch", default=0, type=int)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--dist_eval", action="store_true", default=True)
    p.add_argument("--dry_run", action="store_true",
                   help="Run one batch forward/backward then exit — for pre-flight checks.")

    # DDP (set by torchrun environment automatically)
    p.add_argument("--world_size", default=1, type=int)
    p.add_argument("--local_rank", default=-1, type=int)
    p.add_argument("--dist_on_itp", action="store_true")
    p.add_argument("--dist_url", default="env://")

    return p


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _pair_roots_and_labels(roots, label_files):
    """Zip roots with optional label files, padding with None."""
    pairs = []
    for i, root in enumerate(roots):
        lf = label_files[i] if i < len(label_files) else None
        pairs.append((root, lf) if lf else root)
    return pairs


def build_datasets(args, is_train: bool):
    """Build train or test dataset from txt-pair files (preferred) or directory roots."""
    real_txt = args.real_train_txt if is_train else args.real_test_txt
    fake_txt = args.fake_train_txt if is_train else args.fake_test_txt

    if real_txt and fake_txt:
        return build_from_txt_pair(
            real_txt=real_txt,
            fake_txt=fake_txt,
            is_train=is_train and not args.eval,
            args=args,
            balance=args.balance and is_train,
            max_per_class=args.max_per_dataset,
            path_remap_from=getattr(args, "txt_path_remap_from", ""),
            path_remap_to=getattr(args, "txt_path_remap_to", ""),
        )

    # Fall back to directory-based loading
    roots = args.train_root if is_train else args.val_root
    label_files = args.train_label_file if is_train else args.val_label_file

    if not roots:
        raise ValueError(
            "Provide --real_train_txt + --fake_train_txt  OR  --train_root "
            "(and corresponding test/val variants)."
        )

    pairs = _pair_roots_and_labels(roots, label_files)
    return build_composite_dataset(
        roots=pairs,
        is_train=is_train and not args.eval,
        args=args,
        balance=args.balance and is_train,
        max_per_dataset=args.max_per_dataset,
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(args, epoch, model_without_ddp, optimizer, loss_scaler, tag,
                    best_auc=0.0, best_acc=0.0, min_test_loss=float("inf")):
    ckpt_dir = Path(args.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"checkpoint-{tag}.pth"
    payload = {
        "model": model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "scaler": loss_scaler.state_dict(),
        "args": vars(args),
        # Best stats persisted so resume continues from correct baseline
        "best_auc": best_auc,
        "best_acc": best_acc,
        "min_test_loss": min_test_loss,
    }
    misc.save_on_master(payload, path)
    if misc.is_main_process():
        log.info(f"Checkpoint saved: {path}")
    return str(path)


def load_best_stats(resume_path: str):
    """Load best_auc/best_acc/min_test_loss from a checkpoint (for resume)."""
    if not resume_path or not Path(resume_path).exists():
        return 0.0, 0.0, float("inf")
    try:
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        best_auc      = float(ckpt.get("best_auc",      0.0))
        best_acc      = float(ckpt.get("best_acc",      0.0))
        min_test_loss = float(ckpt.get("min_test_loss", float("inf")))
        return best_auc, best_acc, min_test_loss
    except Exception as e:
        log.warning(f"Could not load best stats from checkpoint: {e}")
        return 0.0, 0.0, float("inf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _write_progress(progress_path: str, text: str):
    """Append text to progress.txt (main process only)."""
    if not misc.is_main_process():
        return
    with open(progress_path, "a", encoding="utf-8") as fh:
        fh.write(text)


def _dataset_stats_block(dataset, label: str) -> str:
    """Return a formatted stats block for a dataset."""
    n_real = sum(1 for _, l in dataset.all_samples if l == 0)
    n_fake = sum(1 for _, l in dataset.all_samples if l == 1)
    total = len(dataset.all_samples)
    return (
        f"  {label}:\n"
        f"    Total  : {total:,}\n"
        f"    Real   : {n_real:,}\n"
        f"    Fake   : {n_fake:,}\n"
    )


def _write_result_row(result_path: str, epoch: int, stats: dict, tag: str = ""):
    """Append one result row to result.txt."""
    if not misc.is_main_process():
        return
    marker = f"[{tag}] " if tag else ""
    line = (
        f"Epoch {epoch+1:3d} {marker}| "
        f"Accuracy={stats.get('accuracy', 0):.2f}% | "
        f"AUC={stats.get('auc', 0):.2f}% | "
        f"Real_Accuracy={stats.get('real_accuracy', 0):.2f}% | "
        f"Fake_Accuracy={stats.get('fake_accuracy', 0):.2f}% | "
        f"Loss={stats.get('loss', 0):.4f}\n"
    )
    with open(result_path, "a", encoding="utf-8") as fh:
        fh.write(line)


def main(args):
    global _CRASH_SAVE_PATH

    misc.init_distributed_mode(args)

    if misc.is_main_process():
        log.info(f"Job dir: {os.path.dirname(os.path.realpath(__file__))}")
        log.info(json.dumps(vars(args), indent=2))

    device = torch.device(args.device if hasattr(args, "device") else "cuda")

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # ----- Datasets -----
    dataset_train = build_datasets(args, is_train=True)

    has_test = bool(
        (args.real_test_txt and args.fake_test_txt)
        or args.val_root
    )
    dataset_test = build_datasets(args, is_train=False) if has_test else None

    if misc.is_main_process():
        log.info(f"Train samples: {len(dataset_train)}")
        if dataset_test:
            log.info(f"Test  samples: {len(dataset_test)}")

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # ----- Training sampler: balanced or standard -----
    if args.balanced_batch and not args.eval:
        sampler_train = DistributedBalancedSampler(
            dataset_train,
            batch_size_per_gpu=args.batch_size,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=True,
            seed=args.seed,
        )
        if misc.is_main_process():
            log.info(
                f"Using DistributedBalancedSampler: "
                f"{args.batch_size // 2} real + {args.batch_size // 2} fake per GPU per batch."
            )
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    # ----- Test sampler (sequential on each rank, gather at metric time) -----
    if dataset_test is not None:
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank,
            shuffle=False, drop_last=True,
        )

    if global_rank == 0 and not args.eval:
        log_dir = args.log_dir or args.output_dir
        os.makedirs(log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=log_dir)
        log.info(f"TensorBoard log dir: {log_dir}")
    else:
        log_writer = None

    loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=True,
    )
    loader_test = None
    if dataset_test is not None:
        loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=args.pin_mem, drop_last=False,
        )

    # ----- Mixup -----
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes,
        )

    # ----- Model -----
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    if args.finetune and not args.eval:
        log.info(f"Loading pre-trained weights from: {args.finetune}")
        ckpt = torch.load(args.finetune, map_location="cpu", weights_only=False)
        ckpt_model = ckpt["model"]
        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if k in ckpt_model and ckpt_model[k].shape != state_dict[k].shape:
                log.warning(f"Removing mismatched key: {k}")
                del ckpt_model[k]
        interpolate_pos_embed(model, ckpt_model)
        msg = model.load_state_dict(ckpt_model, strict=False)
        log.info(f"Checkpoint load: {msg}")
        trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)
    model_without_ddp = model

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable params: {n_params / 1e6:.2f}M")

    eff_batch = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch / 256
    log.info(f"Effective batch size: {eff_batch}  |  LR: {args.lr:.2e}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # ----- Optimizer -----
    param_groups = lrd.param_groups_lrd(
        model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay,
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    # ----- Loss -----
    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # ----- Resume -----
    misc.load_model(args=args, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler)

    # Restore best-metric baselines from the resume checkpoint so comparisons
    # after restart continue from the correct high-water marks, not from zero.
    best_auc, best_acc, min_test_loss = load_best_stats(args.resume)
    if args.resume and misc.is_main_process():
        log.info(
            f"Resumed best stats — AUC={best_auc:.2f}% "
            f"Acc={best_acc:.2f}% min_loss={min_test_loss:.4f}"
        )

    # Set crash-save path so signal handler can report it
    if args.output_dir:
        _CRASH_SAVE_PATH = str(Path(args.output_dir) / "checkpoint-latest.pth")

    # ----- Log dataset info to progress.txt -----
    progress_path = str(Path(args.output_dir) / "progress.txt")
    result_path   = str(Path(args.output_dir) / "result.txt")

    if misc.is_main_process() and not args.eval and not args.dry_run:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        exp_name = Path(args.output_dir).name
        gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
        eff_batch = args.batch_size * args.accum_iter * num_tasks
        half = args.batch_size // 2 if args.balanced_batch else "N/A"

        header = (
            "\n" + "=" * 70 + "\n"
            f"Training started : {now}\n"
            f"Experiment       : {exp_name}\n"
            f"Output dir       : {args.output_dir}\n"
            + "-" * 70 + "\n"
            + _dataset_stats_block(dataset_train, "TRAIN SET")
        )
        if dataset_test:
            header += _dataset_stats_block(dataset_test, "TEST SET")
        header += (
            "-" * 70 + "\n"
            f"  Model          : {args.model}\n"
            f"  Pretrained ckpt: {args.finetune}\n"
            f"  GPUs           : CUDA_VISIBLE_DEVICES={gpu_ids}  ({num_tasks} rank(s))\n"
            f"  Batch / GPU    : {args.batch_size}  "
            f"({half} real + {half} fake per GPU)\n"
            f"  Effective batch: {eff_batch}\n"
            f"  Epochs         : {args.epochs}  (warmup {args.warmup_epochs})\n"
            f"  Base LR        : {args.blr}  →  actual {args.lr:.2e}\n"
            f"  Balanced batch : {args.balanced_batch}\n"
            + "=" * 70 + "\n\n"
        )
        _write_progress(progress_path, header)

        # Write result.txt header
        with open(result_path, "w", encoding="utf-8") as fh:
            fh.write(f"# FSVFM Training Results — {now}\n")
            fh.write(f"# Experiment: {exp_name}\n")
            fh.write(
                "# Epoch | Accuracy | AUC | Real_Accuracy | Fake_Accuracy | Loss\n"
            )
            fh.write("-" * 80 + "\n")

    # ----- Dry run -----
    if args.dry_run:
        log.info("=== DRY RUN: one forward+backward pass ===")
        model.train()
        batch = next(iter(loader_train))
        imgs, labels = batch[0].to(device), batch[1].to(device)
        n_real = (labels == 0).sum().item()
        n_fake = (labels == 1).sum().item()
        log.info(f"Dry-run batch label mix: {n_real} real / {n_fake} fake (total {len(labels)})")
        # Use plain CrossEntropyLoss for dry-run (bypasses mixup soft-target format)
        dry_criterion = torch.nn.CrossEntropyLoss().to(device)
        with torch.cuda.amp.autocast():
            out = model(imgs)
            loss = dry_criterion(out, labels)
        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=True)
        log.info(f"Dry-run train loss: {loss.item():.4f}  — PASSED")
        if loader_test is not None:
            log.info("=== DRY RUN: one test-set forward pass ===")
            model.eval()
            with torch.no_grad():
                tbatch = next(iter(loader_test))
                timgs = tbatch[0].to(device)
                with torch.cuda.amp.autocast():
                    _ = model(timgs)
            log.info("Dry-run test forward  — PASSED")
        if log_writer:
            log_writer.close()
        return

    # ----- Training loop -----
    log.info(f"Start training for {args.epochs} epochs from epoch {args.start_epoch}")
    log.info(f"Best stats at loop start — AUC={best_auc:.2f}% Acc={best_acc:.2f}% min_loss={min_test_loss:.4f}")
    start_time = time.time()
    epoch_times = []   # track per-epoch wall time for ETA

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start = time.time()
        if args.distributed:
            loader_train.sampler.set_epoch(epoch)
            if loader_test is not None:
                loader_test.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer, args=args,
        )

        # Per-epoch evaluation on test set
        if loader_test is not None:
            test_stats = evaluate_full(loader_test, model, device)
        else:
            test_stats = {}

        epoch_secs = time.time() - epoch_start
        epoch_times.append(epoch_secs)
        avg_epoch_secs = sum(epoch_times) / len(epoch_times)
        epochs_done = epoch - args.start_epoch + 1
        epochs_left = args.epochs - epoch - 1
        eta_secs = int(avg_epoch_secs * epochs_left)
        elapsed_total = int(time.time() - start_time)

        if misc.is_main_process():
            auc   = test_stats.get("auc",           0.0)
            acc   = test_stats.get("accuracy",       0.0)
            racc  = test_stats.get("real_accuracy",  0.0)
            facc  = test_stats.get("fake_accuracy",  0.0)
            tloss = test_stats.get("loss",           float("inf"))
            pct   = 100.0 * epochs_done / args.epochs
            bar_filled = int(pct / 5)
            bar = "█" * bar_filled + "░" * (20 - bar_filled)
            print(
                f"\n{'='*70}\n"
                f"  EPOCH {epoch+1:>3}/{args.epochs}  [{bar}] {pct:.0f}%\n"
                f"  Elapsed : {str(datetime.timedelta(seconds=elapsed_total))}  "
                f"ETA: {str(datetime.timedelta(seconds=eta_secs))}  "
                f"(~{avg_epoch_secs/60:.1f} min/epoch)\n"
                f"  ── TEST ──────────────────────────────────────────────────\n"
                f"  Accuracy      : {acc:.2f}%   (best {best_acc:.2f}%)\n"
                f"  AUC           : {auc:.2f}%   (best {best_auc:.2f}%)\n"
                f"  Real_Accuracy : {racc:.2f}%\n"
                f"  Fake_Accuracy : {facc:.2f}%\n"
                f"  Test Loss     : {tloss:.4f}\n"
                f"  Train Loss    : {train_stats.get('loss', 0):.4f}\n"
                f"{'='*70}\n",
                flush=True,
            )
            log.info(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"Acc={acc:.2f}% AUC={auc:.2f}% "
                f"Real={racc:.2f}% Fake={facc:.2f}% "
                f"loss={tloss:.4f} | ETA {str(datetime.timedelta(seconds=eta_secs))}"
            )

        # Write result.txt row after each epoch
        if loader_test is not None:
            tag = ""
            if test_stats.get("auc", 0) >= best_auc and test_stats.get("auc", 0) > 0:
                tag = "BEST_AUC" if test_stats["auc"] > best_auc else ""
            _write_result_row(result_path, epoch, test_stats, tag=tag)

        # Progress log update
        if misc.is_main_process() and loader_test is not None:
            _write_progress(
                progress_path,
                f"Epoch {epoch+1:3d} | "
                f"train_loss={train_stats.get('loss', 0):.4f} | "
                f"Acc={test_stats.get('accuracy', 0):.2f}% | "
                f"AUC={test_stats.get('auc', 0):.2f}% | "
                f"Real_Acc={test_stats.get('real_accuracy', 0):.2f}% | "
                f"Fake_Acc={test_stats.get('fake_accuracy', 0):.2f}%\n",
            )

        # Update best stats before saving so every checkpoint carries current bests
        new_best_auc   = max(best_auc,      test_stats.get("auc",      0.0))
        new_best_acc   = max(best_acc,      test_stats.get("accuracy", 0.0))
        new_min_loss   = min(min_test_loss, test_stats.get("loss",     float("inf")))

        def _ckpt(tag):
            save_checkpoint(
                args, epoch, model_without_ddp, optimizer, loss_scaler, tag,
                best_auc=new_best_auc, best_acc=new_best_acc, min_test_loss=new_min_loss,
            )

        # Always save latest checkpoint (used by watchdog to resume)
        if args.output_dir and (epoch % args.save_every == 0 or epoch + 1 == args.epochs):
            _ckpt("latest")

        # Save best-AUC checkpoint
        if test_stats.get("auc", 0) > best_auc:
            best_auc = new_best_auc
            _ckpt("best_auc")
            if misc.is_main_process():
                log.info(f"New best AUC at epoch {epoch}: {best_auc:.2f}%")

        # Save best-Accuracy checkpoint
        if test_stats.get("accuracy", 0) > best_acc:
            best_acc = new_best_acc
            _ckpt("best_acc")
            if misc.is_main_process():
                log.info(f"New best Accuracy at epoch {epoch}: {best_acc:.2f}%")

        # Save min-loss checkpoint
        if test_stats.get("loss", float("inf")) < min_test_loss:
            min_test_loss = new_min_loss
            _ckpt("min_loss")
            if misc.is_main_process():
                log.info(f"New min test loss at epoch {epoch}: {min_test_loss:.4f}")

        # TensorBoard
        if log_writer is not None:
            log_writer.add_scalar("test/auc",           test_stats.get("auc", 0),           epoch)
            log_writer.add_scalar("test/accuracy",      test_stats.get("accuracy", 0),      epoch)
            log_writer.add_scalar("test/real_accuracy", test_stats.get("real_accuracy", 0), epoch)
            log_writer.add_scalar("test/fake_accuracy", test_stats.get("fake_accuracy", 0), epoch)
            log_writer.add_scalar("test/loss",          test_stats.get("loss", 0),          epoch)
            log_writer.add_scalar("train/loss",         train_stats.get("loss", 0),         epoch)

        # JSON log
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "best_auc": best_auc,
            "best_acc": best_acc,
            "min_test_loss": min_test_loss,
        }
        if args.output_dir and misc.is_main_process():
            if log_writer:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "train_log.txt"), "a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    if misc.is_main_process():
        log.info(f"Training complete in {elapsed}")
        log.info(f"Best AUC:  {best_auc:.2f}%  → {args.output_dir}/checkpoint-best_auc.pth")
        log.info(f"Best Acc:  {best_acc:.2f}%  → {args.output_dir}/checkpoint-best_acc.pth")
        log.info(f"Min loss:  {min_test_loss:.4f}  → {args.output_dir}/checkpoint-min_loss.pth")

        # Write final summary to result.txt
        _write_result_row(result_path, -1, {"accuracy": best_acc, "auc": best_auc, "real_accuracy": 0, "fake_accuracy": 0, "loss": min_test_loss}, tag="FINAL_BEST")
        _write_progress(
            progress_path,
            f"\nTraining finished in {elapsed}\n"
            f"Best AUC: {best_auc:.2f}%  Best Acc: {best_acc:.2f}%\n"
            + "=" * 70 + "\n",
        )

    if log_writer:
        log_writer.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    if not args.output_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"experiments/run_{args.model}_{timestamp}"

    if not args.log_dir:
        args.log_dir = args.output_dir

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    if misc.is_main_process() or not torch.distributed.is_initialized():
        config_path = Path(args.output_dir) / "config.json"
        with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=2)

    # Redirect stdout to log file (main process only)
    if os.environ.get("RANK", "0") == "0":
        log_file = open(os.path.join(args.output_dir, "log_detail.txt"), "a")
        sys.stdout = log_file
        sys.stderr = log_file

    main(args)
