#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Robust FSVFM fine-tuning on composite deepfake datasets
# Fixes: timm 1.0.20 compat, DDP via torchrun, crash recovery, best+latest ckpt saving
#
# Launch:  CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 train_composite.py [args]
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
from util.composite_dataset import CompositeDeepfakeDataset, build_composite_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import models_vit

# Patch engine_finetune path
sys.path.insert(0, os.path.dirname(__file__))
from engine_finetune import train_one_epoch, evaluate

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
    p.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU")
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
    roots = args.train_root if is_train else args.val_root
    label_files = args.train_label_file if is_train else args.val_label_file

    if not roots:
        raise ValueError("Provide at least one --train_root / --val_root.")

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

def save_checkpoint(args, epoch, model_without_ddp, optimizer, loss_scaler, tag):
    ckpt_dir = Path(args.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"checkpoint-{tag}.pth"
    payload = {
        "model": model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "scaler": loss_scaler.state_dict(),
        "args": vars(args),
    }
    misc.save_on_master(payload, path)
    if misc.is_main_process():
        log.info(f"Checkpoint saved: {path}")
    return str(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    dataset_val = build_datasets(args, is_train=False)

    if misc.is_main_process():
        log.info(f"Train samples: {len(dataset_train)}   Val samples: {len(dataset_val)}")

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    sampler_val = (
        torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        if args.dist_eval
        else torch.utils.data.SequentialSampler(dataset_val)
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
    loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
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

    # Set crash-save path so signal handler can report it
    if args.output_dir:
        _CRASH_SAVE_PATH = str(Path(args.output_dir) / "checkpoint-latest.pth")

    # ----- Dry run -----
    if args.dry_run:
        log.info("=== DRY RUN: one forward+backward pass ===")
        model.train()
        batch = next(iter(loader_train))
        imgs, labels = batch[0].to(device), batch[1].to(device)
        with torch.cuda.amp.autocast():
            out = model(imgs)
            loss = criterion(out, labels)
        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=True)
        log.info(f"Dry-run loss: {loss.item():.4f}  — PASSED")
        if log_writer:
            log_writer.close()
        return

    # ----- Training loop -----
    log.info(f"Start training for {args.epochs} epochs from epoch {args.start_epoch}")
    start_time = time.time()
    best_auc = 0.0
    min_val_loss = float("inf")

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer, args=args,
        )

        val_stats = evaluate(loader_val, model, device)

        if misc.is_main_process():
            auc = val_stats.get("auc", 0.0)
            vloss = val_stats.get("loss", float("inf"))
            log.info(f"Epoch {epoch} | val_auc={auc:.3f}% | val_loss={vloss:.4f} | best_auc={best_auc:.3f}%")

        # Always save latest checkpoint for crash recovery
        if args.output_dir and (epoch % args.save_every == 0 or epoch + 1 == args.epochs):
            save_checkpoint(args, epoch, model_without_ddp, optimizer, loss_scaler, "latest")

        # Save best-AUC checkpoint
        if val_stats.get("auc", 0) > best_auc:
            best_auc = val_stats["auc"]
            save_checkpoint(args, epoch, model_without_ddp, optimizer, loss_scaler, "best_auc")
            if misc.is_main_process():
                log.info(f"New best AUC checkpoint at epoch {epoch}: {best_auc:.3f}%")

        # Save best-loss checkpoint
        if val_stats.get("loss", float("inf")) < min_val_loss:
            min_val_loss = val_stats["loss"]
            save_checkpoint(args, epoch, model_without_ddp, optimizer, loss_scaler, "min_val_loss")
            if misc.is_main_process():
                log.info(f"New min val-loss checkpoint at epoch {epoch}: {min_val_loss:.4f}")

        # TensorBoard
        if log_writer is not None:
            log_writer.add_scalar("perf/val_auc", val_stats.get("auc", 0), epoch)
            log_writer.add_scalar("perf/val_loss", val_stats.get("loss", 0), epoch)
            log_writer.add_scalar("perf/train_loss", train_stats.get("loss", 0), epoch)

        # JSON log
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_stats.items()},
            "epoch": epoch,
            "best_auc": best_auc,
            "min_val_loss": min_val_loss,
        }
        if args.output_dir and misc.is_main_process():
            if log_writer:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "train_log.txt"), "a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    if misc.is_main_process():
        log.info(f"Training complete in {elapsed}")
        log.info(f"Best AUC:     {best_auc:.3f}%  → {args.output_dir}/checkpoint-best_auc.pth")
        log.info(f"Min val loss: {min_val_loss:.4f}  → {args.output_dir}/checkpoint-min_val_loss.pth")

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
