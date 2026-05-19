#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frame-level inference for image-robust FSVFM model.

Supports three dataset modes:
  1. --real_txt / --fake_txt  — path-list files (same as run_inference.py)
  2. --dataset_root           — folder with 0_real / 1_fake anywhere in path tree
  3. --aigi_root              — AIGI-style: per-generator subdirs each with 0_real/1_fake

Output per run:
  inference_results.csv   — fileid, probability, decision
  inference_summary.txt   — metrics + dataset info

Launch (3 GPUs):
  CUDA_VISIBLE_DEVICES=5,6,7 torchrun --nproc_per_node=3 --master_port=29522 \\
      experiment_image/eval_image.py \\
      --checkpoint  experiment_image/runs/.../checkpoint-best_auc.pth \\
      --dataset_root /mnt/h200_dataset/saad/datasets/images/image_eval24 \\
      --dataset_name image_eval24 \\
      --output_dir   experiment_image/runs/.../eval_image_eval24

Single-GPU (no torchrun):
  CUDA_VISIBLE_DEVICES=5 python experiment_image/eval_image.py \\
      --checkpoint ... --dataset_root ... --output_dir ...
"""

import argparse
import csv
import datetime
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

ImageFile.LOAD_TRUNCATED_IMAGES = True

FSVFM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(FSVFM_ROOT, "fsvfm"))
import models_vit

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_EXTS      = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Label detection from path
# ---------------------------------------------------------------------------

def label_from_path(path: str) -> int:
    """
    Infer ground-truth label from path components.
    Returns 0 (real), 1 (fake), or -1 (unknown).
    """
    parts = Path(path).parts
    for part in parts:
        p = part.lower()
        if p == "0_real" or p in {"real", "original", "youtube", "actors", "raw"}:
            return 0
        if p == "1_fake" or p in {"fake", "manipulated", "tampered", "forgery"}:
            return 1
    return -1


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ImageEvalDataset(Dataset):
    """Loads images from a list of (path, label) pairs."""

    def __init__(self, samples, transform):
        self.samples   = samples   # list of (abs_path, label)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label, path
        except Exception:
            return torch.zeros(3, 224, 224), label, path + "__CORRUPT__"


def _val_transform(input_size: int = 224) -> transforms.Compose:
    crop_pct = 224 / 256
    size = int(input_size / crop_pct)
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _scan_folder(root: str):
    """Recursively collect (path, label) from a 0_real / 1_fake folder tree."""
    samples = []
    for p in sorted(Path(root).rglob("*")):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        lbl = label_from_path(str(p))
        if lbl == -1:
            continue
        samples.append((str(p), lbl))
    return samples


def _load_txt_pair(real_txt: str, fake_txt: str,
                   remap_from: str = "", remap_to: str = ""):
    samples = []
    for txt, label in [(real_txt, 0), (fake_txt, 1)]:
        with open(txt) as f:
            paths = [ln.strip() for ln in f if ln.strip()]
        if remap_from and remap_to:
            paths = [p.replace(remap_from, remap_to, 1) for p in paths]
        for p in paths:
            samples.append((p, label))
    return samples


def build_eval_samples(args) -> list:
    """Return list of (path, label) based on the provided args."""
    if args.real_txt and args.fake_txt:
        return _load_txt_pair(
            args.real_txt, args.fake_txt,
            args.path_remap_from, args.path_remap_to,
        )
    if args.dataset_root:
        return _scan_folder(args.dataset_root)
    raise ValueError(
        "Provide --real_txt + --fake_txt  OR  --dataset_root"
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(labels, probs, threshold=0.5):
    labels = np.array(labels)
    probs  = np.array(probs)
    preds  = (probs >= threshold).astype(int)
    acc = float(accuracy_score(labels, preds) * 100)
    try:
        auc = float(roc_auc_score(labels, probs) * 100)
    except Exception:
        auc = 0.0
    real_mask = labels == 0
    fake_mask = labels == 1
    real_acc = float((preds[real_mask] == 0).mean() * 100) if real_mask.sum() > 0 else 0.0
    fake_acc = float((preds[fake_mask] == 1).mean() * 100) if fake_mask.sum() > 0 else 0.0
    return {
        "accuracy": acc, "auc": auc,
        "real_accuracy": real_acc, "fake_accuracy": fake_acc,
        "n_real": int(real_mask.sum()), "n_fake": int(fake_mask.sum()),
        "n_total": len(labels),
    }


def video_level_metrics(paths, probs, labels, threshold=0.5):
    """Aggregate frame probs per parent-dir video ID, compute metrics."""
    vid_data = defaultdict(lambda: {"probs": [], "labels": []})
    for path, prob, label in zip(paths, probs, labels):
        vid = Path(path).parent.name
        vid_data[vid]["probs"].append(prob)
        vid_data[vid]["labels"].append(label)

    v_probs, v_labels = [], []
    for d in vid_data.values():
        v_probs.append(float(np.mean(d["probs"])))
        v_labels.append(int(round(np.mean(d["labels"]))))

    m = compute_metrics(v_labels, v_probs, threshold)
    m["n_videos"] = len(vid_data)
    return m


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",       required=True)
    parser.add_argument("--real_txt",         default="")
    parser.add_argument("--fake_txt",         default="")
    parser.add_argument("--dataset_root",     default="",
                        help="Root with 0_real/1_fake folder structure (recursive).")
    parser.add_argument("--dataset_name",     default="",
                        help="Dataset tag for output files (inferred from root if blank).")
    parser.add_argument("--output_dir",       required=True)
    parser.add_argument("--path_remap_from",  default="")
    parser.add_argument("--path_remap_to",    default="")
    parser.add_argument("--input_size",       default=224, type=int)
    parser.add_argument("--batch_size",       default=256, type=int)
    parser.add_argument("--num_workers",      default=8,   type=int)
    parser.add_argument("--threshold",        default=0.5, type=float)
    parser.add_argument("--model",            default="vit_large_patch16")
    parser.add_argument("--fake_class_idx",   default=1,   type=int,
                        help="Softmax index for FAKE probability (1=composite, 0=baseline).")
    args = parser.parse_args()

    # DDP init (works with torchrun or single-process)
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    dataset_name = args.dataset_name or (
        Path(args.dataset_root).name if args.dataset_root else
        Path(args.real_txt).stem.replace("0_real_", "")
    )

    # ----- Model -----
    model = models_vit.__dict__[args.model](
        num_classes=2, drop_path_rate=0.0, global_pool=True,
    )
    ckpt      = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_epoch = ckpt.get("epoch", "?")
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()

    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    if is_main():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {args.model}  ({n_params/1e6:.1f}M params)")
        print(f"Checkpoint epoch: {ckpt_epoch+1 if isinstance(ckpt_epoch, int) else ckpt_epoch}")
        print(f"best_auc={ckpt.get('best_auc',0):.2f}%  "
              f"best_acc={ckpt.get('best_acc',0):.2f}%")

    # ----- Dataset -----
    if is_main():
        print(f"\nBuilding dataset: {dataset_name}")
    all_samples = build_eval_samples(args)

    if is_main():
        n_real = sum(1 for _, l in all_samples if l == 0)
        n_fake = sum(1 for _, l in all_samples if l == 1)
        print(f"  Samples: {len(all_samples):,}  (real={n_real:,}  fake={n_fake:,})")

    transform = _val_transform(args.input_size)
    dataset   = ImageEvalDataset(all_samples, transform)

    if dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
        loader  = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size,
                             num_workers=args.num_workers, pin_memory=True, drop_last=False)
    else:
        loader = DataLoader(dataset, batch_size=args.batch_size,
                            num_workers=args.num_workers, pin_memory=True,
                            shuffle=False, drop_last=False)

    # ----- Inference -----
    local_paths, local_probs, local_labels = [], [], []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Inference", disable=not is_main(), dynamic_ncols=True)
        for imgs, labels, paths in pbar:
            imgs = imgs.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                logits = model(imgs)
            probs = F.softmax(logits, dim=1)[:, args.fake_class_idx].cpu().numpy()
            local_paths.extend(paths)
            local_probs.extend(probs.tolist())
            local_labels.extend(labels.numpy().tolist())

    # ----- Gather across ranks -----
    if dist.is_initialized():
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, {
            "paths": local_paths, "probs": local_probs, "labels": local_labels,
        })
        if not is_main():
            dist.destroy_process_group()
            return
        all_paths, all_probs, all_labels = [], [], []
        for g in gathered:
            all_paths.extend(g["paths"])
            all_probs.extend(g["probs"])
            all_labels.extend(g["labels"])
    else:
        all_paths, all_probs, all_labels = local_paths, local_probs, local_labels

    # Remove corrupt frames
    clean = [(p, pr, l) for p, pr, l in zip(all_paths, all_probs, all_labels)
             if not p.endswith("__CORRUPT__")]
    if not clean:
        print("ERROR: No valid samples after corruption filter.")
        return
    all_paths, all_probs, all_labels = zip(*clean)
    all_paths  = list(all_paths)
    all_probs  = list(all_probs)
    all_labels = list(all_labels)

    # Deduplicate (DDP may give duplicate entries at epoch boundaries)
    seen = {}
    for p, pr, l in zip(all_paths, all_probs, all_labels):
        if p not in seen:
            seen[p] = (pr, l)
    all_paths  = list(seen.keys())
    all_probs  = [seen[p][0] for p in all_paths]
    all_labels = [seen[p][1] for p in all_paths]

    decisions = [1 if pr >= args.threshold else 0 for pr in all_probs]

    # ----- Save inference_results.csv -----
    csv_path = Path(args.output_dir) / "inference_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fileid", "probability", "decision"])
        for p, pr, d in zip(all_paths, all_probs, decisions):
            w.writerow([p, f"{pr:.6f}", d])
    print(f"\nFrame CSV: {csv_path}  ({len(all_paths):,} rows)")

    # ----- Metrics -----
    frame_m = compute_metrics(all_labels, all_probs, args.threshold)
    video_m = video_level_metrics(all_paths, all_probs, all_labels, args.threshold)

    print(
        f"\nFRAME-LEVEL  Acc={frame_m['accuracy']:.2f}%  AUC={frame_m['auc']:.2f}%  "
        f"Real={frame_m['real_accuracy']:.2f}%  Fake={frame_m['fake_accuracy']:.2f}%"
    )
    print(
        f"VIDEO-LEVEL  Acc={video_m['accuracy']:.2f}%  AUC={video_m['auc']:.2f}%  "
        f"Real={video_m['real_accuracy']:.2f}%  Fake={video_m['fake_accuracy']:.2f}%  "
        f"(n_videos={video_m['n_videos']})"
    )

    # ----- Summary -----
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_lines = [
        "=" * 70,
        f"FSVFM Image Evaluation Summary — {now}",
        f"Dataset    : {dataset_name}",
        f"Checkpoint : {args.checkpoint}",
        f"  Epoch    : {ckpt_epoch+1 if isinstance(ckpt_epoch, int) else ckpt_epoch}  "
        f"(best_auc={ckpt.get('best_auc',0):.2f}%)",
        f"Samples    : {len(all_paths):,}  "
        f"real={frame_m['n_real']:,}  fake={frame_m['n_fake']:,}",
        f"Threshold  : {args.threshold}",
        "-" * 70,
        "FRAME-LEVEL RESULTS:",
        f"  Accuracy      : {frame_m['accuracy']:.2f}%",
        f"  AUC           : {frame_m['auc']:.2f}%",
        f"  Real_Accuracy : {frame_m['real_accuracy']:.2f}%",
        f"  Fake_Accuracy : {frame_m['fake_accuracy']:.2f}%",
        "-" * 70,
        f"IMAGE/VIDEO-LEVEL RESULTS (mean-prob per parent-dir clip):",
        f"  Videos/clips  : {video_m['n_videos']:,}",
        f"  Accuracy      : {video_m['accuracy']:.2f}%",
        f"  AUC           : {video_m['auc']:.2f}%",
        f"  Real_Accuracy : {video_m['real_accuracy']:.2f}%",
        f"  Fake_Accuracy : {video_m['fake_accuracy']:.2f}%",
        "-" * 70,
        f"Output: {csv_path}",
        "=" * 70,
    ]
    summary_text = "\n".join(summary_lines) + "\n"

    summary_path = Path(args.output_dir) / "inference_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"\nSummary: {summary_path}")
    print(summary_text)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
