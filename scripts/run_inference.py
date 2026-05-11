#!/usr/bin/env python3
"""
FSVFM frame-level inference on test set.

Saves:
  inference_results.csv   — fileid, probability, decision  (frame-level)
  video_results.csv       — video_id, gt_label, mean_prob, decision, n_frames
  inference_summary.txt   — metrics + video-level AUC/Acc

Launch (4 GPUs):
  CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node=4 --master_port=29520 \
      scripts/run_inference.py \
      --checkpoint experiments/.../checkpoint-best_auc.pth \
      --real_txt  /mnt/h200_dataset/saad/datasets/gend_unified/0_real_test.txt \
      --fake_txt  /mnt/h200_dataset/saad/datasets/gend_unified/1_fake_test.txt \
      --output_dir experiments/.../inference_epoch3 \
      --path_remap_from /data/saad/datasets/gend_unified \
      --path_remap_to   /mnt/h200_dataset/saad/datasets/gend_unified
"""

import argparse
import csv
import os
import sys
import json
import datetime
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fsvfm")))
import models_vit


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FrameDataset(Dataset):
    """Loads frames from two path-list txt files (real=0, fake=1)."""

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, real_txt, fake_txt, transform, path_remap_from="", path_remap_to=""):
        self.transform = transform
        self.samples = []   # (abs_path, gt_label)

        do_remap = bool(path_remap_from and path_remap_to)
        for txt, label in [(real_txt, 0), (fake_txt, 1)]:
            with open(txt) as f:
                paths = [ln.strip() for ln in f if ln.strip()]
            if do_remap:
                paths = [p.replace(path_remap_from, path_remap_to, 1) for p in paths]
            for p in paths:
                self.samples.append((p, label))

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
            # Return a black frame on read error — flagged by path
            dummy = torch.zeros(3, 224, 224)
            return dummy, label, path + "__CORRUPT__"


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
VGGFACE2_MEAN = [0.5482207536697388,  0.42340534925460815, 0.3654651641845703]
VGGFACE2_STD  = [0.2789176106452942,  0.2438540756702423,  0.23493893444538116]


def build_val_transform(input_size=224, use_vggface_norm=False):
    crop_pct = 224 / 256
    size = int(input_size / crop_pct)
    mean = VGGFACE2_MEAN if use_vggface_norm else IMAGENET_MEAN
    std  = VGGFACE2_STD  if use_vggface_norm else IMAGENET_STD
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def video_id_from_path(path: str) -> str:
    """Extract video-level ID = parent directory name."""
    return Path(path).parent.name


def compute_metrics(labels, probs, threshold=0.5):
    preds = (np.array(probs) >= threshold).astype(int)
    labels = np.array(labels)
    acc = float(accuracy_score(labels, preds) * 100)
    try:
        auc = float(roc_auc_score(labels, probs) * 100)
    except Exception:
        auc = 0.0
    real_mask = labels == 0
    fake_mask = labels == 1
    real_acc = float((preds[real_mask] == 0).mean() * 100) if real_mask.sum() > 0 else 0.0
    fake_acc = float((preds[fake_mask] == 1).mean() * 100) if fake_mask.sum() > 0 else 0.0
    return {"accuracy": acc, "auc": auc, "real_accuracy": real_acc, "fake_accuracy": fake_acc}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",       required=True)
    parser.add_argument("--real_txt",         required=True)
    parser.add_argument("--fake_txt",         required=True)
    parser.add_argument("--output_dir",       required=True)
    parser.add_argument("--path_remap_from",  default="")
    parser.add_argument("--path_remap_to",    default="")
    parser.add_argument("--input_size",       default=224, type=int)
    parser.add_argument("--batch_size",       default=256, type=int)
    parser.add_argument("--num_workers",      default=8,   type=int)
    parser.add_argument("--threshold",        default=0.5, type=float)
    parser.add_argument("--model",            default="vit_large_patch16")
    parser.add_argument("--fake_class_idx",   default=1,   type=int,
                        help="Class index the model uses for FAKE (1=default, 0=baseline paper)")
    parser.add_argument("--use_vggface_norm", action="store_true", default=False,
                        help="Use VGGFace2 normalization instead of ImageNet (for paper baseline)")
    args = parser.parse_args()

    # DDP init
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ----- Model -----
    model = models_vit.__dict__[args.model](
        num_classes=2, drop_path_rate=0.0, global_pool=True,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_epoch = ckpt.get("epoch", "?")
    state = ckpt["model"]
    msg = model.load_state_dict(state, strict=True)
    model.to(device).eval()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    if is_main():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model loaded from epoch {ckpt_epoch+1} | {n_params/1e6:.1f}M params")
        print(f"best_auc={ckpt.get('best_auc',0):.2f}%  "
              f"best_acc={ckpt.get('best_acc',0):.2f}%  "
              f"min_loss={ckpt.get('min_test_loss',0):.4f}")

    # ----- Dataset -----
    transform = build_val_transform(args.input_size, args.use_vggface_norm)
    if is_main():
        norm_tag = "VGGFace2" if args.use_vggface_norm else "ImageNet"
        print(f"Normalization: {norm_tag}  |  fake_class_idx={args.fake_class_idx}")
    dataset = FrameDataset(
        args.real_txt, args.fake_txt, transform,
        args.path_remap_from, args.path_remap_to,
    )
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
    loader  = DataLoader(
        dataset, sampler=sampler,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, drop_last=False,
    )

    if is_main():
        print(f"Dataset: {len(dataset):,} frames  |  {len(loader):,} batches/rank")

    # ----- Inference -----
    local_paths, local_probs, local_labels = [], [], []

    pbar = tqdm(loader, desc=f"[rank {dist.get_rank()}] Inference",
                disable=not is_main(), dynamic_ncols=True)

    with torch.no_grad():
        for imgs, labels, paths in pbar:
            imgs = imgs.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                logits = model(imgs)
            probs = F.softmax(logits, dim=1)[:, args.fake_class_idx].cpu().numpy()
            local_paths.extend(paths)
            local_probs.extend(probs.tolist())
            local_labels.extend(labels.numpy().tolist())

    # ----- Gather across ranks -----
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, {
        "paths": local_paths, "probs": local_probs, "labels": local_labels
    })

    if not is_main():
        dist.destroy_process_group()
        return

    all_paths  = []
    all_probs  = []
    all_labels = []
    for g in gathered:
        all_paths.extend(g["paths"])
        all_probs.extend(g["probs"])
        all_labels.extend(g["labels"])

    # Remove corrupt frames
    clean = [(p, pr, l) for p, pr, l in zip(all_paths, all_probs, all_labels)
             if not p.endswith("__CORRUPT__")]
    all_paths, all_probs, all_labels = zip(*clean) if clean else ([], [], [])
    all_paths  = list(all_paths)
    all_probs  = list(all_probs)
    all_labels = list(all_labels)

    all_decisions = [1 if p >= args.threshold else 0 for p in all_probs]

    # ----- Frame-level CSV -----
    frame_csv = Path(args.output_dir) / "inference_results.csv"
    with open(frame_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fileid", "probability", "decision"])
        for path, prob, dec in zip(all_paths, all_probs, all_decisions):
            w.writerow([path, f"{prob:.6f}", dec])
    print(f"Frame CSV saved: {frame_csv}  ({len(all_paths):,} rows)")

    # ----- Video-level aggregation -----
    video_frames = defaultdict(lambda: {"probs": [], "labels": []})
    for path, prob, label in zip(all_paths, all_probs, all_labels):
        vid = video_id_from_path(path)
        video_frames[vid]["probs"].append(prob)
        video_frames[vid]["labels"].append(label)

    video_ids, video_gt, video_probs, video_preds, video_n = [], [], [], [], []
    for vid, data in sorted(video_frames.items()):
        mean_prob = float(np.mean(data["probs"]))
        gt = int(round(np.mean(data["labels"])))   # majority GT (should be uniform)
        decision = 1 if mean_prob >= args.threshold else 0
        video_ids.append(vid)
        video_gt.append(gt)
        video_probs.append(mean_prob)
        video_preds.append(decision)
        video_n.append(len(data["probs"]))

    video_csv = Path(args.output_dir) / "video_results.csv"
    with open(video_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "gt_label", "mean_probability", "decision", "n_frames"])
        for vid, gt, prob, dec, n in zip(video_ids, video_gt, video_probs, video_preds, video_n):
            w.writerow([vid, gt, f"{prob:.6f}", dec, n])
    print(f"Video CSV saved: {video_csv}  ({len(video_ids):,} videos)")

    # ----- Metrics -----
    frame_metrics = compute_metrics(all_labels, all_probs, args.threshold)
    video_metrics = compute_metrics(video_gt, video_probs, args.threshold)

    # ----- Summary -----
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_lines = [
        "=" * 70,
        f"FSVFM Inference Summary — {now}",
        f"Checkpoint : {args.checkpoint}",
        f"  Epoch    : {ckpt_epoch+1}/20  (best_auc={ckpt.get('best_auc',0):.2f}%)",
        f"Test set   : {len(all_paths):,} frames  |  {len(video_ids):,} videos",
        f"Threshold  : {args.threshold}",
        "-" * 70,
        "FRAME-LEVEL RESULTS:",
        f"  Accuracy      : {frame_metrics['accuracy']:.2f}%",
        f"  AUC           : {frame_metrics['auc']:.2f}%",
        f"  Real_Accuracy : {frame_metrics['real_accuracy']:.2f}%",
        f"  Fake_Accuracy : {frame_metrics['fake_accuracy']:.2f}%",
        "-" * 70,
        "VIDEO-LEVEL RESULTS (mean-probability aggregation):",
        f"  Accuracy      : {video_metrics['accuracy']:.2f}%",
        f"  AUC           : {video_metrics['auc']:.2f}%",
        f"  Real_Accuracy : {video_metrics['real_accuracy']:.2f}%",
        f"  Fake_Accuracy : {video_metrics['fake_accuracy']:.2f}%",
        "-" * 70,
        f"Outputs:",
        f"  {frame_csv}",
        f"  {video_csv}",
        "=" * 70,
    ]
    summary_text = "\n".join(summary_lines) + "\n"

    summary_path = Path(args.output_dir) / "inference_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text)

    print(summary_text)

    # ----- Append to result.txt -----
    result_txt = Path(args.output_dir).parent.parent / "result.txt"
    # Try to find result.txt one level up from output_dir (experiments/<run>/...)
    # output_dir is experiments/<run>/inference_epoch3
    candidate = Path(args.output_dir).parent / "result.txt"
    if candidate.exists():
        result_txt = candidate

    # Write to result.txt only once — use a sentinel file to prevent re-runs from duplicating
    sentinel = Path(args.output_dir) / ".result_written"
    if result_txt.exists() and not sentinel.exists():
        append_block = (
            "\n" + "=" * 70 + "\n"
            f"INFERENCE RESULTS — Epoch {ckpt_epoch+1} (checkpoint-best_auc.pth)\n"
            f"Run at: {now}\n"
            f"Test set: {len(all_paths):,} frames  |  {len(video_ids):,} videos  |  8 unseen generators\n"
            f"Threshold: {args.threshold}\n"
            "-" * 70 + "\n"
            "FRAME-LEVEL:\n"
            f"  Accuracy      : {frame_metrics['accuracy']:.2f}%\n"
            f"  AUC           : {frame_metrics['auc']:.2f}%\n"
            f"  Real_Accuracy : {frame_metrics['real_accuracy']:.2f}%\n"
            f"  Fake_Accuracy : {frame_metrics['fake_accuracy']:.2f}%\n"
            "-" * 70 + "\n"
            f"VIDEO-LEVEL ({len(video_ids):,} videos — mean-prob aggregation):\n"
            f"  Accuracy      : {video_metrics['accuracy']:.2f}%\n"
            f"  AUC           : {video_metrics['auc']:.2f}%\n"
            f"  Real_Accuracy : {video_metrics['real_accuracy']:.2f}%\n"
            f"  Fake_Accuracy : {video_metrics['fake_accuracy']:.2f}%\n"
            "-" * 70 + "\n"
            f"Output files:\n"
            f"  {Path(args.output_dir).name}/inference_results.csv  ({len(all_paths):,} rows)\n"
            f"  {Path(args.output_dir).name}/video_results.csv      ({len(video_ids):,} rows)\n"
            "=" * 70 + "\n"
        )
        with open(result_txt, "a") as f:
            f.write(append_block)
        sentinel.touch()
        print(f"Appended to: {result_txt}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
