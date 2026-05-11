#!/usr/bin/env python3
"""
Build summary_results.txt comparing composite (FSFM) vs baseline model
across DF40test and FFIW datasets.

Frame-level and video-level results for both models.

Usage:
  python scripts/make_summary.py \
      --composite_csv  .../inference_results.csv \
      --baseline_csv   .../baseline/inference_results.csv \
      --output_dir     .../inference_epoch3_best_auc
"""
import argparse
import csv
import datetime
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gt_from_path(path: str) -> int:
    """Derive ground-truth label from path: /0_real/ → 0, /1_fake/ → 1."""
    if "/0_real/" in path:
        return 0
    if "/1_fake/" in path:
        return 1
    return -1


def dataset_from_path(path: str) -> str:
    """Tag each frame with a dataset name."""
    parts = path.split("/")
    for p in parts:
        if p.startswith("FFIW"):
            return "FFIW"
        if p.startswith("DF40test"):
            return "DF40test"
        if "ReWIND" in p or "rewind" in p.lower():
            return "ReWIND"
        if p.upper().startswith("DFDC"):
            return "DFDC"
    return "OTHER"


def video_id_from_path(path: str) -> str:
    return Path(path).parent.name


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

    return {"accuracy": acc, "auc": auc, "real_accuracy": real_acc, "fake_accuracy": fake_acc,
            "n_real": int(real_mask.sum()), "n_fake": int(fake_mask.sum()),
            "n_total": len(labels)}


def video_level_metrics(paths, probs, labels, threshold=0.5):
    """Aggregate frame probs per video (parent dir), then compute metrics."""
    vid_data = defaultdict(lambda: {"probs": [], "labels": []})
    for path, prob, label in zip(paths, probs, labels):
        vid = video_id_from_path(path)
        vid_data[vid]["probs"].append(prob)
        vid_data[vid]["labels"].append(label)

    v_labels, v_probs = [], []
    for vid, d in vid_data.items():
        v_probs.append(float(np.mean(d["probs"])))
        v_labels.append(int(round(np.mean(d["labels"]))))

    m = compute_metrics(v_labels, v_probs, threshold)
    m["n_videos"] = len(vid_data)
    return m


def load_csv(csv_path: str):
    """Load inference CSV → {path: (prob, decision)}. Skip CORRUPT rows."""
    data = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["probability"] == "CORRUPT":
                continue
            try:
                data[row["fileid"]] = float(row["probability"])
            except ValueError:
                pass
    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--composite_csv",  required=True)
    parser.add_argument("--baseline_csv",   required=True)
    parser.add_argument("--output_dir",     required=True)
    parser.add_argument("--threshold",      default=0.5, type=float)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading CSVs...")
    composite_data = load_csv(args.composite_csv)
    baseline_data  = load_csv(args.baseline_csv)
    print(f"  Composite : {len(composite_data):,} frames")
    print(f"  Baseline  : {len(baseline_data):,}  frames")

    # Use intersection of paths so both models are evaluated on identical frames
    common_paths = sorted(set(composite_data) & set(baseline_data))
    print(f"  Common    : {len(common_paths):,} frames")

    # Build per-dataset buckets
    datasets = {"DF40test": [], "FFIW": [], "ReWIND": [], "DFDC": []}
    for path in common_paths:
        ds = dataset_from_path(path)
        if ds in datasets:
            gt = gt_from_path(path)
            if gt == -1:
                continue
            datasets[ds].append({
                "path": path,
                "gt": gt,
                "composite_prob": composite_data[path],
                "baseline_prob":  baseline_data[path],
            })

    for ds, rows in datasets.items():
        print(f"  {ds}: {len(rows):,} frames  "
              f"(real={sum(1 for r in rows if r['gt']==0):,}  "
              f"fake={sum(1 for r in rows if r['gt']==1):,})")

    # -----------------------------------------------------------------------
    # Per-dataset: save individual CSVs and compute metrics
    # -----------------------------------------------------------------------
    results = {}    # results[dataset][model] = {frame: ..., video: ...}

    MODELS = {
        "FSFM_Composite": "composite_prob",
        "FSFM_Baseline":  "baseline_prob",
    }

    for ds, rows in datasets.items():
        results[ds] = {}
        paths  = [r["path"]  for r in rows]
        labels = [r["gt"]    for r in rows]

        for model_name, prob_key in MODELS.items():
            probs = [r[prob_key] for r in rows]
            decisions = [1 if p >= args.threshold else 0 for p in probs]

            # Save CSV: FSFM_{dataset}_{model}.csv
            safe_ds = ds.replace(" ", "_")
            csv_name = f"FSFM_{safe_ds}_{model_name}.csv"
            csv_path = Path(args.output_dir) / csv_name
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["fileid", "probability", "decision"])
                for p, pr, d in zip(paths, probs, decisions):
                    w.writerow([p, f"{pr:.6f}", d])
            print(f"  Saved: {csv_name}  ({len(rows):,} rows)")

            frame_m = compute_metrics(labels, probs, args.threshold)
            video_m = video_level_metrics(paths, probs, labels, args.threshold)
            results[ds][model_name] = {"frame": frame_m, "video": video_m}

    # -----------------------------------------------------------------------
    # Build summary_results.txt
    # -----------------------------------------------------------------------
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    COMPOSITE_CKPT = "/data/awais/projects/FSVFM/experiments/fsvfm_vitl_composite_20260508_160444/FSFM_best_checkpoint.pth"
    BASELINE_CKPT  = "/data/awais/projects/GenD_NeSy/weights/FS-VFM/FS-VFM-ViT-L.pth"
    DATASET_ROOT   = "/mnt/h200_dataset/saad/datasets/gend_unified"

    lines = []
    w = lines.append

    w("=" * 78)
    w("FSVFM MODEL COMPARISON — Frame-level & Video-level Results")
    w(f"Generated : {now}")
    w("=" * 78)
    w("")
    w("MODELS")
    w("------")
    w(f"  FSFM_Composite : {COMPOSITE_CKPT}")
    w(f"    (Epoch 3, fine-tuned on gend_unified composite dataset)")
    w(f"  FSFM_Baseline  : {BASELINE_CKPT}")
    w(f"    (Paper's provided ViT-L/16 checkpoint, trained on DfD/DiFF)")
    w("")
    w("DATASET PATHS")
    w("-------------")
    w(f"  Test set root  : {DATASET_ROOT}")
    w(f"  Real txt       : {DATASET_ROOT}/0_real_test.txt  (108,507 frames, incl. ReWIND + DFDC)")
    w(f"  Fake txt       : {DATASET_ROOT}/1_fake_test.txt  (105,893 frames, incl. ReWIND + DFDC)")
    w(f"  Composite CSV  : {args.composite_csv}")
    w(f"  Baseline CSV   : {args.baseline_csv}")
    w("")
    w("DATASETS EVALUATED")
    w("------------------")
    for ds, rows in datasets.items():
        n_real = sum(1 for r in rows if r["gt"] == 0)
        n_fake = sum(1 for r in rows if r["gt"] == 1)
        w(f"  {ds:<12}: {len(rows):>7,} frames  (real={n_real:,}  fake={n_fake:,})")
    w("")

    # Per-dataset detailed results
    for ds, rows in datasets.items():
        n_real = sum(1 for r in rows if r["gt"] == 0)
        n_fake = sum(1 for r in rows if r["gt"] == 1)
        vid_count = len(set(video_id_from_path(r["path"]) for r in rows))

        w("=" * 78)
        w(f"DATASET: {ds}")
        w(f"  Frames : {len(rows):,}  (real={n_real:,}  fake={n_fake:,})")
        w(f"  Videos : {vid_count:,}")
        w("=" * 78)
        w("")

        # Frame-level table
        w("  FRAME-LEVEL RESULTS")
        w("  " + "-" * 60)
        w(f"  {'Model':<22}  {'Accuracy':>9}  {'AUC':>8}  {'Real_Acc':>9}  {'Fake_Acc':>9}")
        w("  " + "-" * 60)
        for model_name in MODELS:
            fm = results[ds][model_name]["frame"]
            w(f"  {model_name:<22}  {fm['accuracy']:>8.2f}%  {fm['auc']:>7.2f}%  "
              f"{fm['real_accuracy']:>8.2f}%  {fm['fake_accuracy']:>8.2f}%")
        w("  " + "-" * 60)
        # Delta row
        comp_f = results[ds]["FSFM_Composite"]["frame"]
        base_f = results[ds]["FSFM_Baseline"]["frame"]
        w(f"  {'Delta (Comp - Base)':<22}  "
          f"{comp_f['accuracy']-base_f['accuracy']:>+8.2f}%  "
          f"{comp_f['auc']-base_f['auc']:>+7.2f}%  "
          f"{comp_f['real_accuracy']-base_f['real_accuracy']:>+8.2f}%  "
          f"{comp_f['fake_accuracy']-base_f['fake_accuracy']:>+8.2f}%")
        w("")

        # Video-level table
        w("  VIDEO-LEVEL RESULTS  (mean-probability per clip)")
        w("  " + "-" * 60)
        w(f"  {'Model':<22}  {'Accuracy':>9}  {'AUC':>8}  {'Real_Acc':>9}  {'Fake_Acc':>9}")
        w("  " + "-" * 60)
        for model_name in MODELS:
            vm = results[ds][model_name]["video"]
            w(f"  {model_name:<22}  {vm['accuracy']:>8.2f}%  {vm['auc']:>7.2f}%  "
              f"{vm['real_accuracy']:>8.2f}%  {vm['fake_accuracy']:>8.2f}%")
        w("  " + "-" * 60)
        comp_v = results[ds]["FSFM_Composite"]["video"]
        base_v = results[ds]["FSFM_Baseline"]["video"]
        w(f"  {'Delta (Comp - Base)':<22}  "
          f"{comp_v['accuracy']-base_v['accuracy']:>+8.2f}%  "
          f"{comp_v['auc']-base_v['auc']:>+7.2f}%  "
          f"{comp_v['real_accuracy']-base_v['real_accuracy']:>+8.2f}%  "
          f"{comp_v['fake_accuracy']-base_v['fake_accuracy']:>+8.2f}%")
        w("")

    # -----------------------------------------------------------------------
    # Average across all datasets
    # -----------------------------------------------------------------------
    w("=" * 78)
    w("AVERAGE ACROSS ALL DATASETS")
    w("=" * 78)
    w("")

    for level in ["frame", "video"]:
        level_label = "FRAME-LEVEL" if level == "frame" else "VIDEO-LEVEL"
        w(f"  {level_label} AVERAGES")
        w("  " + "-" * 60)
        w(f"  {'Model':<22}  {'Accuracy':>9}  {'AUC':>8}  {'Real_Acc':>9}  {'Fake_Acc':>9}")
        w("  " + "-" * 60)
        for model_name in MODELS:
            avg_acc  = np.mean([results[ds][model_name][level]["accuracy"]      for ds in datasets])
            avg_auc  = np.mean([results[ds][model_name][level]["auc"]           for ds in datasets])
            avg_racc = np.mean([results[ds][model_name][level]["real_accuracy"] for ds in datasets])
            avg_facc = np.mean([results[ds][model_name][level]["fake_accuracy"] for ds in datasets])
            w(f"  {model_name:<22}  {avg_acc:>8.2f}%  {avg_auc:>7.2f}%  "
              f"{avg_racc:>8.2f}%  {avg_facc:>8.2f}%")
        w("  " + "-" * 60)
        avg_d_acc  = np.mean([results[ds]["FSFM_Composite"][level]["accuracy"]      - results[ds]["FSFM_Baseline"][level]["accuracy"]      for ds in datasets])
        avg_d_auc  = np.mean([results[ds]["FSFM_Composite"][level]["auc"]           - results[ds]["FSFM_Baseline"][level]["auc"]           for ds in datasets])
        avg_d_racc = np.mean([results[ds]["FSFM_Composite"][level]["real_accuracy"] - results[ds]["FSFM_Baseline"][level]["real_accuracy"] for ds in datasets])
        avg_d_facc = np.mean([results[ds]["FSFM_Composite"][level]["fake_accuracy"] - results[ds]["FSFM_Baseline"][level]["fake_accuracy"] for ds in datasets])
        w(f"  {'Avg Delta (C - B)':<22}  {avg_d_acc:>+8.2f}%  {avg_d_auc:>+7.2f}%  "
          f"{avg_d_racc:>+8.2f}%  {avg_d_facc:>+8.2f}%")
        w("")

    # -----------------------------------------------------------------------
    # Output CSV list
    # -----------------------------------------------------------------------
    w("=" * 78)
    w("OUTPUT FILES")
    w("=" * 78)
    for ds in datasets:
        for model_name in MODELS:
            safe_ds = ds.replace(" ", "_")
            w(f"  FSFM_{safe_ds}_{model_name}.csv")
    w(f"  summary_results.txt  (this file)")
    w("=" * 78)

    summary_path = Path(args.output_dir) / "summary_results.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nSummary saved: {summary_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
