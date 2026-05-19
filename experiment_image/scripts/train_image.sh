#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Launch image-robust FSVFM training (GPUs 5, 6, 7)
#
# Usage:
#   bash experiment_image/scripts/train_image.sh                     # new run
#   bash experiment_image/scripts/train_image.sh --output_dir <dir>  # fixed dir
#   bash experiment_image/scripts/train_image.sh --resume <ckpt.pth> # resume
#
# Called by watchdog_image.sh — all path args should be absolute.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="/data/awais/anaconda/envs/d3/bin/python"
TRAIN_SCRIPT="$REPO_ROOT/experiment_image/train_image_robust.py"

# Pretrained checkpoint — composite model epoch 3 (best_auc)
FINETUNE_CKPT="$REPO_ROOT/experiments/fsvfm_vitl_composite_20260508_160444/FSFM_best_checkpoint.pth"

# Dataset paths (h200 CIFS mount)
DATASET_ROOT="/mnt/h200_dataset/saad/datasets/gend_unified"
REAL_TRAIN="$DATASET_ROOT/0_real_train.txt"
FAKE_TRAIN="$DATASET_ROOT/1_fake_train.txt"
REAL_TEST="$DATASET_ROOT/0_real_test.txt"
FAKE_TEST="$DATASET_ROOT/1_fake_test.txt"
REMAP_FROM="/data/saad/datasets/gend_unified"
REMAP_TO="$DATASET_ROOT"

# Parse pass-through args
OUTPUT_DIR=""
RESUME_ARG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --resume)     RESUME_ARG="--resume $2"; shift 2 ;;
        *) shift ;;
    esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="$REPO_ROOT/experiment_image/runs/img_robust_vitl_${TIMESTAMP}"
fi

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo " Image-Robust FSVFM Training"
echo " GPUs       : 5, 6, 7"
echo " Finetune   : $FINETUNE_CKPT"
echo " Output     : $OUTPUT_DIR"
echo " Compression aug: ENABLED"
echo " Batch/GPU  : 128 (64 real + 64 fake)"
echo " Epochs     : 20  Warmup: 2"
echo "============================================================"

CUDA_VISIBLE_DEVICES=5,6,7 \
OMP_NUM_THREADS=4 \
"$PYTHON" -m torch.distributed.run \
    --nproc_per_node=3 \
    --master_port=29521 \
    "$TRAIN_SCRIPT" \
    --finetune         "$FINETUNE_CKPT" \
    --real_train_txt   "$REAL_TRAIN" \
    --fake_train_txt   "$FAKE_TRAIN" \
    --real_test_txt    "$REAL_TEST" \
    --fake_test_txt    "$FAKE_TEST" \
    --txt_path_remap_from "$REMAP_FROM" \
    --txt_path_remap_to   "$REMAP_TO" \
    --output_dir       "$OUTPUT_DIR" \
    --batch_size       128 \
    --epochs           20 \
    --warmup_epochs    2 \
    --blr              5e-5 \
    --layer_decay      0.90 \
    --weight_decay     0.01 \
    --drop_path        0.1 \
    --mixup            0.8 \
    --cutmix           1.0 \
    --model            vit_large_patch16 \
    --num_workers      8 \
    --use_compression_aug \
    --aug_prob_scale   1.0 \
    --balanced_batch \
    $RESUME_ARG \
    2>&1 | tee -a "$OUTPUT_DIR/train_stdout.log"

echo ""
echo "Done. Key outputs:"
echo "  $OUTPUT_DIR/result.txt"
echo "  $OUTPUT_DIR/progress.txt"
echo "  $OUTPUT_DIR/checkpoint-best_auc.pth"
echo "  $OUTPUT_DIR/checkpoint-best_acc.pth"
