#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# FSVFM Composite-Dataset Fine-tuning — ViT-L/16 on GPUs 4,5
#
# Usage:
#   bash scripts/train_fsvfm_composite.sh [dataset_root1] [dataset_root2] ...
#
# If no dataset roots are passed as args, edit TRAIN_ROOTS / VAL_ROOTS below.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO_ROOT/fsvfm/finetune/cross_dataset_DFD_and_DiFF/train_composite.py"
PYTHON="/data/awais/anaconda/envs/d3/bin/python"

# ---------------------------------------------------------------------------
# ---- EDIT THESE to point at your datasets ----
# Each --train_root and --val_root is a directory whose immediate
# subdirectories are named to indicate real/fake (e.g., "real", "fake",
# or any name — "real" substring → label 0, otherwise → label 1).
# You may pass the same root paths as both train and val roots or
# use different splits.
# ---------------------------------------------------------------------------
TRAIN_ROOTS=(
    # "/path/to/dataset1"
    # "/path/to/dataset2"
    "${@}"          # also accept CLI arguments
)
VAL_ROOTS=(
    # "/path/to/val_dataset"
    "${@}"
)

# If you have txt label files instead of folder-structure datasets, add:
# --train_label_file /path/to/train.txt
# --val_label_file   /path/to/val.txt

# Pre-trained checkpoint (FS-VFM ViT-L from GenD_NeSy or upstream)
PRETRAINED="/data/awais/projects/GenD_NeSy/weights/FS-VFM/FS-VFM-ViT-L.pth"

# Output
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="fsvfm_vitl_composite_${TIMESTAMP}"
OUTPUT_DIR="$REPO_ROOT/experiments/${EXPERIMENT_NAME}"
LOG_DIR="$REPO_ROOT/logs/${EXPERIMENT_NAME}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "============================================================"
echo " FSVFM Composite Training"
echo " Experiment : $EXPERIMENT_NAME"
echo " Output     : $OUTPUT_DIR"
echo " Logs       : $LOG_DIR"
echo " GPUs       : 4,5 (2 GPUs)"
echo "============================================================"

# Build --train_root args
TRAIN_ROOT_ARGS=()
for root in "${TRAIN_ROOTS[@]}"; do
    TRAIN_ROOT_ARGS+=(--train_root "$root")
done

VAL_ROOT_ARGS=()
for root in "${VAL_ROOTS[@]}"; do
    VAL_ROOT_ARGS+=(--val_root "$root")
done

CUDA_VISIBLE_DEVICES=4,5 \
OMP_NUM_THREADS=4 \
"$PYTHON" -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port=29510 \
    "$SCRIPT" \
    "${TRAIN_ROOT_ARGS[@]}" \
    "${VAL_ROOT_ARGS[@]}" \
    --finetune    "$PRETRAINED" \
    --model       vit_large_patch16 \
    --nb_classes  2 \
    --input_size  224 \
    --batch_size  32 \
    --accum_iter  1 \
    --epochs      50 \
    --warmup_epochs 5 \
    --blr         5e-5 \
    --layer_decay 0.90 \
    --weight_decay 0.01 \
    --drop_path   0.1 \
    --mixup       0.8 \
    --cutmix      1.0 \
    --reprob      0.25 \
    --apply_simple_augment \
    --normalize_from_IMN \
    --balance \
    --dist_eval \
    --num_workers 10 \
    --save_every  1 \
    --output_dir  "$OUTPUT_DIR" \
    --log_dir     "$LOG_DIR" \
    2>&1 | tee "$LOG_DIR/train_stdout.log"

echo "Training complete. Checkpoints at: $OUTPUT_DIR"
