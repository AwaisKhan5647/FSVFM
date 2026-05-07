#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Pre-flight dry-run: verifies dataloaders, GPU allocation, checkpoint load,
# DDP init, and one-batch forward/backward before full training.
#
# Usage: bash scripts/dry_run.sh <train_root> [val_root]
#   If val_root is omitted, train_root is used for validation too.
# ---------------------------------------------------------------------------
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <train_root> [val_root]"
    exit 1
fi

TRAIN_ROOT="$1"
VAL_ROOT="${2:-$TRAIN_ROOT}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO_ROOT/fsvfm/finetune/cross_dataset_DFD_and_DiFF/train_composite.py"
PYTHON="/data/awais/anaconda/envs/d3/bin/python"
PRETRAINED="/data/awais/projects/GenD_NeSy/weights/FS-VFM/FS-VFM-ViT-L.pth"
OUTPUT_DIR="$REPO_ROOT/experiments/dry_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo " FSVFM Dry-Run Verification"
echo " Train root: $TRAIN_ROOT"
echo " Val root  : $VAL_ROOT"
echo " Pretrained: $PRETRAINED"
echo " Output    : $OUTPUT_DIR"
echo " GPUs      : 4,5"
echo "============================================================"

CUDA_VISIBLE_DEVICES=4,5 \
OMP_NUM_THREADS=4 \
"$PYTHON" -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port=29512 \
    "$SCRIPT" \
    --train_root  "$TRAIN_ROOT" \
    --val_root    "$VAL_ROOT" \
    --finetune    "$PRETRAINED" \
    --model       vit_large_patch16 \
    --nb_classes  2 \
    --input_size  224 \
    --batch_size  4 \
    --num_workers 2 \
    --apply_simple_augment \
    --normalize_from_IMN \
    --output_dir  "$OUTPUT_DIR" \
    --dry_run \
    2>&1

echo ""
echo "============================================================"
echo " Dry-run PASSED — system is ready for full training."
echo " Run:  bash scripts/train_fsvfm_composite.sh <dataset_root>"
echo "============================================================"
