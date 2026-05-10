#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Launch FSVFM inference on test set using 4 GPUs (1,2,3,4)
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/data/awais/anaconda/envs/d3/bin/python"
EXP="$REPO_ROOT/experiments/fsvfm_vitl_composite_20260508_160444"
CHECKPOINT="$EXP/checkpoint-best_auc.pth"
DATASET_ROOT="/mnt/h200_dataset/saad/datasets/gend_unified"
OUTPUT_DIR="$EXP/inference_epoch3_best_auc"
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo " FSVFM Inference — Epoch 3 best_auc checkpoint"
echo " Checkpoint : $CHECKPOINT"
echo " Output     : $OUTPUT_DIR"
echo " GPUs       : 1,2,3,4"
echo " Test frames: 142,038 (8 unseen generators)"
echo "============================================================"

CUDA_VISIBLE_DEVICES=1,2,3,4 \
OMP_NUM_THREADS=4 \
"$PYTHON" -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port=29520 \
    "$REPO_ROOT/scripts/run_inference.py" \
    --checkpoint      "$CHECKPOINT" \
    --real_txt        "$DATASET_ROOT/0_real_test.txt" \
    --fake_txt        "$DATASET_ROOT/1_fake_test.txt" \
    --output_dir      "$OUTPUT_DIR" \
    --path_remap_from "/data/saad/datasets/gend_unified" \
    --path_remap_to   "$DATASET_ROOT" \
    --batch_size      256 \
    --num_workers     8 \
    --model           vit_large_patch16 \
    2>&1 | tee "$OUTPUT_DIR/inference.log"

echo ""
echo "Done. Outputs:"
echo "  $OUTPUT_DIR/inference_results.csv"
echo "  $OUTPUT_DIR/video_results.csv"
echo "  $OUTPUT_DIR/inference_summary.txt"
