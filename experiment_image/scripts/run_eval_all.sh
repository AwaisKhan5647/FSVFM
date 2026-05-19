#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run frame-level evaluation on all target datasets for a given experiment.
#
# Evaluates the best_auc checkpoint on:
#   1. gend_unified test  (FFIW + ReWIND video frames + DFDC) — txt-based
#   2. image_eval24       — folder-based
#   3. ReWIND_images      — folder-based
#   4. AIGI_TEST          — folder-based (recursive)
#
# Usage:
#   bash experiment_image/scripts/run_eval_all.sh --exp_dir <path>
#   bash experiment_image/scripts/run_eval_all.sh --exp_dir <path> --gpus 5,6,7
#   bash experiment_image/scripts/run_eval_all.sh --exp_dir <path> --ckpt best_acc
#
# Outputs per dataset: inference_results.csv  +  inference_summary.txt
# Also writes results_summary.txt to the experiment root.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="/data/awais/anaconda/envs/d3/bin/python"
EVAL_SCRIPT="$REPO_ROOT/experiment_image/eval_image.py"

# Dataset roots
GEND_ROOT="/mnt/h200_dataset/saad/datasets/gend_unified"
IMAGE_EVAL24_ROOT="/mnt/h200_dataset/saad/datasets/images/image_eval24"
REWIND_IMG_ROOT="/mnt/h200_dataset/saad/datasets/images/ancestree/ReWIND"
AIGI_TEST_ROOT="/mnt/h200_dataset/kutub/datasets/AIGI/TEST"

REMAP_FROM="/data/saad/datasets/gend_unified"
REMAP_TO="$GEND_ROOT"

# Defaults
EXP_DIR=""
CKPT_TAG="best_auc"
GPUS="5,6,7"
NPROC=3
MASTER_PORT=29522
BATCH=256

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp_dir)  EXP_DIR="$2";  shift 2 ;;
        --ckpt)     CKPT_TAG="$2"; shift 2 ;;
        --gpus)     GPUS="$2";     shift 2 ;;
        --nproc)    NPROC="$2";    shift 2 ;;
        --port)     MASTER_PORT="$2"; shift 2 ;;
        *) shift ;;
    esac
done

if [[ -z "$EXP_DIR" ]]; then
    echo "ERROR: --exp_dir is required"
    echo "Usage: bash run_eval_all.sh --exp_dir experiment_image/runs/img_robust_vitl_..."
    exit 1
fi

CHECKPOINT="$EXP_DIR/checkpoint-${CKPT_TAG}.pth"
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "Available checkpoints in $EXP_DIR:"
    ls "$EXP_DIR"/checkpoint-*.pth 2>/dev/null || echo "  (none)"
    exit 1
fi

echo "============================================================"
echo " Image-Robust FSVFM Evaluation"
echo " Experiment : $EXP_DIR"
echo " Checkpoint : $CHECKPOINT"
echo " GPUs       : $GPUS  (nproc=$NPROC)"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Helper: run eval for one dataset
# ---------------------------------------------------------------------------
run_eval() {
    local name="$1"
    local out_dir="$EXP_DIR/eval_${name}"
    shift

    echo "------------------------------------------------------------"
    echo " Evaluating: $name  →  $out_dir"
    echo "------------------------------------------------------------"

    mkdir -p "$out_dir"

    CUDA_VISIBLE_DEVICES="$GPUS" \
    OMP_NUM_THREADS=4 \
    "$PYTHON" -m torch.distributed.run \
        --nproc_per_node="$NPROC" \
        --master_port="$MASTER_PORT" \
        "$EVAL_SCRIPT" \
        --checkpoint  "$CHECKPOINT" \
        --output_dir  "$out_dir" \
        --dataset_name "$name" \
        --batch_size  "$BATCH" \
        --num_workers 8 \
        --fake_class_idx 1 \
        "$@" \
        2>&1 | tee "$out_dir/eval.log"

    MASTER_PORT=$((MASTER_PORT + 1))
    echo ""
}

# ---------------------------------------------------------------------------
# 1. gend_unified test set  (FFIW + ReWIND video + DFDC video frames)
# ---------------------------------------------------------------------------
if [[ -f "$GEND_ROOT/0_real_test.txt" ]]; then
    run_eval "gend_unified_test" \
        --real_txt "$GEND_ROOT/0_real_test.txt" \
        --fake_txt "$GEND_ROOT/1_fake_test.txt" \
        --path_remap_from "$REMAP_FROM" \
        --path_remap_to   "$REMAP_TO"
else
    echo "WARNING: gend_unified test txt not found at $GEND_ROOT — skipping"
fi

# ---------------------------------------------------------------------------
# 2. image_eval24
# ---------------------------------------------------------------------------
if [[ -d "$IMAGE_EVAL24_ROOT" ]]; then
    run_eval "image_eval24" --dataset_root "$IMAGE_EVAL24_ROOT"
else
    echo "WARNING: image_eval24 not found at $IMAGE_EVAL24_ROOT — skipping"
fi

# ---------------------------------------------------------------------------
# 3. ReWIND images
# ---------------------------------------------------------------------------
if [[ -d "$REWIND_IMG_ROOT" ]]; then
    run_eval "ReWIND_images" --dataset_root "$REWIND_IMG_ROOT"
else
    echo "WARNING: ReWIND images not found at $REWIND_IMG_ROOT — skipping"
fi

# ---------------------------------------------------------------------------
# 4. AIGI_TEST
# ---------------------------------------------------------------------------
if [[ -d "$AIGI_TEST_ROOT" ]]; then
    run_eval "AIGI_TEST" --dataset_root "$AIGI_TEST_ROOT"
else
    echo "WARNING: AIGI_TEST not found at $AIGI_TEST_ROOT — skipping"
fi

# ---------------------------------------------------------------------------
# Collect results into results_summary.txt
# ---------------------------------------------------------------------------
SUMMARY_FILE="$EXP_DIR/results_summary.txt"
NOW=$(date '+%Y-%m-%d %H:%M:%S')

{
    echo "============================================================"
    echo "IMAGE-ROBUST FSVFM EVALUATION SUMMARY"
    echo "Generated  : $NOW"
    echo "Experiment : $EXP_DIR"
    echo "Checkpoint : $CHECKPOINT"
    echo "============================================================"
    echo ""

    for name in gend_unified_test image_eval24 ReWIND_images AIGI_TEST; do
        summary="$EXP_DIR/eval_${name}/inference_summary.txt"
        if [[ -f "$summary" ]]; then
            echo "------------------------------------------------------------"
            echo "DATASET: $name"
            echo "------------------------------------------------------------"
            grep -E "^(  Accuracy|  AUC|  Real_Accuracy|  Fake_Accuracy|Samples)" "$summary" 2>/dev/null || true
            echo ""
        fi
    done

    echo "============================================================"
    echo "Detailed summaries in:"
    for name in gend_unified_test image_eval24 ReWIND_images AIGI_TEST; do
        echo "  $EXP_DIR/eval_${name}/inference_summary.txt"
    done
    echo "============================================================"
} | tee "$SUMMARY_FILE"

echo ""
echo "All evaluations complete."
echo "Summary: $SUMMARY_FILE"
