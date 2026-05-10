#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# FSVFM Composite-Dataset Fine-tuning — ViT-L/16 on GPUs 5,6
#
# Normal usage (no args needed — paths are hard-coded below):
#   bash scripts/train_fsvfm_composite.sh
#
# Override output dir (used by watchdog.sh on resume):
#   bash scripts/train_fsvfm_composite.sh --output_dir /path/to/existing/exp \
#                                         --resume /path/to/checkpoint-latest.pth
#
# Override specific txt files if needed:
#   bash scripts/train_fsvfm_composite.sh \
#       --real_train /other/0_real_train.txt --fake_train /other/1_fake_train.txt ...
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO_ROOT/fsvfm/finetune/cross_dataset_DFD_and_DiFF/train_composite.py"
PYTHON="/data/awais/anaconda/envs/d3/bin/python"
PRETRAINED="/data/awais/projects/GenD_NeSy/weights/FS-VFM/FS-VFM-ViT-L.pth"

# ---------------------------------------------------------------------------
# Dataset paths (hard-coded — no args needed for normal use)
# ---------------------------------------------------------------------------
DATASET_ROOT="/mnt/h200_dataset/saad/datasets/gend_unified"
REAL_TRAIN_TXT="$DATASET_ROOT/0_real_train.txt"
FAKE_TRAIN_TXT="$DATASET_ROOT/1_fake_train.txt"
REAL_TEST_TXT="$DATASET_ROOT/0_real_test.txt"
FAKE_TEST_TXT="$DATASET_ROOT/1_fake_test.txt"
TXT_PATH_FROM="/data/saad/datasets/gend_unified"
TXT_PATH_TO="$DATASET_ROOT"

# Output (watchdog overrides these by passing --output_dir and --resume)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="fsvfm_vitl_composite_${TIMESTAMP}"
OUTPUT_DIR="$REPO_ROOT/experiments/${EXPERIMENT_NAME}"
LOG_DIR="$REPO_ROOT/logs/${EXPERIMENT_NAME}"
RESUME_ARG=""
POSARGS=()

# ---------------------------------------------------------------------------
# Parse optional overrides
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --real_train)      REAL_TRAIN_TXT="$2"; shift 2 ;;
        --fake_train)      FAKE_TRAIN_TXT="$2"; shift 2 ;;
        --real_test)       REAL_TEST_TXT="$2";  shift 2 ;;
        --fake_test)       FAKE_TEST_TXT="$2";  shift 2 ;;
        --path_remap_from) TXT_PATH_FROM="$2";  shift 2 ;;
        --path_remap_to)   TXT_PATH_TO="$2";    shift 2 ;;
        --output_dir)      OUTPUT_DIR="$2";     shift 2 ;;
        --resume)          RESUME_ARG="$2";     shift 2 ;;
        *) POSARGS+=("$1"); shift ;;
    esac
done

LOG_DIR="${LOG_DIR:-$OUTPUT_DIR}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "============================================================"
echo " FSVFM Composite Training — ViT-L/16"
echo " Experiment : $(basename "$OUTPUT_DIR")"
echo " Output     : $OUTPUT_DIR"
echo " Logs       : $LOG_DIR"
echo " GPUs       : 5,6 (2× RTX 6000 Ada, ~50 GB each)"
echo " Batch/GPU  : 128  (64 real + 64 fake — strict balance)"
echo " Eff. batch : 256  (128 × 2 GPUs)"
echo " Epochs     : 20"
[[ -n "$RESUME_ARG" ]] && echo " Resuming   : $RESUME_ARG"
echo "============================================================"

# Build dataset args
DATASET_ARGS=()
if [[ -n "$REAL_TRAIN_TXT" && -n "$FAKE_TRAIN_TXT" ]]; then
    DATASET_ARGS+=(--real_train_txt "$REAL_TRAIN_TXT" --fake_train_txt "$FAKE_TRAIN_TXT")
    DATASET_ARGS+=(--real_test_txt  "$REAL_TEST_TXT"  --fake_test_txt  "$FAKE_TEST_TXT")
else
    for root in "${POSARGS[@]}"; do
        DATASET_ARGS+=(--train_root "$root" --val_root "$root")
    done
fi

# Resume arg (empty string → not passed)
RESUME_ARGS=()
[[ -n "$RESUME_ARG" ]] && RESUME_ARGS=(--resume "$RESUME_ARG")

CUDA_VISIBLE_DEVICES=5,6 \
OMP_NUM_THREADS=8 \
"$PYTHON" -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port=29510 \
    "$SCRIPT" \
    "${DATASET_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    --txt_path_remap_from  "$TXT_PATH_FROM" \
    --txt_path_remap_to    "$TXT_PATH_TO" \
    --finetune             "$PRETRAINED" \
    --model                vit_large_patch16 \
    --nb_classes           2 \
    --input_size           224 \
    --batch_size           128 \
    --accum_iter           1 \
    --epochs               20 \
    --warmup_epochs        2 \
    --blr                  5e-5 \
    --layer_decay          0.90 \
    --weight_decay         0.01 \
    --drop_path            0.1 \
    --mixup                0.8 \
    --cutmix               1.0 \
    --reprob               0.25 \
    --apply_simple_augment \
    --normalize_from_IMN \
    --balanced_batch \
    --num_workers          12 \
    --pin_mem \
    --save_every           1 \
    --output_dir           "$OUTPUT_DIR" \
    --log_dir              "$LOG_DIR" \
    2>&1 | tee -a "$LOG_DIR/train_stdout.log"

echo "Training complete. Results at : $OUTPUT_DIR/result.txt"
echo "Progress log                  : $OUTPUT_DIR/progress.txt"
