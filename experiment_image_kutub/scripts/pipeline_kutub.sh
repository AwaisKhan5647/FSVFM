#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Single-variant pipeline watcher for experiment_image_kutub.
#
# Runs training with per-epoch evaluation.  Supports resume after crash.
# Each run gets its own timestamped output directory under
#   experiment_image_kutub/runs/<variant>_<timestamp>/
#
# Usage:
#   bash experiment_image_kutub/scripts/pipeline_kutub.sh \
#       --variant a --gpu 3           # no aug, fresh run
#   bash experiment_image_kutub/scripts/pipeline_kutub.sh \
#       --variant b --gpu 4           # with aug, fresh run
#   bash experiment_image_kutub/scripts/pipeline_kutub.sh \
#       --variant a --gpu 3 --resume  # resume latest run of variant a
#   bash experiment_image_kutub/scripts/pipeline_kutub.sh \
#       --variant a --gpu 3 --exp_dir experiment_image_kutub/runs/variant_a_noaug_20260525_120000
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="/data/awais/anaconda/envs/d3/bin/python"
TRAIN_SCRIPT="$REPO_ROOT/experiment_image_kutub/train_kutub.py"

# Defaults
VARIANT="a"
GPU="3"
AUG_FLAG="--no_compression_aug"
EXP_DIR=""
RESUME_FLAG=""
EPOCHS=20
BATCH=64
MAX_RESTARTS=20
MAX_PER_CLASS=150000  # cap per class to keep epochs ~45 min on single GPU; 0=all data
STATE_FILE=""         # override: custom state file path (avoids collision between parallel runs)
USE_AUG_OVERRIDE=""   # override: "yes" forces --use_compression_aug regardless of variant name

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)        VARIANT="$2";          shift 2 ;;
        --gpu)            GPU="$2";              shift 2 ;;
        --exp_dir)        EXP_DIR="$2";          shift 2 ;;
        --resume)         RESUME_FLAG="--resume"; shift ;;
        --epochs)         EPOCHS="$2";           shift 2 ;;
        --batch)          BATCH="$2";            shift 2 ;;
        --max_per_class)  MAX_PER_CLASS="$2";    shift 2 ;;
        --state_file)     STATE_FILE="$2";       shift 2 ;;
        --use_aug)        USE_AUG_OVERRIDE="yes"; shift ;;
        --no_aug)         USE_AUG_OVERRIDE="no";  shift ;;
        *) echo "Unknown arg: $1"; shift ;;
    esac
done

# Set aug flag: explicit override takes priority, otherwise infer from variant name
if [[ "$USE_AUG_OVERRIDE" == "yes" ]]; then
    AUG_FLAG="--use_compression_aug"
elif [[ "$USE_AUG_OVERRIDE" == "no" ]]; then
    AUG_FLAG="--no_compression_aug"
elif [[ "$VARIANT" == "b" ]]; then
    AUG_FLAG="--use_compression_aug"
fi

# Default state file if not overridden
if [[ -z "$STATE_FILE" ]]; then
    STATE_FILE="$REPO_ROOT/experiment_image_kutub/pipeline_state_variant_${VARIANT}.env"
fi

# ---------------------------------------------------------------------------
# Determine experiment directory
# ---------------------------------------------------------------------------
if [[ -n "$EXP_DIR" ]]; then
    # Explicitly provided
    mkdir -p "$EXP_DIR"
elif [[ -n "$RESUME_FLAG" && -f "$STATE_FILE" ]]; then
    # Resume: load saved EXP_DIR
    source "$STATE_FILE"
    echo "Resuming experiment: $EXP_DIR"
else
    # Fresh: create new timestamped directory
    TS=$(date '+%Y%m%d_%H%M%S')
    AUG_LABEL="noaug"
    [[ "$VARIANT" == "b" ]] && AUG_LABEL="aug"
    EXP_DIR="$REPO_ROOT/experiment_image_kutub/runs/variant_${VARIANT}_${AUG_LABEL}_${TS}"
    mkdir -p "$EXP_DIR"
fi

# Save state for resume
echo "EXP_DIR=$EXP_DIR" > "$STATE_FILE"

PIPELINE_LOG="$EXP_DIR/pipeline.log"
echo "" >> "$PIPELINE_LOG"
echo "============================================================" >> "$PIPELINE_LOG"
echo " Kutub pipeline — variant=$VARIANT  gpu=$GPU  aug=${AUG_FLAG}" >> "$PIPELINE_LOG"
echo " Started: $(date '+%Y-%m-%d %H:%M:%S')" >> "$PIPELINE_LOG"
echo " EXP_DIR: $EXP_DIR" >> "$PIPELINE_LOG"
echo "============================================================" >> "$PIPELINE_LOG"

echo ""
echo "============================================================"
echo " Kutub FOUNDATIONS Pipeline"
echo " Variant      : $VARIANT  ($AUG_FLAG)"
echo " GPU          : $GPU"
echo " Exp dir      : $EXP_DIR"
echo " Epochs       : $EPOCHS"
echo " Max/class    : ${MAX_PER_CLASS} (0=all data)"
echo " Resume       : ${RESUME_FLAG:-no}"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Training with auto-restart on crash
# ---------------------------------------------------------------------------
RESTART=0
SUCCESS=0

# Detect if already finished (all epochs done)
LATEST_CKPT="$EXP_DIR/checkpoint-latest.pth"

get_completed_epoch() {
    if [[ -f "$LATEST_CKPT" ]]; then
        "$PYTHON" -c "
import torch, sys
try:
    c = torch.load('$LATEST_CKPT', map_location='cpu', weights_only=False)
    print(int(c.get('epoch', -1)) + 1)
except Exception as e:
    print(0)
" 2>/dev/null
    else
        echo "0"
    fi
}

COMPLETED=$(get_completed_epoch)
echo "Completed epochs so far: $COMPLETED / $EPOCHS"

if [[ "$COMPLETED" -ge "$EPOCHS" ]]; then
    echo "Training already complete ($COMPLETED epochs). Skipping to summary."
    SUCCESS=1
fi

while [[ "$SUCCESS" -eq 0 && "$RESTART" -le "$MAX_RESTARTS" ]]; do
    if [[ "$RESTART" -gt 0 ]]; then
        echo "[$(date '+%H:%M:%S')] Restart $RESTART/$MAX_RESTARTS ..."
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Restart $RESTART/$MAX_RESTARTS" >> "$PIPELINE_LOG"
        sleep 5
    fi

    # Always use --resume if checkpoint exists
    RESUME_ARG=""
    if [[ -f "$LATEST_CKPT" ]]; then
        RESUME_ARG="--resume"
    fi

    echo "[$(date '+%H:%M:%S')] Launching training (attempt $((RESTART+1)))..."

    set +e
    CUDA_VISIBLE_DEVICES="$GPU" \
    "$PYTHON" "$TRAIN_SCRIPT" \
        --output_dir    "$EXP_DIR" \
        --epochs        "$EPOCHS" \
        --batch_size    "$BATCH" \
        --max_per_class "$MAX_PER_CLASS" \
        $AUG_FLAG \
        $RESUME_ARG \
        2>&1 | tee -a "$EXP_DIR/train_stdout.log"
    EXIT_CODE=$?
    set -e

    if [[ "$EXIT_CODE" -eq 0 ]]; then
        SUCCESS=1
        echo "[$(date '+%H:%M:%S')] Training completed successfully."
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training completed (exit 0)" >> "$PIPELINE_LOG"
    else
        echo "[$(date '+%H:%M:%S')] Training exited with code $EXIT_CODE."
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training exited $EXIT_CODE" >> "$PIPELINE_LOG"
        RESTART=$((RESTART + 1))

        # Verify progress before restarting
        NEW_COMPLETED=$(get_completed_epoch)
        if [[ "$NEW_COMPLETED" -ge "$EPOCHS" ]]; then
            echo "All $EPOCHS epochs complete — marking success."
            SUCCESS=1
        fi
    fi
done

if [[ "$SUCCESS" -eq 0 ]]; then
    echo "ERROR: Training failed after $MAX_RESTARTS restarts." | tee -a "$PIPELINE_LOG"
    exit 1
fi

# ---------------------------------------------------------------------------
# Write pipeline progress summary
# ---------------------------------------------------------------------------
PROGRESS_FILE="$EXP_DIR/pipeline_progress.txt"
{
    echo "============================================================"
    echo " PIPELINE COMPLETE — Variant $VARIANT"
    echo " Finished : $(date '+%Y-%m-%d %H:%M:%S')"
    echo " Exp dir  : $EXP_DIR"
    echo "============================================================"
    echo ""
    if [[ -f "$EXP_DIR/result.txt" ]]; then
        echo "--- TRAINING RESULTS ---"
        cat "$EXP_DIR/result.txt"
    fi
    echo ""
    echo "============================================================"
} | tee "$PROGRESS_FILE"

echo "" >> "$PIPELINE_LOG"
echo "Pipeline complete at $(date '+%Y-%m-%d %H:%M:%S')" >> "$PIPELINE_LOG"
echo "EXP_DIR=$EXP_DIR" >> "$PIPELINE_LOG"

echo ""
echo "Pipeline complete."
echo "Results: $EXP_DIR/result.txt"
echo "Checkpoint: $EXP_DIR/checkpoint-best_avg_auc.pth"
