#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Full pipeline watcher: train (5 epochs) → eval all datasets → summary
#
# Fully resumable — reads sentinel files on startup and skips completed steps.
# Auto-restarts training on crash (up to MAX_RESTARTS).  All output goes into
# one fixed experiment directory whose path is persisted in pipeline_state.env.
#
# Usage:
#   # Fresh run — creates a new experiment directory automatically
#   bash experiment_image/scripts/pipeline_watcher.sh
#
#   # Resume an interrupted run (reads last EXP_DIR from state file)
#   bash experiment_image/scripts/pipeline_watcher.sh --resume
#
#   # Resume into a specific experiment directory
#   bash experiment_image/scripts/pipeline_watcher.sh --resume /path/to/exp_dir
#
# Steps (each protected by a .step_<name>.done sentinel):
#   1. train        — 5 epochs, DistributedBalancedSampler, 3 GPUs
#   2. eval_gend    — gend_unified test set (FFIW + ReWIND video + DFDC)
#   3. eval_images  — image_eval24
#   4. eval_rewind  — ReWIND images
#   5. eval_aigi    — AIGI_TEST
#   6. summary      — collect all inference_summary.txt → results_summary.txt
#
# Progress:
#   $EXP_DIR/pipeline_progress.txt  — human-readable step log
#   $EXP_DIR/pipeline.log           — timestamped watcher log
#   $EXP_DIR/.step_<name>.done      — sentinel after each step completes
#   $EXP_DIR/pipeline_state.env     — persisted EXP_DIR for --resume
# ---------------------------------------------------------------------------
set -uo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="/data/awais/anaconda/envs/d3/bin/python"
TRAIN_SCRIPT="$REPO_ROOT/experiment_image/train_image_robust.py"
EVAL_SCRIPT="$REPO_ROOT/experiment_image/eval_image.py"

# Shared state file stores the experiment directory path across invocations
STATE_FILE="$REPO_ROOT/experiment_image/pipeline_state.env"

# Training
TARGET_EPOCHS=5
TRAIN_GPUS="5,6,7"
TRAIN_NPROC=3
TRAIN_PORT=29521
BATCH_SIZE=128
MAX_RESTARTS=20
COOLDOWN_SECS=30

# Eval
EVAL_GPUS="5,6,7"
EVAL_NPROC=3
EVAL_BASE_PORT=29530
EVAL_BATCH=256

# Dataset paths
GEND_ROOT="/mnt/h200_dataset/saad/datasets/gend_unified"
IMAGE_EVAL24_ROOT="/mnt/h200_dataset/saad/datasets/images/image_eval24"
REWIND_IMG_ROOT="/mnt/h200_dataset/saad/datasets/images/ancestree/ReWIND"
AIGI_TEST_ROOT="/mnt/h200_dataset/kutub/datasets/AIGI/TEST"
FINETUNE_CKPT="$REPO_ROOT/experiments/fsvfm_vitl_composite_20260508_160444/FSFM_best_checkpoint.pth"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
RESUME_MODE=0
RESUME_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)
            RESUME_MODE=1
            if [[ $# -gt 1 && "$2" != --* ]]; then
                RESUME_DIR="$2"; shift
            fi
            shift ;;
        *) shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Determine experiment directory
# ---------------------------------------------------------------------------
if [[ $RESUME_MODE -eq 1 ]]; then
    if [[ -n "$RESUME_DIR" ]]; then
        EXP_DIR="$RESUME_DIR"
    elif [[ -f "$STATE_FILE" ]]; then
        # shellcheck disable=SC1090
        source "$STATE_FILE"
        # EXP_DIR is now set by sourcing the state file
    else
        echo "ERROR: --resume given but no state file found at $STATE_FILE"
        echo "       Specify the directory: --resume /path/to/exp_dir"
        exit 1
    fi
else
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    EXP_DIR="$REPO_ROOT/experiment_image/runs/img_robust_vitl_${TIMESTAMP}"
fi

mkdir -p "$EXP_DIR"

# Persist EXP_DIR so future --resume can find it without an explicit path
echo "EXP_DIR=\"$EXP_DIR\"" > "$STATE_FILE"

PIPELINE_LOG="$EXP_DIR/pipeline.log"
PROGRESS_FILE="$EXP_DIR/pipeline_progress.txt"
LATEST_CKPT="$EXP_DIR/checkpoint-latest.pth"
BEST_AUC_CKPT="$EXP_DIR/checkpoint-best_auc.pth"

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
plog() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg" | tee -a "$PIPELINE_LOG"
}

progress() {
    # Append a human-readable line to pipeline_progress.txt
    echo "$*" | tee -a "$PROGRESS_FILE"
    echo "$*" >> "$PIPELINE_LOG"
}

progress_header() {
    local title="$1"
    progress ""
    progress "$(printf '=%.0s' {1..68})"
    progress "  $title"
    progress "  $(date '+%Y-%m-%d %H:%M:%S')"
    progress "$(printf '=%.0s' {1..68})"
}

progress_ok() {
    progress "  [DONE] $*"
}

progress_skip() {
    progress "  [SKIP] $* (already completed)"
}

progress_fail() {
    progress "  [FAIL] $*"
}

sentinel() { echo "$EXP_DIR/.step_${1}.done"; }
is_done()  { [[ -f "$(sentinel "$1")" ]]; }
mark_done() {
    touch "$(sentinel "$1")"
    plog "Step '$1' marked complete."
}

# ---------------------------------------------------------------------------
# Read epoch from checkpoint
# ---------------------------------------------------------------------------
read_epoch() {
    local ckpt="$1"
    "$PYTHON" - "$ckpt" 2>/dev/null <<'PYEOF'
import sys, torch
try:
    c = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
    print(int(c.get('epoch', -1)))
except Exception:
    print(-1)
PYEOF
}

# ---------------------------------------------------------------------------
# Write a step separator to progress.txt
# ---------------------------------------------------------------------------
write_progress_table() {
    # Called after training finishes — prints known epoch results
    local result_txt="$EXP_DIR/result.txt"
    if [[ ! -f "$result_txt" ]]; then return; fi
    progress ""
    progress "  Epoch results:"
    grep -E "^Epoch" "$result_txt" 2>/dev/null | while IFS= read -r line; do
        progress "    $line"
    done
}

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
plog "========================================================"
plog "IMAGE-ROBUST FSVFM PIPELINE WATCHER"
plog "Experiment : $EXP_DIR"
plog "Steps      : train(${TARGET_EPOCHS}ep) → eval×4 → summary"
plog "Resume mode: $RESUME_MODE"
plog "========================================================"

# Write pipeline_progress.txt header only on first run
if [[ ! -f "$PROGRESS_FILE" ]]; then
    cat > "$PROGRESS_FILE" <<HDR
========================================================================
IMAGE-ROBUST FSVFM PIPELINE
Started    : $(date '+%Y-%m-%d %H:%M:%S')
Experiment : $EXP_DIR
Steps      : train(${TARGET_EPOCHS} epochs) → eval×4 → summary
GPUs       : train=$TRAIN_GPUS  eval=$EVAL_GPUS
Finetune   : $FINETUNE_CKPT
Dataset    : $GEND_ROOT
Compression aug: ENABLED
========================================================================

PIPELINE STEPS
--------------
HDR
fi

# ---------------------------------------------------------------------------
# STEP 1: Training
# ---------------------------------------------------------------------------
STEP="train"
if is_done "$STEP"; then
    progress_skip "Training (${TARGET_EPOCHS} epochs)"
    plog "Training already done — skipping."
else
    progress_header "STEP 1 / 6 — Training (${TARGET_EPOCHS} epochs)"
    plog "Starting training step."

    attempt=0
    consecutive_no_progress=0

    while true; do
        attempt=$((attempt + 1))

        RESUME_ARG=""
        CURRENT_EPOCH=-1
        if [[ -f "$LATEST_CKPT" ]]; then
            CURRENT_EPOCH=$(read_epoch "$LATEST_CKPT")
            RESUME_ARG="--resume $LATEST_CKPT"
            plog "Train attempt $attempt: resuming from epoch $CURRENT_EPOCH"
        else
            plog "Train attempt $attempt: starting from scratch"
        fi

        # Already reached target?
        if [[ $CURRENT_EPOCH -ge $((TARGET_EPOCHS - 1)) ]]; then
            plog "Target epochs ($TARGET_EPOCHS) already in checkpoint. Training done."
            mark_done "$STEP"
            break
        fi

        plog "Launching torchrun (GPUs=$TRAIN_GPUS nproc=$TRAIN_NPROC)..."
        TRAIN_START=$SECONDS

        CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
        OMP_NUM_THREADS=4 \
        "$PYTHON" -m torch.distributed.run \
            --nproc_per_node="$TRAIN_NPROC" \
            --master_port="$TRAIN_PORT" \
            "$TRAIN_SCRIPT" \
            --finetune         "$FINETUNE_CKPT" \
            --real_train_txt   "$GEND_ROOT/0_real_train.txt" \
            --fake_train_txt   "$GEND_ROOT/1_fake_train.txt" \
            --real_test_txt    "$GEND_ROOT/0_real_test.txt" \
            --fake_test_txt    "$GEND_ROOT/1_fake_test.txt" \
            --txt_path_remap_from "/data/saad/datasets/gend_unified" \
            --txt_path_remap_to   "$GEND_ROOT" \
            --output_dir       "$EXP_DIR" \
            --batch_size       "$BATCH_SIZE" \
            --epochs           "$TARGET_EPOCHS" \
            --warmup_epochs    1 \
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
            2>&1 | tee -a "$EXP_DIR/train_stdout.log"
        TRAIN_EXIT=${PIPESTATUS[0]}

        ELAPSED=$(( SECONDS - TRAIN_START ))
        plog "Training exited after ${ELAPSED}s with code $TRAIN_EXIT"

        if [[ $TRAIN_EXIT -eq 0 ]]; then
            plog "Training completed normally."
            mark_done "$STEP"
            break
        fi

        # Check progress
        NEW_EPOCH=-1
        [[ -f "$LATEST_CKPT" ]] && NEW_EPOCH=$(read_epoch "$LATEST_CKPT")

        if [[ $NEW_EPOCH -ge $((TARGET_EPOCHS - 1)) ]]; then
            plog "Target epochs reached in checkpoint ($NEW_EPOCH). Training done."
            mark_done "$STEP"
            break
        fi

        if [[ $NEW_EPOCH -le $CURRENT_EPOCH && $CURRENT_EPOCH -ge 0 ]]; then
            consecutive_no_progress=$((consecutive_no_progress + 1))
            plog "No epoch progress ($CURRENT_EPOCH → $NEW_EPOCH). Consecutive: $consecutive_no_progress"
        else
            consecutive_no_progress=0
            plog "Progress: epoch $CURRENT_EPOCH → $NEW_EPOCH"
        fi

        if [[ $consecutive_no_progress -ge 5 ]]; then
            progress_fail "Training: 5 consecutive crashes with no progress. Aborting pipeline."
            plog "FATAL: giving up after 5 no-progress crashes."
            exit 1
        fi

        if [[ $attempt -ge $MAX_RESTARTS ]]; then
            progress_fail "Training: reached $MAX_RESTARTS restarts. Aborting pipeline."
            plog "FATAL: max restarts exceeded."
            exit 1
        fi

        plog "Waiting ${COOLDOWN_SECS}s before retry..."
        progress "  [WAIT] Training crash (attempt $attempt) — retrying in ${COOLDOWN_SECS}s ..."
        sleep "$COOLDOWN_SECS"
    done

    # Resolve which checkpoint to use for evaluation
    if [[ -f "$BEST_AUC_CKPT" ]]; then
        EVAL_CKPT="$BEST_AUC_CKPT"
        plog "Will use best_auc checkpoint for eval: $BEST_AUC_CKPT"
    else
        EVAL_CKPT="$LATEST_CKPT"
        plog "best_auc not found — will use latest: $LATEST_CKPT"
    fi

    write_progress_table
    progress_ok "Training — best_auc checkpoint: $(basename "$EVAL_CKPT")"
fi

# Resolve eval checkpoint (needed even on resume)
if [[ -f "$BEST_AUC_CKPT" ]]; then
    EVAL_CKPT="$BEST_AUC_CKPT"
elif [[ -f "$LATEST_CKPT" ]]; then
    EVAL_CKPT="$LATEST_CKPT"
else
    plog "ERROR: No checkpoint found in $EXP_DIR after training."
    exit 1
fi
plog "Eval checkpoint: $EVAL_CKPT"

# ---------------------------------------------------------------------------
# Helper: run one evaluation step
# ---------------------------------------------------------------------------
EVAL_PORT=$EVAL_BASE_PORT

run_eval_step() {
    local step_name="$1"
    local display_name="$2"
    local step_num="$3"
    shift 3  # remaining args passed to eval_image.py

    local out_dir="$EXP_DIR/eval_${step_name}"

    if is_done "$step_name"; then
        progress_skip "$display_name"
        plog "Eval '$step_name' already done — skipping."
        EVAL_PORT=$((EVAL_PORT + 1))
        return 0
    fi

    progress_header "STEP ${step_num} / 6 — Eval: ${display_name}"
    plog "Starting eval: $step_name  →  $out_dir"
    mkdir -p "$out_dir"

    CUDA_VISIBLE_DEVICES="$EVAL_GPUS" \
    OMP_NUM_THREADS=4 \
    "$PYTHON" -m torch.distributed.run \
        --nproc_per_node="$EVAL_NPROC" \
        --master_port="$EVAL_PORT" \
        "$EVAL_SCRIPT" \
        --checkpoint   "$EVAL_CKPT" \
        --output_dir   "$out_dir" \
        --dataset_name "$display_name" \
        --batch_size   "$EVAL_BATCH" \
        --num_workers  8 \
        --fake_class_idx 1 \
        "$@" \
        2>&1 | tee "$out_dir/eval.log"
    local eval_exit=${PIPESTATUS[0]}

    EVAL_PORT=$((EVAL_PORT + 1))

    if [[ $eval_exit -ne 0 ]]; then
        progress_fail "$display_name eval exited with code $eval_exit"
        plog "ERROR: eval '$step_name' failed with code $eval_exit"
        return 1
    fi

    # Extract key metrics from inference_summary.txt
    local summary="$out_dir/inference_summary.txt"
    if [[ -f "$summary" ]]; then
        local acc auc racc facc
        acc=$(grep  "Accuracy"      "$summary" | grep -v AUC | head -1 | awk '{print $NF}')
        auc=$(grep  "AUC"           "$summary" | head -1 | awk '{print $NF}')
        racc=$(grep "Real_Accuracy" "$summary" | head -1 | awk '{print $NF}')
        facc=$(grep "Fake_Accuracy" "$summary" | head -1 | awk '{print $NF}')
        progress_ok "$display_name — Acc=$acc  AUC=$auc  Real=$racc  Fake=$facc"
    else
        progress_ok "$display_name — done (no summary file found)"
    fi

    mark_done "$step_name"
}

# ---------------------------------------------------------------------------
# STEP 2: Eval — gend_unified test set
# ---------------------------------------------------------------------------
run_eval_step "eval_gend" "gend_unified_test" "2" \
    --real_txt "$GEND_ROOT/0_real_test.txt" \
    --fake_txt "$GEND_ROOT/1_fake_test.txt" \
    --path_remap_from "/data/saad/datasets/gend_unified" \
    --path_remap_to   "$GEND_ROOT"

# ---------------------------------------------------------------------------
# STEP 3: Eval — image_eval24
# ---------------------------------------------------------------------------
run_eval_step "eval_image_eval24" "image_eval24" "3" \
    --dataset_root "$IMAGE_EVAL24_ROOT"

# ---------------------------------------------------------------------------
# STEP 4: Eval — ReWIND images
# ---------------------------------------------------------------------------
run_eval_step "eval_rewind_img" "ReWIND_images" "4" \
    --dataset_root "$REWIND_IMG_ROOT"

# ---------------------------------------------------------------------------
# STEP 5: Eval — AIGI_TEST
# ---------------------------------------------------------------------------
run_eval_step "eval_aigi" "AIGI_TEST" "5" \
    --dataset_root "$AIGI_TEST_ROOT"

# ---------------------------------------------------------------------------
# STEP 6: Summary
# ---------------------------------------------------------------------------
STEP="summary"
if is_done "$STEP"; then
    progress_skip "Results summary"
    plog "Summary already done — skipping."
else
    progress_header "STEP 6 / 6 — Results Summary"
    plog "Building results_summary.txt"

    SUMMARY_FILE="$EXP_DIR/results_summary.txt"
    NOW=$(date '+%Y-%m-%d %H:%M:%S')

    {
        echo "========================================================================"
        echo "IMAGE-ROBUST FSVFM — FULL PIPELINE RESULTS"
        echo "Generated  : $NOW"
        echo "Experiment : $EXP_DIR"
        echo "Checkpoint : $EVAL_CKPT ($(basename "$EVAL_CKPT"))"
        echo "Training   : ${TARGET_EPOCHS} epochs, compression augmentation ENABLED"
        echo "========================================================================"
        echo ""

        # Training results table
        local_result="$EXP_DIR/result.txt"
        if [[ -f "$local_result" ]]; then
            echo "TRAINING RESULTS"
            echo "----------------"
            cat "$local_result"
            echo ""
        fi

        # Per-dataset eval results
        echo "EVALUATION RESULTS"
        echo "=================="
        echo ""
        for pair in \
            "eval_gend:gend_unified_test" \
            "eval_image_eval24:image_eval24" \
            "eval_rewind_img:ReWIND_images" \
            "eval_aigi:AIGI_TEST"
        do
            step_name="${pair%%:*}"
            display="${pair##*:}"
            summary="$EXP_DIR/${step_name}/inference_summary.txt"
            echo "--------------------------------------------------------------------"
            echo "DATASET: $display"
            if [[ -f "$summary" ]]; then
                grep -E "(Accuracy|AUC|Real_Accuracy|Fake_Accuracy|Samples|Videos)" \
                    "$summary" 2>/dev/null || true
            else
                echo "  (no summary — eval not run or failed)"
            fi
            echo ""
        done

        echo "========================================================================"
        echo "PREVIOUS MODEL (composite, no compression aug) — for comparison"
        echo "========================================================================"
        prev_summary="$REPO_ROOT/experiments/fsvfm_vitl_composite_20260508_160444/inference_epoch3_best_auc/summary_results.txt"
        if [[ -f "$prev_summary" ]]; then
            echo ""
            echo "  image_eval24  : Composite AUC=70.77%  Baseline AUC=56.39%"
            echo "  ReWIND_images : Composite AUC=67.81%"
            echo "  AIGI_TEST     : Composite AUC=59.68%"
            echo "  (see $prev_summary for full table)"
        fi
        echo ""
        echo "========================================================================"
        echo "OUTPUT FILES"
        echo "========================================================================"
        for pair in \
            "eval_gend:gend_unified_test" \
            "eval_image_eval24:image_eval24" \
            "eval_rewind_img:ReWIND_images" \
            "eval_aigi:AIGI_TEST"
        do
            step_name="${pair%%:*}"
            echo "  $EXP_DIR/${step_name}/inference_results.csv"
            echo "  $EXP_DIR/${step_name}/inference_summary.txt"
        done
        echo "========================================================================"
    } | tee "$SUMMARY_FILE"

    plog "Summary written: $SUMMARY_FILE"
    mark_done "$STEP"
    progress_ok "Results summary: $SUMMARY_FILE"
fi

# ---------------------------------------------------------------------------
# Final status
# ---------------------------------------------------------------------------
progress ""
progress "$(printf '=%.0s' {1..68})"
progress "  PIPELINE COMPLETE"
progress "  Finished at : $(date '+%Y-%m-%d %H:%M:%S')"
progress "  Experiment  : $EXP_DIR"
progress "  Summary     : $EXP_DIR/results_summary.txt"
progress "  Best model  : $BEST_AUC_CKPT"
progress "$(printf '=%.0s' {1..68})"

plog "========================================================"
plog "Pipeline complete."
plog "results_summary.txt : $EXP_DIR/results_summary.txt"
plog "pipeline_progress   : $PROGRESS_FILE"
plog "========================================================"

echo ""
echo "All done. Experiment: $EXP_DIR"
echo "  results_summary.txt : $EXP_DIR/results_summary.txt"
echo "  pipeline_progress   : $PROGRESS_FILE"
