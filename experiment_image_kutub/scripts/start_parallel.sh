#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Launch both kutub variants in parallel.
#
# Variant A — no compression augmentation — GPU 3
# Variant B — with compression augmentation — GPU 4
#
# Usage:
#   bash experiment_image_kutub/scripts/start_parallel.sh            # fresh
#   bash experiment_image_kutub/scripts/start_parallel.sh --resume   # resume both
#   bash experiment_image_kutub/scripts/start_parallel.sh --epochs 10
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PIPELINE_SCRIPT="$REPO_ROOT/experiment_image_kutub/scripts/pipeline_kutub.sh"

EXTRA_ARGS=""
EPOCHS=20

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)        EXTRA_ARGS="$EXTRA_ARGS --resume"; shift ;;
        --epochs)        EPOCHS="$2"; EXTRA_ARGS="$EXTRA_ARGS --epochs $2"; shift 2 ;;
        --max_per_class) EXTRA_ARGS="$EXTRA_ARGS --max_per_class $2"; shift 2 ;;
        *)               EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

LAUNCH_LOG="$REPO_ROOT/experiment_image_kutub/parallel_launch.log"
echo "============================================================" | tee -a "$LAUNCH_LOG"
echo " Starting parallel Kutub pipelines" | tee -a "$LAUNCH_LOG"
echo " $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LAUNCH_LOG"
echo " Epochs : $EPOCHS" | tee -a "$LAUNCH_LOG"
echo " Args   : $EXTRA_ARGS" | tee -a "$LAUNCH_LOG"
echo "============================================================" | tee -a "$LAUNCH_LOG"
echo ""

LOG_A="$REPO_ROOT/experiment_image_kutub/variant_a_watcher.log"
LOG_B="$REPO_ROOT/experiment_image_kutub/variant_b_watcher.log"

echo "Launching Variant A (no aug, GPU 3) → log: $LOG_A"
bash "$PIPELINE_SCRIPT" --variant a --gpu 3 $EXTRA_ARGS \
    > "$LOG_A" 2>&1 &
PID_A=$!
echo "  PID: $PID_A"

echo "Launching Variant B (with aug, GPU 4) → log: $LOG_B"
bash "$PIPELINE_SCRIPT" --variant b --gpu 4 $EXTRA_ARGS \
    > "$LOG_B" 2>&1 &
PID_B=$!
echo "  PID: $PID_B"

echo ""
echo "Both pipelines running."
echo "  Variant A (PID $PID_A): tail -f $LOG_A"
echo "  Variant B (PID $PID_B): tail -f $LOG_B"
echo ""
echo "Waiting for both to finish..."

# Wait for both and capture exit codes
EXIT_A=0
EXIT_B=0
wait "$PID_A" || EXIT_A=$?
echo "[$(date '+%H:%M:%S')] Variant A finished (exit $EXIT_A)" | tee -a "$LAUNCH_LOG"

wait "$PID_B" || EXIT_B=$?
echo "[$(date '+%H:%M:%S')] Variant B finished (exit $EXIT_B)" | tee -a "$LAUNCH_LOG"

echo "" | tee -a "$LAUNCH_LOG"
echo "============================================================" | tee -a "$LAUNCH_LOG"
echo " Both variants complete." | tee -a "$LAUNCH_LOG"
echo " Variant A exit: $EXIT_A" | tee -a "$LAUNCH_LOG"
echo " Variant B exit: $EXIT_B" | tee -a "$LAUNCH_LOG"

# ---------------------------------------------------------------------------
# Generate comparison summary
# ---------------------------------------------------------------------------
SUMMARY="$REPO_ROOT/experiment_image_kutub/comparison_summary.txt"
NOW=$(date '+%Y-%m-%d %H:%M:%S')

{
    echo "================================================================"
    echo " KUTUB VARIANTS COMPARISON SUMMARY"
    echo " Generated: $NOW"
    echo "================================================================"
    echo ""

    for VARIANT in a b; do
        STATE="$REPO_ROOT/experiment_image_kutub/pipeline_state_variant_${VARIANT}.env"
        LABEL="Variant A (no augmentation)"
        [[ "$VARIANT" == "b" ]] && LABEL="Variant B (compression augmentation)"
        echo "----------------------------------------------------------------"
        echo " $LABEL"
        echo "----------------------------------------------------------------"
        if [[ -f "$STATE" ]]; then
            source "$STATE"
            echo " Exp dir : $EXP_DIR"
            if [[ -f "$EXP_DIR/result.txt" ]]; then
                echo ""
                grep -E "(BEST_AVG_AUC|avg_AUC|image_eval24|ReWIND_images|AIGI_TEST|FINAL)" \
                    "$EXP_DIR/result.txt" | head -20 || true
            else
                echo " (result.txt not found)"
            fi
        else
            echo " (pipeline state not found — variant not run?)"
        fi
        echo ""
    done

    echo "================================================================"
    echo " Full result files:"
    for VARIANT in a b; do
        STATE="$REPO_ROOT/experiment_image_kutub/pipeline_state_variant_${VARIANT}.env"
        if [[ -f "$STATE" ]]; then
            source "$STATE"
            echo "  Variant $VARIANT: $EXP_DIR/result.txt"
        fi
    done
    echo "================================================================"
} | tee "$SUMMARY"

echo ""
echo "Comparison summary: $SUMMARY"

if [[ "$EXIT_A" -ne 0 || "$EXIT_B" -ne 0 ]]; then
    echo "WARNING: One or both variants exited with errors. Check logs."
    exit 1
fi
