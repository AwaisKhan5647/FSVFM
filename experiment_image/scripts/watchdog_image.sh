#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Watchdog for image-robust FSVFM training.
#
# Creates one fixed output directory at startup, then auto-restarts training
# from checkpoint-latest.pth whenever the process crashes or is killed.
#
# Usage:
#   bash experiment_image/scripts/watchdog_image.sh                   # new run
#   bash experiment_image/scripts/watchdog_image.sh --resume <dir>    # resume run
#
# Stops when:
#   (a) training exits 0 (completed normally)
#   (b) checkpoint epoch >= TARGET_EPOCHS - 1
#   (c) 5 consecutive crashes without epoch progress
#   (d) MAX_RESTARTS attempts exceeded
# ---------------------------------------------------------------------------
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="/data/awais/anaconda/envs/d3/bin/python"
TRAIN_SCRIPT="$REPO_ROOT/experiment_image/scripts/train_image.sh"

MAX_RESTARTS=20
COOLDOWN_SECS=30
TARGET_EPOCHS=20

# Parse args
EXISTING_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume) EXISTING_DIR="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# Output directory
if [[ -n "$EXISTING_DIR" ]]; then
    OUTPUT_DIR="$EXISTING_DIR"
    echo "[watchdog] Resuming: $OUTPUT_DIR"
else
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="$REPO_ROOT/experiment_image/runs/img_robust_vitl_${TIMESTAMP}"
    echo "[watchdog] New run: $OUTPUT_DIR"
fi

mkdir -p "$OUTPUT_DIR"
WATCHDOG_LOG="$OUTPUT_DIR/watchdog.log"
LATEST_CKPT="$OUTPUT_DIR/checkpoint-latest.pth"

# Read epoch from checkpoint
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

wlog() {
    local msg="[watchdog][$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$WATCHDOG_LOG"
}

# Start progress watcher (parses tqdm output → progress.txt live block)
WATCHER_PID=""
start_progress_watcher() {
    if [[ -n "$WATCHER_PID" ]] && kill -0 "$WATCHER_PID" 2>/dev/null; then
        return
    fi
    "$PYTHON" -u "$REPO_ROOT/scripts/progress_watcher.py" \
        "$OUTPUT_DIR" --interval 120 \
        >> "$OUTPUT_DIR/progress_watcher.log" 2>&1 &
    WATCHER_PID=$!
    wlog "Progress watcher started (PID $WATCHER_PID)"
}
start_progress_watcher

wlog "========================================================"
wlog "Image-robust FSVFM Watchdog"
wlog "Output dir   : $OUTPUT_DIR"
wlog "Max restarts : $MAX_RESTARTS"
wlog "Target epochs: $TARGET_EPOCHS"
wlog "========================================================"

attempt=0
consecutive_no_progress=0

while true; do
    attempt=$((attempt + 1))

    RESUME_ARG=""
    CURRENT_EPOCH=-1
    if [[ -f "$LATEST_CKPT" ]]; then
        CURRENT_EPOCH=$(read_epoch "$LATEST_CKPT")
        RESUME_ARG="$LATEST_CKPT"
        wlog "Attempt $attempt: resuming from epoch $CURRENT_EPOCH"
    else
        wlog "Attempt $attempt: starting from scratch"
    fi

    if [[ $CURRENT_EPOCH -ge $((TARGET_EPOCHS - 1)) ]]; then
        wlog "Target epochs ($TARGET_EPOCHS) reached. Done."
        break
    fi

    wlog "Launching train_image.sh --output_dir $OUTPUT_DIR ${RESUME_ARG:+--resume $RESUME_ARG}"
    START_TS=$SECONDS

    bash "$TRAIN_SCRIPT" \
        --output_dir "$OUTPUT_DIR" \
        ${RESUME_ARG:+--resume "$RESUME_ARG"} \
        2>&1 | tee -a "$OUTPUT_DIR/watchdog_stdout.log"
    EXIT_CODE=${PIPESTATUS[0]}

    ELAPSED=$(( SECONDS - START_TS ))
    wlog "Exited after ${ELAPSED}s with code $EXIT_CODE"

    if [[ $EXIT_CODE -eq 0 ]]; then
        wlog "Completed normally. Watchdog done."
        break
    fi

    NEW_EPOCH=-1
    [[ -f "$LATEST_CKPT" ]] && NEW_EPOCH=$(read_epoch "$LATEST_CKPT")

    if [[ $NEW_EPOCH -ge $((TARGET_EPOCHS - 1)) ]]; then
        wlog "Target epochs reached (epoch $NEW_EPOCH). Done."
        break
    fi

    if [[ $NEW_EPOCH -le $CURRENT_EPOCH && $CURRENT_EPOCH -ge 0 ]]; then
        consecutive_no_progress=$((consecutive_no_progress + 1))
        wlog "No progress ($CURRENT_EPOCH → $NEW_EPOCH). Consecutive failures: $consecutive_no_progress"
    else
        consecutive_no_progress=0
        wlog "Progress: epoch $CURRENT_EPOCH → $NEW_EPOCH"
    fi

    if [[ $consecutive_no_progress -ge 5 ]]; then
        wlog "ERROR: 5 consecutive failures. Giving up."
        exit 1
    fi

    if [[ $attempt -ge $MAX_RESTARTS ]]; then
        wlog "ERROR: Max restarts ($MAX_RESTARTS) reached. Giving up."
        exit 1
    fi

    wlog "Waiting ${COOLDOWN_SECS}s before restart..."
    sleep "$COOLDOWN_SECS"
    start_progress_watcher
done

wlog "========================================================"
wlog "Watchdog finished after $attempt attempt(s)"
wlog "Best AUC ckpt: $OUTPUT_DIR/checkpoint-best_auc.pth"
wlog "Results      : $OUTPUT_DIR/result.txt"
wlog "Progress     : $OUTPUT_DIR/progress.txt"
wlog "========================================================"

echo ""
echo "Training done."
echo "  Best AUC  : $OUTPUT_DIR/checkpoint-best_auc.pth"
echo "  Best Acc  : $OUTPUT_DIR/checkpoint-best_acc.pth"
echo "  Results   : $OUTPUT_DIR/result.txt"
echo ""
echo "Next: run evaluation with:"
echo "  bash experiment_image/scripts/run_eval_all.sh --exp_dir $OUTPUT_DIR"
