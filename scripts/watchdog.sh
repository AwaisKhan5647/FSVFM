#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# FSVFM Training Watchdog
#
# Launches training and auto-restarts from checkpoint-latest.pth if the
# process crashes or is killed.  All restarts use the SAME output directory
# so checkpoints accumulate and best-model comparisons remain consistent.
#
# Usage:
#   bash scripts/watchdog.sh                  # start fresh
#   bash scripts/watchdog.sh --resume <dir>   # resume an existing run
#
# Watchdog stops when:
#   (a) training exits with code 0 (completed normally), OR
#   (b) target epochs have been reached (reads epoch from checkpoint), OR
#   (c) MAX_RESTARTS consecutive crashes without progress
# ---------------------------------------------------------------------------
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/data/awais/anaconda/envs/d3/bin/python"
TRAIN_SCRIPT="$REPO_ROOT/scripts/train_fsvfm_composite.sh"

# ---------------------------------------------------------------------------
# Watchdog config
# ---------------------------------------------------------------------------
MAX_RESTARTS=20          # maximum restart attempts before giving up
COOLDOWN_SECS=30         # wait between restarts
TARGET_EPOCHS=20         # stop when checkpoint epoch >= this - 1 (0-indexed)

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
EXISTING_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume) EXISTING_DIR="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Set up output directory — fixed for entire watchdog session
# ---------------------------------------------------------------------------
if [[ -n "$EXISTING_DIR" ]]; then
    OUTPUT_DIR="$EXISTING_DIR"
    echo "[watchdog] Resuming existing run at: $OUTPUT_DIR"
else
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    EXPERIMENT_NAME="fsvfm_vitl_composite_${TIMESTAMP}"
    OUTPUT_DIR="$REPO_ROOT/experiments/${EXPERIMENT_NAME}"
    echo "[watchdog] Starting new run: $OUTPUT_DIR"
fi

LOG_DIR="$REPO_ROOT/logs/$(basename "$OUTPUT_DIR")"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
WATCHDOG_LOG="$OUTPUT_DIR/watchdog.log"
LATEST_CKPT="$OUTPUT_DIR/checkpoint-latest.pth"

# ---------------------------------------------------------------------------
# Helper: read epoch number from a checkpoint file
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
# Helper: log message to both stdout and watchdog.log
# ---------------------------------------------------------------------------
wlog() {
    local msg="[watchdog][$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$WATCHDOG_LOG"
}

# ---------------------------------------------------------------------------
# Watchdog loop
# ---------------------------------------------------------------------------
attempt=0
consecutive_no_progress=0

wlog "========================================================"
wlog "FSVFM Training Watchdog started"
wlog "Output dir   : $OUTPUT_DIR"
wlog "Max restarts : $MAX_RESTARTS"
wlog "Target epochs: $TARGET_EPOCHS"
wlog "========================================================"

# ---------------------------------------------------------------------------
# Start progress watcher (updates progress.txt every 2 minutes)
# ---------------------------------------------------------------------------
WATCHER_PID=""
start_progress_watcher() {
    if [[ -n "$WATCHER_PID" ]] && kill -0 "$WATCHER_PID" 2>/dev/null; then
        return  # already running
    fi
    "$PYTHON" -u "$REPO_ROOT/scripts/progress_watcher.py" \
        "$OUTPUT_DIR" --interval 120 \
        >> "$OUTPUT_DIR/progress_watcher.log" 2>&1 &
    WATCHER_PID=$!
    wlog "Progress watcher started (PID $WATCHER_PID)"
}
start_progress_watcher

while true; do
    attempt=$((attempt + 1))

    # ---- Determine resume checkpoint ----
    RESUME_ARG=""
    CURRENT_EPOCH=-1
    if [[ -f "$LATEST_CKPT" ]]; then
        RESUME_ARG="$LATEST_CKPT"
        CURRENT_EPOCH=$(read_epoch "$LATEST_CKPT")
        wlog "Attempt $attempt: resuming from epoch $CURRENT_EPOCH  ($LATEST_CKPT)"
    else
        wlog "Attempt $attempt: starting from scratch (no checkpoint yet)"
    fi

    # ---- Check if already done ----
    if [[ $CURRENT_EPOCH -ge $((TARGET_EPOCHS - 1)) ]]; then
        wlog "Target epochs ($TARGET_EPOCHS) already reached at epoch $CURRENT_EPOCH. Done."
        break
    fi

    # ---- Launch training ----
    wlog "Launching: bash $TRAIN_SCRIPT --output_dir $OUTPUT_DIR ${RESUME_ARG:+--resume $RESUME_ARG}"
    START_TS=$SECONDS

    bash "$TRAIN_SCRIPT" \
        --output_dir "$OUTPUT_DIR" \
        ${RESUME_ARG:+--resume "$RESUME_ARG"} \
        2>&1 | tee -a "$LOG_DIR/train_stdout.log"
    EXIT_CODE=${PIPESTATUS[0]}

    ELAPSED=$(( SECONDS - START_TS ))
    wlog "Training exited after ${ELAPSED}s with code $EXIT_CODE"

    # ---- Check exit code ----
    if [[ $EXIT_CODE -eq 0 ]]; then
        wlog "Training completed successfully (exit 0). Watchdog done."
        break
    fi

    # ---- Check epoch progress since last restart ----
    NEW_EPOCH=-1
    [[ -f "$LATEST_CKPT" ]] && NEW_EPOCH=$(read_epoch "$LATEST_CKPT")

    if [[ $NEW_EPOCH -ge $((TARGET_EPOCHS - 1)) ]]; then
        wlog "Target epochs reached (epoch $NEW_EPOCH). Watchdog done."
        break
    fi

    if [[ $NEW_EPOCH -le $CURRENT_EPOCH && $CURRENT_EPOCH -ge 0 ]]; then
        consecutive_no_progress=$((consecutive_no_progress + 1))
        wlog "No epoch progress detected ($CURRENT_EPOCH → $NEW_EPOCH). Consecutive failures: $consecutive_no_progress"
    else
        consecutive_no_progress=0
        wlog "Progress: epoch $CURRENT_EPOCH → $NEW_EPOCH"
    fi

    # ---- Abort if too many consecutive failures without progress ----
    if [[ $consecutive_no_progress -ge 5 ]]; then
        wlog "ERROR: 5 consecutive restarts with no epoch progress. Giving up."
        exit 1
    fi

    # ---- Abort if max restarts reached ----
    if [[ $attempt -ge $MAX_RESTARTS ]]; then
        wlog "ERROR: Max restarts ($MAX_RESTARTS) reached. Giving up."
        exit 1
    fi

    # ---- Cooldown before next attempt ----
    wlog "Waiting ${COOLDOWN_SECS}s before restart..."
    sleep "$COOLDOWN_SECS"
    start_progress_watcher  # ensure watcher is alive after restart
done

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
wlog "========================================================"
wlog "Watchdog finished after $attempt attempt(s)"
wlog "Results : $OUTPUT_DIR/result.txt"
wlog "Progress: $OUTPUT_DIR/progress.txt"
wlog "Best AUC checkpoint : $OUTPUT_DIR/checkpoint-best_auc.pth"
wlog "Best Acc checkpoint : $OUTPUT_DIR/checkpoint-best_acc.pth"
wlog "========================================================"

echo ""
echo "Training done. Key outputs:"
echo "  result.txt   : $OUTPUT_DIR/result.txt"
echo "  progress.txt : $OUTPUT_DIR/progress.txt"
echo "  Best AUC     : $OUTPUT_DIR/checkpoint-best_auc.pth"
echo "  Best Acc     : $OUTPUT_DIR/checkpoint-best_acc.pth"
