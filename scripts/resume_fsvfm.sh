#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Resume FSVFM fine-tuning from a checkpoint.
# Usage: bash scripts/resume_fsvfm.sh <experiment_dir> [checkpoint_tag]
#   <experiment_dir> : path to experiments/<run_name>
#   [checkpoint_tag] : latest | best_auc | min_val_loss  (default: latest)
# ---------------------------------------------------------------------------
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <experiment_dir> [checkpoint_tag=latest]"
    exit 1
fi

EXPERIMENT_DIR="$1"
TAG="${2:-latest}"
CHECKPOINT="${EXPERIMENT_DIR}/checkpoint-${TAG}.pth"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO_ROOT/fsvfm/finetune/cross_dataset_DFD_and_DiFF/train_composite.py"
PYTHON="/data/awais/anaconda/envs/d3/bin/python"

# Reload config from saved run
CONFIG="${EXPERIMENT_DIR}/config.json"
if [[ ! -f "$CONFIG" ]]; then
    echo "Config not found: $CONFIG — cannot safely resume."
    exit 1
fi

echo "============================================================"
echo " Resuming FSVFM training"
echo " Checkpoint  : $CHECKPOINT"
echo " Config      : $CONFIG"
echo " GPUs        : 4,5"
echo "============================================================"

# Extract all args from config.json and rebuild --train_root / --val_root flags
TRAIN_ROOTS=$(python3 -c "import json; d=json.load(open('$CONFIG')); [print(r) for r in d.get('train_root',[])]" 2>/dev/null)
VAL_ROOTS=$(python3 -c "import json; d=json.load(open('$CONFIG')); [print(r) for r in d.get('val_root',[])]" 2>/dev/null)
TOTAL_EPOCHS=$(python3 -c "import json; d=json.load(open('$CONFIG')); print(d.get('epochs',50))" 2>/dev/null)
OUTPUT_DIR=$(python3 -c "import json; d=json.load(open('$CONFIG')); print(d.get('output_dir','$EXPERIMENT_DIR'))" 2>/dev/null)
LOG_DIR=$(python3 -c "import json; d=json.load(open('$CONFIG')); print(d.get('log_dir','$EXPERIMENT_DIR'))" 2>/dev/null)

TRAIN_ROOT_ARGS=()
while IFS= read -r root; do
    [[ -n "$root" ]] && TRAIN_ROOT_ARGS+=(--train_root "$root")
done <<< "$TRAIN_ROOTS"

VAL_ROOT_ARGS=()
while IFS= read -r root; do
    [[ -n "$root" ]] && VAL_ROOT_ARGS+=(--val_root "$root")
done <<< "$VAL_ROOTS"

CUDA_VISIBLE_DEVICES=4,5 \
OMP_NUM_THREADS=4 \
"$PYTHON" -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port=29511 \
    "$SCRIPT" \
    "${TRAIN_ROOT_ARGS[@]}" \
    "${VAL_ROOT_ARGS[@]}" \
    --resume      "$CHECKPOINT" \
    --model       vit_large_patch16 \
    --nb_classes  2 \
    --input_size  224 \
    --batch_size  32 \
    --accum_iter  1 \
    --epochs      "$TOTAL_EPOCHS" \
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
    2>&1 | tee "${LOG_DIR}/resume_stdout.log"
