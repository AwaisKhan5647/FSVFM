#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Pre-flight dry-run: verifies dataloaders (real/fake balance), GPU allocation,
# checkpoint load, DDP init, and one-batch forward/backward before full training.
#
# Usage (txt-pair mode — recommended):
#   bash scripts/dry_run.sh \
#       --real_train /data/saad/datasets/gend_unified/0_real_train.txt \
#       --fake_train /data/saad/datasets/gend_unified/1_fake_train.txt \
#       --real_test  /data/saad/datasets/gend_unified/0_real_test.txt \
#       --fake_test  /data/saad/datasets/gend_unified/1_fake_test.txt
#
# Usage (directory mode — legacy):
#   bash scripts/dry_run.sh <train_root> [test_root]
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO_ROOT/fsvfm/finetune/cross_dataset_DFD_and_DiFF/train_composite.py"
PYTHON="/data/awais/anaconda/envs/d3/bin/python"
PRETRAINED="/data/awais/projects/GenD_NeSy/weights/FS-VFM/FS-VFM-ViT-L.pth"
OUTPUT_DIR="$REPO_ROOT/experiments/dry_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Parse args
DATASET_ROOT="/mnt/h200_dataset/saad/datasets/gend_unified"
REAL_TRAIN_TXT="$DATASET_ROOT/0_real_train.txt"
FAKE_TRAIN_TXT="$DATASET_ROOT/1_fake_train.txt"
REAL_TEST_TXT="$DATASET_ROOT/0_real_test.txt"
FAKE_TEST_TXT="$DATASET_ROOT/1_fake_test.txt"
TXT_PATH_FROM="/data/saad/datasets/gend_unified"
TXT_PATH_TO="$DATASET_ROOT"
POSARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --real_train)      REAL_TRAIN_TXT="$2"; shift 2 ;;
        --fake_train)      FAKE_TRAIN_TXT="$2"; shift 2 ;;
        --real_test)       REAL_TEST_TXT="$2";  shift 2 ;;
        --fake_test)       FAKE_TEST_TXT="$2";  shift 2 ;;
        --path_remap_from) TXT_PATH_FROM="$2";  shift 2 ;;
        --path_remap_to)   TXT_PATH_TO="$2";    shift 2 ;;
        *) POSARGS+=("$1"); shift ;;
    esac
done

# Build dataset args
DATASET_ARGS=()
if [[ -n "$REAL_TRAIN_TXT" && -n "$FAKE_TRAIN_TXT" ]]; then
    DATASET_ARGS+=(--real_train_txt "$REAL_TRAIN_TXT" --fake_train_txt "$FAKE_TRAIN_TXT")
    [[ -n "$REAL_TEST_TXT" ]]  && DATASET_ARGS+=(--real_test_txt "$REAL_TEST_TXT")
    [[ -n "$FAKE_TEST_TXT" ]]  && DATASET_ARGS+=(--fake_test_txt "$FAKE_TEST_TXT")
elif [[ ${#POSARGS[@]} -ge 1 ]]; then
    DATASET_ARGS+=(--train_root "${POSARGS[0]}")
    if [[ ${#POSARGS[@]} -ge 2 ]]; then
        DATASET_ARGS+=(--val_root "${POSARGS[1]}")
    else
        DATASET_ARGS+=(--val_root "${POSARGS[0]}")
    fi
else
    echo "Usage: $0 --real_train <txt> --fake_train <txt> [--real_test <txt> --fake_test <txt>]"
    echo "   or: $0 <train_root> [test_root]"
    exit 1
fi

echo "============================================================"
echo " FSVFM Dry-Run Verification"
echo " Pretrained : $PRETRAINED"
echo " Output     : $OUTPUT_DIR"
echo " GPUs       : 5,6"
echo "============================================================"

CUDA_VISIBLE_DEVICES=5,6 \
OMP_NUM_THREADS=8 \
"$PYTHON" -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port=29512 \
    "$SCRIPT" \
    "${DATASET_ARGS[@]}" \
    --txt_path_remap_from "$TXT_PATH_FROM" \
    --txt_path_remap_to   "$TXT_PATH_TO" \
    --finetune      "$PRETRAINED" \
    --model         vit_large_patch16 \
    --nb_classes    2 \
    --input_size    224 \
    --batch_size    128 \
    --num_workers   4 \
    --apply_simple_augment \
    --normalize_from_IMN \
    --balanced_batch \
    --output_dir    "$OUTPUT_DIR" \
    --dry_run \
    2>&1

echo ""
echo "============================================================"
echo " Dry-run PASSED — batch label mix logged above."
echo " Run full training:"
echo "   bash scripts/train_fsvfm_composite.sh \\"
echo "       --real_train <0_real_train.txt> \\"
echo "       --fake_train <1_fake_train.txt> \\"
echo "       --real_test  <0_real_test.txt>  \\"
echo "       --fake_test  <1_fake_test.txt>"
echo "============================================================"
