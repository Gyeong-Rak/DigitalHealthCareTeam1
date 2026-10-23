#!/usr/bin/env bash
# prepare | check bce|label_smoothing | train bce|label_smoothing | evaluate
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
STAGE=${1:?usage: run_tiny_calibration.sh prepare|check|train|evaluate [bce|label_smoothing]}
LOSS=${2:-bce}
DATA_DIR=${DATA_DIR:-$HOME/workspace/datasets/chexpert}
SPLITS_DIR=${SPLITS_DIR:-$HOME/workspace/outputs/chexpert/splits_seed${SPLIT_SEED:-42}}
WEIGHTS_DIR=${WEIGHTS_DIR:-$HOME/workspace/pretrained}
PRETRAINED=$WEIGHTS_DIR/eva_x_tiny_patch16_merged520k_mim.pt
OUT_ROOT=${OUT_ROOT:-$HOME/workspace/outputs/chexpert}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTHONPATH="$REPO/compat${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
if [[ "$STAGE" == prepare ]]; then
    python scripts/prepare_chexpert_splits.py --data-root "$DATA_DIR" --output-dir "$SPLITS_DIR" \
        --seed "${SPLIT_SEED:-42}" --check-images
    exit
fi
ARGS=("$STAGE" --loss "$LOSS" --data-root "$DATA_DIR" --splits-dir "$SPLITS_DIR"
      --seed "${SEED:-0}" --batch-size "${BATCH_SIZE:-128}" --workers "${NUM_WORKERS:-8}"
      --epochs "${EPOCHS:-5}" --smoothing "${SMOOTHING:-0.05}" --accumulation "${ACCUMULATION:-1}"
      --lr "${LR:-0.001}" --layer-decay "${LAYER_DECAY:-0.55}" --weight-decay "${WEIGHT_DECAY:-0.05}")
case "$STAGE" in
    check)
        python scripts/train_tiny_calibration.py "${ARGS[@]}" --pretrained "$PRETRAINED"
        ;;
    train)
        if [[ -n "${RESUME:-}" ]]; then
            RUN_DIR=${RUN_DIR:-$(dirname "$RESUME")}
            ARGS+=(--resume "$RESUME")
        else
            RUN_DIR=${RUN_DIR:-$OUT_ROOT/train_tiny_${LOSS}_seed${SEED:-0}_$(date +%Y%m%d_%H%M%S)}
        fi
        mkdir -p "$RUN_DIR"
        torchrun --standalone --nproc_per_node="${NGPU:-4}" scripts/train_tiny_calibration.py \
            "${ARGS[@]}" --pretrained "$PRETRAINED" --output-dir "$RUN_DIR" 2>&1 | tee -a "$RUN_DIR/run.log"
        ;;
    evaluate)
        CHECKPOINT=${CHECKPOINT:?set CHECKPOINT to the trained run best.pth}
        EVAL_SPLIT=${EVAL_SPLIT:-test}
        RESULT_DIR=${RESULT_DIR:-$(dirname "$CHECKPOINT")/eval_${EVAL_SPLIT}_$(date +%Y%m%d_%H%M%S)}
        python scripts/train_tiny_calibration.py "${ARGS[@]}" --checkpoint "$CHECKPOINT" \
            --eval-split "$EVAL_SPLIT" --output-dir "$RESULT_DIR"
        if [[ "$EVAL_SPLIT" == test ]]; then
            python scripts/evaluate_calibration.py --eval-dir "$RESULT_DIR"
        fi
        ;;
    *) echo "stage는 prepare, check, train, evaluate 중 하나입니다" >&2; exit 2 ;;
esac
