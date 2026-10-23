#!/usr/bin/env bash
# EVA-X CheXpert 평가/학습 (서버에서 실행)
#   bash scripts/run_chexpert.sh eval  tiny|small   # 공개된 fine-tuned 가중치로 valid AUC 확인
#   bash scripts/run_chexpert.sh train tiny|small   # MIM 사전학습 가중치에서 fine-tuning
# 환경변수로 변경 가능: NGPU, EPOCHS, BATCH_SIZE, NUM_WORKERS, WEIGHTS_DIR, DATA_DIR, OUT_ROOT
# 하이퍼파라미터는 EVA-X/classification/train_files/eva_x/chexpert/vit_{ti,s}.sh와 동일
set -eo pipefail
MODE=${1:?usage: run_chexpert.sh eval|train tiny|small}
SIZE=${2:?usage: run_chexpert.sh eval|train tiny|small}

REPO=$(cd "$(dirname "$0")/.." && pwd)
WEIGHTS_DIR=${WEIGHTS_DIR:-$HOME/workspace/pretrained}
DATA_DIR=${DATA_DIR:-$HOME/workspace/datasets/chexpert/}   # 끝의 / 필수
OUT_ROOT=${OUT_ROOT:-$HOME/workspace/outputs}

case $SIZE in
    tiny)
        MODEL=eva02_tiny_patch16_xattn_fusedLN_SwiGLU_preln_RoPE
        PRETRAINED=eva_x_tiny_patch16_merged520k_mim.pt
        FINETUNED=eva_x_tiny_patch16_mrerged520k_mim_chexpert_ft.pth   # 원본 파일명 오타(mrerged) 그대로
        DEFAULT_EPOCHS=100
        EXTRA="--drop_path 0.2 --stop_grad_conv1 --use_smooth_label" ;;
    small)
        MODEL=eva02_small_patch16_xattn_fusedLN_SwiGLU_preln_RoPE
        PRETRAINED=eva_x_small_patch16_merged520k_mim.pt
        FINETUNED=eva_x_small_patch16_merged520k_mim_chexpert_ft.pth
        DEFAULT_EPOCHS=60
        EXTRA="--drop_path 0.15 --find_unused_parameters --seed 14" ;;
    *) echo "SIZE는 tiny 또는 small"; exit 1 ;;
esac

case $MODE in
    eval)  NGPU=${NGPU:-1}; CKPT=$WEIGHTS_DIR/$FINETUNED; EXTRA="$EXTRA --eval" ;;
    train) NGPU=${NGPU:-4}; CKPT=$WEIGHTS_DIR/$PRETRAINED ;;
    *) echo "MODE는 eval 또는 train"; exit 1 ;;
esac
[ -f "$CKPT" ] || { echo "가중치 없음: $CKPT"; exit 1; }
[ -s "$CKPT" ] || { echo "가중치가 0바이트입니다. Share 업로드가 완료된 파일을 지정하세요: $CKPT"; exit 1; }

SAVE_DIR=$OUT_ROOT/chexpert/${MODE}_${SIZE}_$(date +%m%d_%H%M)
mkdir -p "$SAVE_DIR"
export PYTHONPATH="$REPO/compat${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO/EVA-X/classification"

# 에폭 표기: EVA-X는 chexpert에서 "fake epoch"를 쓰며 로그의 Epoch = fake epoch // 20
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node="$NGPU" train.py \
    --dataset chexpert \
    --input_size 224 \
    --finetune "$CKPT" \
    --output_dir "$SAVE_DIR" \
    --log_dir "$SAVE_DIR" \
    --batch_size "${BATCH_SIZE:-256}" \
    --checkpoint_type "" \
    --epochs "${EPOCHS:-$DEFAULT_EPOCHS}" \
    --blr 1e-3 --layer_decay 0.55 --weight_decay 0.05 \
    --fixed_lr \
    --model "$MODEL" \
    --warmup_epochs 0 \
    --mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 \
    --data_path "$DATA_DIR" \
    --num_workers "${NUM_WORKERS:-8}" \
    --train_list "$DATA_DIR/train.csv" \
    --val_list ./ \
    --test_list "$DATA_DIR/valid.csv" \
    --nb_classes 5 \
    --eval_interval 1 \
    --use_mean_pooling \
    $EXTRA 2>&1 | tee "$SAVE_DIR/run.log"
