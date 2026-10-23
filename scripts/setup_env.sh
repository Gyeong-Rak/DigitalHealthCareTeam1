#!/usr/bin/env bash
# EVA-X 실행용 conda 환경(evax) 설치. 서버에서 1회 실행: bash scripts/setup_env.sh
set -e
ENV_NAME=evax
TORCH_INDEX=http://torch-download.snuhub-cache-server/whl/cu118

source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
    conda create -y -n "$ENV_NAME" python=3.10 || {
        echo "[setup] conda create 실패 (conda 패키지 미러 없음?). 출력 결과를 공유해 주세요."; exit 1; }
fi
conda activate "$ENV_NAME"

# torch 2.0.1 + CUDA 11.8 (xformers 0.0.21과 짝이 맞는 버전)
pip install torch==2.0.1 torchvision==0.15.2 --index-url "$TORCH_INDEX"
# --no-deps: torch 재설치 방지 / libauc가 요구하는 opencv-python(GUI 의존성) 대신 headless 사용
pip install --no-deps xformers==0.0.21 libauc==1.3.0
pip install timm==0.4.12 einops==0.6.1 "numpy<2" pandas==2.0.3 scikit-learn==1.3.0 scipy tqdm \
    opencv-python-headless tensorboard tensorboardX matplotlib

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO/EVA-X/classification"
PYTHONPATH="$REPO/compat" python - <<'PY'
import torch, xformers.ops as xops
print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| gpus', torch.cuda.device_count())
q = torch.randn(1, 16, 4, 32, device='cuda', dtype=torch.half)
xops.memory_efficient_attention(q, q, q)
from timm.models import create_model
from models import models_eva  # noqa: F401  (모델 등록)
m = create_model('eva02_tiny_patch16_xattn_fusedLN_SwiGLU_preln_RoPE', num_classes=5, use_mean_pooling=True).cuda().eval()
with torch.no_grad(), torch.cuda.amp.autocast():
    print('forward ok:', tuple(m(torch.randn(2, 3, 224, 224, device='cuda')).shape))
print('[setup] 완료')
PY
