#!/usr/bin/env bash
# EVA-X 실행용 conda 환경(evax) 설치. 서버에서 1회 실행: bash scripts/setup_env.sh
set -e
ENV_NAME=evax

source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
    conda create -y -n "$ENV_NAME" python=3.10 || {
        echo "[setup] conda create 실패 (conda 패키지 미러 없음?). 출력 결과를 공유해 주세요."; exit 1; }
fi
conda activate "$ENV_NAME"

# torch 미러(torch-download)는 실제 파일을 download-r2.pytorch.org로 연결해 서버에서 받을 수 없음
# → PyPI 미러의 torch 2.1.2(cu121) + 짝이 맞는 xformers 0.0.23.post1 사용
python -m pip install torch==2.1.2 torchvision==0.16.2 xformers==0.0.23.post1 \
    timm==0.4.12 einops==0.6.1 "numpy<2" pandas==2.0.3 scikit-learn==1.3.0 scipy tqdm \
    opencv-python-headless tensorboard tensorboardX matplotlib MedPy==0.5.2
# --no-deps: libauc가 요구하는 opencv-python(GUI 의존성) 대신 headless 사용
python -m pip install --no-deps libauc==1.3.0

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO/EVA-X/classification"
PYTHONPATH="$REPO/compat" python - <<'PY'
import torch, xformers.ops as xops
import train  # 데이터 로더와 평가 코드까지 import 가능한지 확인 (학습은 실행하지 않음)
print('classification imports: ok')
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
