# DigitalHealthCareTeam1

## 작업 흐름
SNUHub 서버는 외부 인터넷이 차단되어 있어 서버에서는 GitHub 캐시 서버를 통해 **pull만** 가능합니다.
코드는 각자 로컬 PC에서 작업하고, 서버에서는 pull 후 실행만 합니다.

### 1. 로컬 PC에 clone (최초 1회)
```bash
git clone --recurse-submodules https://github.com/Gyeong-Rak/DigitalHealthCareTeam1.git
```
push하려면 Collaborator 초대를 받아야 합니다. GitHub 아이디를 팀장에게 알려주세요.

### 2. 로컬에서 작업 후 push
```bash
git pull
# 코드 수정
git add <파일> && git commit -m "메시지" && git push
```

### 3. 서버에서 pull 후 실행
서버(공용 계정)에는 `~/workspace/DigitalHealthCareTeam1`에 이미 clone되어 있습니다.
```bash
cd ~/workspace/DigitalHealthCareTeam1 && git pull && git submodule update --init --recursive
```

### 규칙
- **서버에서 코드를 직접 수정하지 않기** (서버에서는 push 불가, pull 시 충돌 발생)
- 서버 폴더를 모두 함께 쓰므로, 다른 사람이 학습을 돌리는 중에 pull하면 코드가 바뀔 수 있음 → pull 전에 단톡방에 공유
- 브랜치는 쓰지 않고 `main` 하나로 작업 (서버에서 checkout하면 모두에게 영향)

## 구성
- `EVA-X/`: [hustvl/EVA-X](https://github.com/hustvl/EVA-X) 서브모듈 (원본 코드, 직접 수정하지 않음)
  - URL을 상대 경로(`../../hustvl/EVA-X.git`)로 지정해 두어 로컬에서는 github.com, 서버에서는 캐시 서버에서 받아집니다.

## 데이터셋 (서버 전용, 저장소에는 포함하지 않음)

### CheXpert-v1.0-small
| 항목 | 서버 경로 |
|---|---|
| train 이미지 | `~/workspace/datasets/chexpert/train/` (223,415장) |
| valid 이미지 | `~/workspace/datasets/chexpert/valid/` (234장) |
| 라벨 | `~/workspace/datasets/chexpert/train.csv`, `valid.csv` |
| 원본 zip | `~/workspace/updown_share/CheXpert-v1.0-small.zip` |

EVA-X 학습 스크립트에서 사용할 경로 (`prepare_dataset.py`로 복사할 필요 없음):
```bash
DATASET_DIR=/home/coder/workspace/datasets/chexpert/   # 끝의 / 필수 (경로를 문자열로 이어붙임)
TRAIN_LIST=/home/coder/workspace/datasets/chexpert/train.csv
TEST_LIST=/home/coder/workspace/datasets/chexpert/valid.csv
```

> `~/workspace`는 네트워크 드라이브(CIFS)라 작은 파일 I/O가 느릴 수 있습니다.

## EVA-X 실행 (서버)

### 1. 환경 설치 (최초 1회)
```bash
cd ~/workspace/DigitalHealthCareTeam1 && bash scripts/setup_env.sh
```
conda 환경 `evax`(python 3.10, torch 2.1.2+cu121, xformers 0.0.23.post1, timm 0.4.12)를 만들고 마지막에 GPU 동작을 확인합니다.
NVIDIA apex는 서버에서 빌드할 수 없어 `compat/apex/`의 PyTorch 대체 구현을 사용합니다 (결과 동일).

### 2. 가중치
`~/workspace/pretrained/`에 둡니다 (HuggingFace [MapleF/eva_x](https://huggingface.co/MapleF/eva_x), 서버에서 직접 다운로드 불가 → 로컬에서 받아 업로드).

| 파일 | 용도 |
|---|---|
| `eva_x_tiny_patch16_merged520k_mim.pt` | Ti 사전학습 (fine-tuning 시작점) |
| `eva_x_tiny_patch16_mrerged520k_mim_chexpert_ft.pth` | Ti CheXpert fine-tuned (평가용) |
| `eva_x_small_patch16_merged520k_mim.pt` | S 사전학습 |
| `eva_x_small_patch16_merged520k_mim_chexpert_ft.pth` | S CheXpert fine-tuned |

### 3. 실행
```bash
conda activate evax
cd ~/workspace/DigitalHealthCareTeam1
bash scripts/run_chexpert.sh eval tiny     # 공개 fine-tuned 가중치로 valid AUC 확인 (GPU 1장)
bash scripts/run_chexpert.sh train tiny    # 사전학습 가중치에서 fine-tuning (GPU 4장)
EPOCHS=20 NGPU=2 bash scripts/run_chexpert.sh train small   # 환경변수로 설정 변경
```
결과/로그/체크포인트는 `~/workspace/outputs/chexpert/<mode>_<size>_<날짜>/`에 저장됩니다.
학습은 오래 걸리므로 `nohup ... &` 또는 `tmux`로 실행하세요.

Share를 통해 올린 가중치를 바로 읽으려면 경로를 지정합니다.
```bash
WEIGHTS_DIR="$HOME/workspace/updown_share" bash scripts/run_chexpert.sh eval tiny
```
가중치가 0바이트이면 실행을 중단합니다. 환경 설치는 데이터 로더와 평가 코드의 import도 확인하며,
기존 환경에서 `medpy`가 없으면 `python -m pip install "numpy<2" "MedPy==0.5.2"`로 보완합니다.

### 4. 저장된 예측으로 calibration 평가

모델 평가가 끝난 결과 폴더의 `y_gt.npy`, `y_pred.npy`를 사용합니다.
CPU에서 실행하며 모델, GPU, 가중치 또는 추가 추론이 필요하지 않습니다.
같은 평가 실행에서 생성된 두 파일을 사용해야 합니다.
```bash
conda activate evax
cd ~/workspace/DigitalHealthCareTeam1
# 아래 경로를 평가 결과 폴더로 변경 (run_chexpert.sh 로그의 output_dir)
python scripts/evaluate_calibration.py \
    --eval-dir "$HOME/workspace/outputs/chexpert/eval_tiny_1023_0811"
```

기본적으로 5개와 10개 구간을 모두 평가합니다. `--bins 5`로 한 가지 구간 수만 지정하거나
`--output-dir <경로>`로 결과 위치를 바꿀 수 있습니다. `--no-plots`는 표만 저장합니다.
NumPy, scikit-learn, matplotlib를 사용하며 `setup_env.sh`에 포함되어 있습니다.

결과는 `<평가 결과 폴더>/calibration/`에 저장됩니다.

| 파일 | 내용 |
|---|---|
| `metrics.json` | 질환별·macro AUROC, Brier, NLL, ECE, Adaptive ECE, 구간별 통계, 입력 파일 해시 |
| `metrics.csv` | 질환별·macro 지표 표 (구간 수별 행) |
| `bin_stats.csv` | 구간 경계, 샘플 수, 평균 확률, 실제 양성 비율, ECE 기여도 |
| `reliability_uniform_5.png`, `reliability_uniform_10.png` | 등간격 구간 Reliability diagram과 구간별 샘플 수 |
| `reliability_adaptive_5.png`, `reliability_adaptive_10.png` | 분위수 구간 Reliability diagram과 구간별 샘플 수 |

계산 기준:
- 입력은 `(N, 5)` 배열이며, 열 순서는 **Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion**입니다.
  EVA-X의 현재 5개 출력 순서이며, 14개 출력이나 다른 순서의 결과는 그대로 넣지 않습니다.
- 정답은 0/1, 예측은 sigmoid 양성 확률이어야 합니다. 불확실 라벨(-1), soft label, NaN, 범위 밖 확률은 오류로 처리합니다.
- 질환별 ECE = `sum(구간 샘플 수 / 전체 샘플 수 * |평균 양성 확률 - 실제 양성 비율|)`.
  다중 클래스의 최대 softmax 확률을 사용하는 ECE와 구분합니다.
- 등간격 구간은 `(하한, 상한]`이며 첫 구간은 0도 포함합니다 (scikit-learn과 동일).
  Adaptive ECE는 확률의 분위수로 경계를 정합니다. 같은 확률을 나누지 않으므로 동점이 많으면 유효 구간 수가 줄어듭니다.
  빈 구간은 지표에서 가중치가 0이고, 통계값은 JSON에서 `null`입니다.
- Brier는 `mean((p-y)^2)`, NLL은 이진 음의 로그우도(자연로그)입니다.
  NLL만 float64에서 확률을 `[1e-7, 1-1e-7]`로 제한하며, `--eps`로 바꿀 수 있습니다.
  저장된 확률의 반올림 때문에 원본 logits에서 계산한 BCE loss와 조금 다를 수 있습니다.
- 모든 지표는 이미지 단위입니다. macro는 질환별 지표의 단순 평균입니다.
  양성 또는 음성이 전혀 없는 질환의 AUROC는 `null`/빈 CSV 셀로 기록하고 macro AUROC에서 제외하며, 유효 질환 수도 JSON에 남깁니다.
- Reliability diagram은 x축이 평균 양성 예측 확률, y축이 실제 양성 비율입니다.
  대각선 아래는 질환 위험을 높게 예측한 구간, 위는 낮게 예측한 구간입니다.
  Brier/NLL은 판별력도 반영하므로 ECE와 그래프를 함께 확인합니다.
- 이 단계는 **TS 적용 전**의 측정입니다. 보정용 데이터 분리, Temperature Scaling, 환자 단위 bootstrap 신뢰구간은 별도 단계입니다.

구간 정의와 확률 지표 참고: [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html).
데이터, 예측 배열, 생성된 결과물은 Git에 올리지 않습니다.

로컬 계산·입출력 테스트 (학습 실행 없음):
```bash
python -m unittest discover -s tests -v
```

## 서버 환경
- GPU: NVIDIA L4 (24GB) × 4
- pip: 내부 미러가 기본 설정되어 있음 (`pip install` 그대로 사용)
- torch는 PyPI 미러에서 설치 (`torch-download` 미러는 일부 버전이 외부 주소로 연결되어 실패함)
- GitHub: `https://github.com` → `http://github.snuhub-cache-server`

## 주의
- **Public 저장소입니다.** 환자 데이터, 이미지, csv, 결과물, 가중치, 계정 정보, API 키를 올리지 마세요.
- `.gitignore`로 막아두었지만 `git add` 전에 한 번 확인해 주세요.
