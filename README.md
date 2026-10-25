# DigitalHealthCareTeam1

PPT 전체 실험 자동 실행·상태 확인·재개·docs 자동 보고서: [STUDY_GUIDE.md](STUDY_GUIDE.md).
이미 완료한 실험 내용과 결과: [docs/experiments](docs/experiments/README.md).

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

## Tiny 재학습: BCE vs label smoothing

### 한 명령으로 실험부터 결과 정리까지

conda `evax` 환경에서 **학습 시작용 Tiny MIM 가중치**를 준비한 뒤 실행합니다.
Share에 업로드했다면:
```bash
WEIGHTS_DIR="$HOME/workspace/updown_share" bash scripts/run_tiny_comparison.sh
```

환자 분할·이미지 검사 → 두 loss의 작은 forward/backward 검사 → BCE 학습 → 공식 test 평가 →
label smoothing 학습 → 공식 test 평가 → calibration 지표·그림 → 비교 보고서까지 자동으로 실행합니다.
두 조건은 미리 고정하며, test 결과로 학습 조건이나 체크포인트를 바꾸지 않습니다.
기본적으로 GPU 4개를 사용해 두 학습을 **순차 실행**합니다. 모델 선택에는 dev만 사용합니다.

예: 두 조건 모두 GPU 1개, 실제 2 epoch로 먼저 실험하려면:
```bash
WEIGHTS_DIR="$HOME/workspace/updown_share" NGPU=1 EPOCHS=2 bash scripts/run_tiny_comparison.sh
```

기본 결과 위치: `~/workspace/outputs/chexpert/tiny_comparison_<날짜>/`
- `experiment.json`, `pipeline.log`: 고정된 실험 설정과 전체 실행 로그
- `bce/`, `label_smoothing/`: 각 조건의 best/last 체크포인트, dev 학습 이력, `test/`의 예측과 지표·그림
- `comparison/report.md`: 질환별·평균 결과와 후보−baseline 차이를 정리한 보고서
- `comparison/comparison.csv`, `deltas.csv`, `comparison.json`: 원본 지표와 차이 표
- `comparison/comparison.png`: 질환별 AUROC·Brier·NLL·ECE·Adaptive ECE 비교 그래프

중간에 중단됐다면 **처음과 같은 환경변수**에 `--output-dir`로 원래 실험 폴더를 지정해 재개합니다.
완료 단계의 해시를 확인해 건너뛰며, 중단된 학습은 마지막으로 완료된 epoch부터 재개합니다.
불완전한 평가 출력은 보존한 후 새로 평가합니다.
```bash
WEIGHTS_DIR="$HOME/workspace/updown_share" bash scripts/run_tiny_comparison.sh \
    --output-dir "$HOME/workspace/outputs/chexpert/실제_tiny_comparison_폴더명"
```

학습 없이 실행 계획만 확인하려면 `bash scripts/run_tiny_comparison.sh --dry-run`을 사용합니다.
비교 보고서는 point estimate를 정리하며, 개선을 보장하거나 통계적 유의성을 자동으로 선언하지 않습니다.
TS fitting은 별도 단계입니다.

### 개별 단계와 계산 기준

새 실험은 공개 **MIM 사전학습 가중치**에서 시작하며, CheXpert fine-tuned 가중치로 시작하지 않습니다.
EVA-X 서브모듈은 수정하지 않고 모델 정의만 재사용합니다.
동일한 데이터 분할·초기화·불확실 라벨·optimizer 조건에서 두 loss를 비교합니다.

| 조건 | 학습 목표 |
|---|---|
| `bce` | `BCEWithLogitsLoss(logits, target)` |
| `label_smoothing` | `target' = 0.95 * target + 0.025` 후 동일한 BCE (기본 smoothing=0.05) |

다중 라벨이므로 각 질환의 Bernoulli target을 0.5 방향으로 부드럽게 합니다.
불확실 라벨(-1)은 두 조건 모두 원본 Tiny 방식과 같은 범위의 고정된 per-class soft target을 사용합니다:
Atelectasis/Edema는 `[0.7, 0.8]`, 나머지는 `[0.2, 0.3]`에서 seed로 한 번 정하며, 미언급 값은 0입니다.
baseline에도 이미 불확실 라벨의 soft target이 있으므로, 실험 변수는 **전체 학습 target에 추가하는 smoothing**입니다.
개선은 가설이며 AUROC·ECE·Brier·NLL로 확인합니다. [Label smoothing 연구](https://arxiv.org/abs/1906.02629).

### 1. 환자 단위 분할 준비 (최초 1회)

```bash
conda activate evax
cd ~/workspace/DigitalHealthCareTeam1
bash scripts/run_tiny_calibration.sh prepare
```

기본적으로 train.csv의 정면 영상 환자를 seed 42로 섞어 train 약 90%, dev 약 5%, calibration 약 5%로 나눕니다.
공식 valid.csv의 정면 영상 202장은 `test.csv`로 별도 보관하며, 네 split 사이 환자 중복을 검사합니다.
모든 이미지 존재 여부·크기를 검사하고 CSV 해시·질환별 라벨 수를 `manifest.json`에 기록합니다.
기본 분할 위치는 `~/workspace/outputs/chexpert/splits_seed42/`이며 `SPLITS_DIR`로 바꿀 수 있습니다.
같은 설정의 재실행은 기존 분할을 검증해 재사용합니다. 설정이 달라지면 새 디렉터리를 사용해야 합니다.

dev는 **학습 중 체크포인트 선택용**, calibration은 **향후 TS 보정용**, test는 **최종 평가용**입니다.
dev/calibration도 보고서 기반 학습 라벨이며, -1은 지표에서 제외하고 미언급 값은 학습과 동일하게 0으로 처리합니다.
이 라벨 품질과 전문가 라벨인 공식 test의 차이는 해석 시 고려해야 합니다.

### 2. 가중치와 작은 GPU 검사

필요 파일: `eva_x_tiny_patch16_merged520k_mim.pt` (79,513,863바이트).
공식 SHA-256은 `f82d6fc450a83ac7c4d32fa9f0d6459ea7a53fd08e47fdef0cb12ce35c97174d`입니다.
로더가 이 해시를 확인하여 빈 파일·불완전한 업로드·fine-tuned 가중치 사용을 막습니다.
기본 위치는 `~/workspace/pretrained/`이며 Share에 있으면 `WEIGHTS_DIR="$HOME/workspace/updown_share"`를 앞에 붙입니다.

```bash
bash scripts/run_tiny_calibration.sh check bce
bash scripts/run_tiny_calibration.sh check label_smoothing
```

실제 이미지 2장의 forward/backward와 모든 학습 파라미터의 유한한 gradient를 검사합니다.
optimizer step은 실행하지 않습니다. 두 조건 모두 `CHECK OK`를 확인한 후 학습합니다.

### 3. 두 조건을 순서대로 학습 (병원 서버에서만)

```bash
bash scripts/run_tiny_calibration.sh train bce
bash scripts/run_tiny_calibration.sh train label_smoothing
```

기본값: GPU 4개, GPU당 batch 128, **실제 5 epoch**, seed 0, AdamW, 고정 LR 1e-3,
layer decay 0.55, weight decay 0.05, drop path 0.2, clip norm 1.0입니다.
augmentation은 두 조건 모두 원본 Tiny와 같은 Resize(224)·CenterCrop(224)·정규화를 사용합니다.
원본이 gradient를 끊는 patch embedding은 명시적으로 freeze하여 DDP의 미사용 gradient 문제를 피합니다.
`EPOCHS`, `NGPU`, `BATCH_SIZE`, `NUM_WORKERS`, `ACCUMULATION`, `LR`, `SEED`, `SMOOTHING`으로 변경할 수 있습니다.
공정한 비교를 위해 smoothing을 제외한 설정은 두 조건에 동일하게 적용합니다.

여기서는 `CheXpert.__len__() // 20` 방식의 fake epoch를 사용하지 않습니다.
epoch마다 전체 train split을 순회합니다 (분산 sampler의 끝부분 padding은 있을 수 있음).
공개 체크포인트와 분할·학습 조건이 다르므로, 공개 Tiny 점수와 새 baseline의 차이를 loss 효과로 해석하지 않습니다.

결과는 `~/workspace/outputs/chexpert/train_tiny_<loss>_seed0_<날짜>/`에 저장됩니다.
- `config.json`: loss·학습 조건·초기 가중치 및 split 해시
- `history.jsonl`: 실제 epoch별 train loss와 dev AUROC·NLL
- `best.pth`: dev AUROC 최대 체크포인트 (동률이면 관측된 이진 라벨의 NLL 최소)
- `last.pth`: optimizer·AMP scaler·rank별 RNG 상태를 포함한 epoch 경계 재개용 체크포인트
- `run.log`: 실행 로그

재개 예시 (원래와 같은 GPU 수·batch·epoch 등 설정 사용):
```bash
RESUME="$HOME/workspace/outputs/chexpert/실제_학습폴더/last.pth" \
    bash scripts/run_tiny_calibration.sh train bce
```

### 4. 학습이 끝난 모델의 공식 test 평가

학습 중에는 공식 test나 calibration에서 모델을 선택하지 않습니다.
모델 선택이 끝난 뒤 각 조건의 best.pth를 한 번씩 평가합니다.
```bash
CHECKPOINT="$HOME/workspace/outputs/chexpert/실제_학습폴더/best.pth" \
    bash scripts/run_tiny_calibration.sh evaluate
```

`eval_test_<날짜>/`에 logits·정답·확률·label mask·샘플 순서·환자 ID·평가 메타데이터를 저장하고,
기존 calibration 평가 스크립트로 지표·그림도 생성합니다. 향후 환자 단위 bootstrap에 ID를 사용할 수 있습니다.
`EVAL_SPLIT=dev` 또는 `EVAL_SPLIT=calibration`으로 학습 라벨 기반 예측을 별도로 저장할 수 있지만,
관측되지 않은 라벨이 포함되므로 `label_mask.npy`를 반영해야 합니다.
기존 완전한 0/1 라벨용 calibration CLI는 이런 배열을 조용히 채점하지 않고 오류로 중단합니다.
TS fitting은 이번 구현에 포함하지 않습니다.

## 서버 환경
- GPU: NVIDIA L4 (24GB) × 4
- pip: 내부 미러가 기본 설정되어 있음 (`pip install` 그대로 사용)
- torch는 PyPI 미러에서 설치 (`torch-download` 미러는 일부 버전이 외부 주소로 연결되어 실패함)
- GitHub: `https://github.com` → `http://github.snuhub-cache-server`

## 주의
- `docs/experiments/`에는 실험 조건과 집계 지표의 Markdown 보고서를 공유합니다. 원시 데이터·환자 ID·이미지 경로·예측 배열·가중치는 제외합니다.
- **Public 저장소입니다.** 환자 데이터, 이미지, csv, 결과물, 가중치, 계정 정보, API 키를 올리지 마세요.
- `.gitignore`로 막아두었지만 `git add` 전에 한 번 확인해 주세요.
