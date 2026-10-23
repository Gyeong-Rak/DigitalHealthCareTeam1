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
conda 환경 `evax`(python 3.10, torch 2.0.1+cu118, xformers 0.0.21, timm 0.4.12)를 만들고 마지막에 GPU 동작을 확인합니다.
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

## 서버 환경
- GPU: NVIDIA L4 (24GB) × 4
- pip: 내부 미러가 기본 설정되어 있음 (`pip install` 그대로 사용)
- 특정 CUDA 버전의 torch: `--index-url http://torch-download.snuhub-cache-server/whl/cu118`
- GitHub: `https://github.com` → `http://github.snuhub-cache-server`

## 주의
- **Public 저장소입니다.** 환자 데이터, 이미지, csv, 결과물, 가중치, 계정 정보, API 키를 올리지 마세요.
- `.gitignore`로 막아두었지만 `git add` 전에 한 번 확인해 주세요.
