# Tiny calibration 전체 실험 실행

## 현재 범위

이 환경에서는 새 실험의 실제 데이터 학습을 실행하지 않았습니다. 아래 절차는 병원 서버에서 실행합니다. 실험 스크립트 자체는 Git commit/push를 수행하지 않습니다.

PPT 제안 실험을 한 번에 순차 실행합니다. **seed는 사용자 요청대로 0 하나**, 독립 조건 13개와 dev로 선택한 조합 1개, 최대 **학습 14회**입니다. 전체 Cartesian product는 아닙니다.

| 축 | 조건 | 고정 설정 |
|---|---|---|
| Baseline | Full + BCE + 원본 soft U | patch embedding은 원본처럼 고정 |
| C | Head / 마지막 2개 block / LoRA | Head는 classifier만, partial·LoRA는 classifier와 pooling norm도 학습 |
| B | U-Ones / U-Zeros / U-Ignore / Soft U | soft U는 불확실 라벨에만 [0.55,0.85]의 재현 가능한 이미지·질환별 target |
| A | Label smoothing / Focal / MDCA / Mixup / SPA+LPR | 각각 epsilon .05 / gamma 2 / beta 1 / alpha .2 / 아래 정의 |
| 조합 | C+B+A에서 각각 선택한 방법의 조합 | test는 사용하지 않음; baseline도 각 축 후보에 포함 |

원본 soft U는 Atelectasis·Edema에 [0.7,0.8], 나머지에 [0.2,0.3]의 seed 고정 per-class target입니다. 미언급은 0입니다. U-Ignore만 불확실 원소를 학습 loss에서 제외합니다.

PPT 5장의 U-SelfTrained·U-MultiClass는 기존 CheXpert 연구 소개로, 10·12장의 제안 실험에 포함되지 않았습니다. 선택 사항인 MIMIC-CXR·ChestX-ray14 외부 검증은 데이터와 사전학습 중복 확인이 필요하므로 이번 실행에 포함하지 않았습니다. 여러 seed 검증도 시간 제약에 따른 사용자 요청으로 제외했습니다.

## 기본 학습 예산

GPU 4개 × batch 256 = global batch 1,024, accumulation 1. AdamW, 고정 LR .001, layer decay .55, weight decay .05, drop path .2, gradient clipping 비활성입니다. 입력 224, Resize·CenterCrop·정규화는 기존 runner와 같습니다.

**최대 10 true epochs**, 최소 5 epoch 이후 dev mAUROC가 2회 연속 0.0001 넘게 개선되지 않으면 종료합니다. 체크포인트는 모든 관측 epoch 중 dev mAUROC 최고점(동률이면 dev NLL 최소)입니다. 작은 NLL 개선이 모델 선택을 바꾸더라도 AUROC plateau 조기 종료 기준은 별개입니다. test·calibration은 학습 중 모델 선택에 사용하지 않습니다.

이전 batch 128에서 약 9분/epoch였다는 관측만 있습니다. 14개 모두 10 epoch라면 같은 속도 기준 **학습만 약 21시간**이고 추론·TS·그림·bootstrap이 추가됩니다. batch 256의 실제 속도는 미측정이며, 하루 내 완료나 10 epoch 내 수렴을 보장하지 않습니다. Head·부분 학습은 더 빠를 수 있습니다. hard time limit으로 결과를 불완전하게 끊지는 않습니다.

기본 설정 파일: [`configs/tiny_study_day.json`](configs/tiny_study_day.json). 실행 중 자동으로 LR·batch·seed를 바꾸거나 test 점수로 재학습하지 않습니다. 자원 부족이나 코드 오류는 기록 후 독립된 다음 조건으로 넘어갑니다.

## 실행

서버에서 코드를 업데이트한 후 기존 `evax` 환경과 Share의 MIM 가중치를 사용합니다. 실행 시 다운로드·추가 설치·Git 변경이 없습니다.

```bash
conda activate evax
cd ~/workspace/DigitalHealthCareTeam1

# 학습 없이 14개 조건과 경로·설정을 확인
bash scripts/run_calibration_study.sh --dry-run

# tmux 안에서 한 번 실행. 폴더명을 고정하면 나중에 찾기 쉽습니다.
WEIGHTS_DIR="$HOME/workspace/updown_share" \
  bash scripts/run_calibration_study.sh \
  --output-dir "$HOME/workspace/outputs/chexpert/tiny_study_seed0"
```

데이터 기본값은 `$HOME/workspace/datasets/chexpert`, 가중치 파일명은 `eva_x_tiny_patch16_merged520k_mim.pt`입니다. `--data-root`, `--pretrained`로 명시할 수도 있습니다. 원본 weights SHA와 GPU backward 검사로 잘못된 checkpoint를 걸러냅니다.

새 실험은 output 폴더 안에 환자 분할을 만들고 원본 CSV 및 모든 이미지의 존재·0바이트 여부를 검사합니다. train/dev/calibration은 원본 train 환자를 90/5/5로 나누고, 공식 valid의 정면 202장을 primary test로 사용합니다. 모든 split의 환자 중복을 검사합니다.

실험 디렉터리에 프로세스 잠금을 걸어 같은 실행을 동시에 시작하지 못하게 합니다. GPU는 독점 예약하지 않으므로 다른 팀원의 실행과 GPU 사용을 조율해야 합니다. 실행 중 코드·설정·템플릿을 변경하거나 pull하지 마세요.

## 진행·오류·중간 결과 확인

**한 파일만 보려면 실험 폴더의 `STATUS.md`를 열어 두세요.** 약 10초마다 현재 단계·epoch·진행률·메모리, 조건별 단계 표, 완료한 조건의 주요 raw/TS 지표, 오류·재시도·보고서 생성 등 최근 주요 이벤트만 갱신합니다. step별 원본 출력은 여기에 쌓지 않습니다. 시간 표시는 UTC입니다. 파일은 마지막 snapshot이므로 프로세스가 죽으면 자동 갱신도 멈춥니다. heartbeat 시각을 확인하세요.

다른 터미널에서:

```bash
python scripts/study_status.py \
  --output-dir "$HOME/workspace/outputs/chexpert/tiny_study_seed0" --watch 10
```

전체 상태, 완료/실패/대기 단계 수, 현재 조건, epoch·step, loss, 해당 epoch 남은 시간, GPU peak memory, 최근 로그, 이미 완료한 조건의 raw/TS 지표를 보여줍니다. heartbeat가 오래되면 `STALE`을 표시합니다. heartbeat는 driver 생존 표시로, GPU 연산이 진전하고 있다는 보장은 아닙니다. epoch 진행 timestamp와 로그를 함께 확인하세요. 모니터 종료(Ctrl+C)는 학습에 영향을 주지 않습니다.

| 파일 | 내용 |
|---|---|
| `STATUS.md` | 한 파일로 보는 현재 상황·핵심 로그·중간 지표, 약 10초마다 갱신 |
| `status.json` | 단계별 running/complete/failed/blocked, 시간, 재시도, 오류, 로그 위치 |
| `pipeline.log`, `logs/<단계>.log` | 전체 및 단계별 원본 stdout/stderr |
| `runs/<조건>/seed0/train/progress.json` | rank 0의 최근 epoch·step·loss·메모리, dev 결과 |
| `runs/<조건>/seed0/train/history.jsonl` | 완료 epoch별 학습·dev 이력 |
| `report.md`, `reports/<조건>_seed0.md` | 중간 전체 표와 개별 상세 보고서 |
| `aggregate.json`, `bootstrap.json` | 구조화된 결과와 최종 환자 bootstrap |

한 조건의 preflight→학습→dev/calibration/test 추론→TS/metric 계산이 끝나면 즉시 개별 MD와 전체 진행 MD를 생성합니다. 학습 중간에는 history/progress로 dev를 확인하고, 최종 test는 학습 완료 후만 평가합니다.

오류는 기본 1회 자동 재시도합니다. 재시도해도 실패하면 원인·로그를 남기고 다음 독립 조건을 계속합니다. 실패한 조건을 성공으로 표시하지 않습니다. 모든 dev 결과가 있어야 조합을 선택하며, 누락되면 조합은 blocked로 기록됩니다. 마지막에 부분 보고서를 남기고 실패가 하나라도 있으면 exit code 1, 전부 완료하면 0입니다. SIGTERM/Ctrl+C에는 자식 torchrun 프로세스 그룹을 종료합니다.

## 재개

처음과 같은 코드·설정으로 **같은 output 폴더만** 지정합니다. 기존 manifest에 저장된 데이터·가중치·설정을 자동 재사용합니다.

```bash
bash scripts/run_calibration_study.sh \
  --output-dir "$HOME/workspace/outputs/chexpert/tiny_study_seed0"
```

완료 단계는 산출물의 SHA-256을 검증하고 건너뜁니다. 학습 중단은 마지막 완료 epoch의 optimizer·scaler·rank별 RNG에서 재개합니다. 미완료 epoch의 진행은 다시 실행됩니다. 첫 checkpoint 이전에 실패한 폴더와 미완료 평가 폴더는 별도 이름으로 보존합니다. 코드·모델 정의·가중치·설정·템플릿이 달라진 경우 새 output 폴더를 사용해야 합니다. 이미 완료한 산출물을 고치면 조용히 덮지 않고 오류로 알립니다.

## TS·평가 프로토콜

각 조건에 raw / 공통 T / 질환별 T 3종을 평가합니다. T는 calibration 라벨의 masked binary NLL로 fitting하며 범위 [.01,100], single-class/미관측 질환은 T=1로 fallback합니다. fit NLL이 identity보다 나쁘면 T=1입니다. fit 결과·표본 수·온도가 bound에 닿았는지도 저장합니다.

1. **Primary `report_holdout`**: 원본 train에서 분리한 calibration 환자로 T를 fit, 전문가 라벨 202장 전체에서 평가합니다. dev/calibration의 명시적 -1은 제외하고 미언급은 0으로 처리합니다. calibration 라벨 노이즈를 고려해야 합니다.
2. **Secondary `expert_subset`**: 공식 valid 환자를 split seed 42로 50/50 분리합니다. 한쪽 전문가 라벨로 fit하고 나머지 환자만 평가합니다. 같은 분할·평가 영상이 모든 조건에 적용됩니다. 이 프로토콜의 raw와 두 TS 결과끼리 비교하며, primary 202장 결과와 직접 비교하지 않습니다. expert fitting subset은 이 보조 프로토콜의 test가 아닙니다.

모든 설정에서 5개 질환별·macro AUROC/Brier/NLL/ECE@5/10/Adaptive ECE@5/10, 구간 통계와 reliability diagram을 저장합니다. 환자 단위 bootstrap 500회, 95% percentile CI는 전체 실행 종료 후 계산하고 MD에 추가합니다. 동일 TS를 적용한 baseline 대비 paired delta와 같은 모델의 raw 대비 TS delta도 JSON에 저장합니다. CI는 고정 학습 모델·고정 fitting 온도에 조건부인 test 환자 표본의 불확실성입니다. seed 차이나 fitting 불확실성은 추정하지 않습니다. 한 bootstrap draw에서 어느 질환의 AUROC가 정의되지 않으면 그 질환 및 5질환 macro AUROC의 해당 draw를 제외하고 유효 반복 수를 기록합니다.

## 조합 선택

C, B, A 각각 baseline을 포함한 후보 중 **dev mAUROC가 baseline − 0.005 이상**인 조건만 허용하고 dev NLL 최소 조건을 선택합니다. 동률이면 dev AUROC, ID 순입니다. 고정한 규칙의 탐색적 조합이며, 효과 중복이 없다는 보장은 없습니다. 선택 근거와 입력 dev 파일 해시는 `combination_selection.json`에 기록합니다. 이 규칙은 test 결과를 읽지 않습니다. test 점수에 따른 질환별 처리 정책 탐색이나 자동 hyperparameter 탐색은 하지 않습니다.

## 구현 정의·출처

- **BCE/label smoothing**: 다중 라벨 Bernoulli target을 사용합니다. softmax를 적용하지 않습니다. [Label smoothing](https://arxiv.org/abs/1906.02629).
- **Focal**: 양성 `(1-p)^gamma * softplus(-z)`, 음성 `p^gamma * softplus(z)`의 target 가중합, gamma=2, 별도 alpha 없음. soft U에도 같은 partial-loss 확장을 적용합니다. [Focal Loss](https://arxiv.org/abs/1708.02002).
- **MDCA**: BCE + beta × 질환별 `abs(mean(sigmoid(z)) − mean(target))` 평균. U-Ignore mask를 반영합니다. 각 GPU의 microbatch에서 계산해 DDP 평균하는 **multi-label adaptation**입니다. global batch MDCA나 원본 softmax 구현과 같다고 주장하지 않습니다. [MDCA 원 연구](https://mdca-loss.github.io/).
- **Mixup**: Beta(.2,.2), 이미지와 Bernoulli target을 함께 혼합합니다. U-Ignore 조합에서는 유효 target mass를 섞고 그 mass로 loss를 가중합니다. [Mixup](https://arxiv.org/abs/1710.09412).
- **LoRA**: EVA attention의 QKV(또는 Q/K/V)와 output projection, rank 8·alpha 16. 원본 코드의 QKV 경로는 module forward를 호출하지 않고 `F.linear(..., weight)`를 사용하므로 합성 weight에도 adapter를 반영합니다. A random/B zero 초기화, adapter dropout 없음. [LoRA](https://arxiv.org/abs/2106.09685).
- **SPA+LPR**: [Cheng & Vasconcelos, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_Towards_Calibrated_Multi-label_Deep_Neural_Networks_CVPR_2024_paper.html)의 Eq.16/19/20을 독립 구현했습니다. positive loss는 `softplus(-v)`, negative loss는 `softplus(3*(v-1))/5`; LPR weight .1은 이번 탐색의 고정값입니다. 확률을 단순 sigmoid(v)로 내보내지 않습니다. Eq.16의 conditional-risk derivative에서 도출한 증가 inverse link의 log odds `log(3/5)+logsigmoid(3*(v-1))-logsigmoid(-v)`를 사용합니다. **인쇄본 Eq.17의 sigmoid 부호는 Eq.16의 미분 및 증가 그래프와 일치하지 않아 그대로 복사하지 않았습니다.** BCE로의 환원·단조성·conditional-risk stationarity를 수치 테스트합니다. 불확실 soft target에는 두 partial loss의 기대값, LPR에는 독립 Bernoulli pair의 기대값을 적용한 확장입니다. 원 논문의 공개 수치 재현을 검증한 구현은 아닙니다.

## docs 자동 정리

기본 경로는 `docs/experiments/generated/<output 폴더명>/`입니다. `--docs-dir <경로>`로 바꿀 수 있습니다. 각 조건 완료 직후 `<조건>_seed0.md`, 전체 표 `README.md`를 갱신하고 전체 종료 시 CI를 추가합니다. 해당 실행이 생성한 `<!-- calibration-study: generated -->` 문서만 갱신하므로 팀원이 직접 쓴 MD를 덮어쓰지 않습니다.

```bash
# 팀 템플릿 적용: 학습 시작 전에 지정
WEIGHTS_DIR="$HOME/workspace/updown_share" \
  bash scripts/run_calibration_study.sh \
  --report-template docs/experiments/_template.md \
  --output-dir "$HOME/workspace/outputs/chexpert/tiny_study_seed0"

# 저장된 결과만으로 전체 보고서·CI 다시 계산 (GPU 불필요)
python scripts/study_report.py \
  --output-dir "$HOME/workspace/outputs/chexpert/tiny_study_seed0"
```

템플릿은 `title`, `setup`, `metrics`, `interpretation`, `limitations`, `artifacts` 여섯 `{{placeholder}}`를 포함해야 합니다. 팀 문서 형식이 올라오면 이 렌더링 단계만 맞추면 됩니다. 생성 MD에는 지표·학습 조건·표본 수만 담고 환자 ID, 이미지 경로, 원시 배열, 가중치를 넣지 않습니다. 실행 실패의 원본 traceback은 서버 로그에만 남습니다. MD 파일을 Git에 공유하는 commit/push는 별도 수동 단계입니다.

과거 완료 결과: [`docs/experiments/`](docs/experiments/README.md). 새 runner는 공개 가중치 평가나 과거 5-epoch 결과를 현재 baseline으로 섞지 않습니다.

## 로컬 검증

```bash
python -m unittest discover -s tests -v
RUN_DISTRIBUTED_TESTS=1 python -m unittest discover -s tests -p test_tiny_distributed.py -v
```

synthetic toy 학습으로 optimizer/resume/DDP를, 실제 Tiny architecture의 synthetic 이미지 backward로 scope·LoRA를 검사합니다. 실제 환자 데이터에 대한 학습이나 병원 GPU 처리량 검증을 대체하지 않습니다.
