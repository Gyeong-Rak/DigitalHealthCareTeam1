# 결과 설명·검증 실험 계획

작성: 2026-10-27. 기존 Tiny calibration 연구([README](README.md))의 결과를 **설명**하고 **검증**하기 위한 추가 실험입니다.
아래 판정 기준은 결과를 보기 전에 정했습니다.

## 질문

1. **왜 주 평가(report_holdout)에서는 개선이 없고, 전문가 보정 보조 평가(expert_subset)에서만 개선이 보였나?**
   공개 Tiny 평가 로그에서 질환별 유병률이 판독문 라벨 train과 전문가 라벨 valid 사이에 크게 다릅니다
   (예: Cardiomegaly 12.2% → 32.7%, Consolidation 6.8% → 15.8%, Edema 32.2% → 20.8%).
   가설: 모델 확률은 판독문 라벨의 유병률에 맞춰져 있고, 질환별 TS는 기울기만 바꾸므로 **절편(유병률) 차이를 옮기지 못한다.**
2. **Mixup+TS(AUROC·ECE 동시 개선)와 LoRA+TS(ECE 최저)는 seed를 바꿔도 재현되는가?**

## 실험 1 — 사후 보정 비교와 유병률 분석 (학습 없음, CPU)

결과: [2026-10-27_calibration_shift_analysis.md](2026-10-27_calibration_shift_analysis.md) (seed 0, 완료)

`scripts/analyze_calibration_shift.py`. 저장된 logits만 사용하며 기존 TS와 **같은 fitting 데이터·같은 평가 데이터**를 씁니다.

| 보정 | 식 (질환별) | fitting |
|---|---|---|
| ts | sigmoid(z / T) | 기존과 동일 (수치 재현 확인) |
| bias | sigmoid(z + b) | 절편만 |
| platt | sigmoid(a·z + b) | 기울기 + 절편 |
| prior | sigmoid(z + logit(π_fit) − logit(π_report)) | 라벨 없이 유병률만 사용, expert_subset에만 해당 |

- 모든 fitted 보정은 fitting 데이터의 NLL이 나빠지면 항등 변환으로 되돌립니다(기존 TS 규칙과 동일).
- 단조 변환이므로 AUROC는 바뀌지 않습니다. 비교 지표는 Brier·NLL·ECE@10·Adaptive ECE@10입니다.
- 산출: 유병률 표, 14조건 × 보정 방법 표, baseline의 질환별 (평균 예측 − 유병률), 전체 조건 평균 Δ(보정 − TS)와 환자 bootstrap 95% CI.

**판정 기준**
- 가설 1 지지: expert_subset에서 bias 또는 platt의 ΔECE@10(vs TS) 95% CI 상한 < 0, **그리고** baseline TS의 질환별
  |평균 예측 − 유병률|이 유병률 변화가 큰 질환(Cardiomegaly, Consolidation)에서 가장 큼.
- report_holdout에서 bias/platt가 TS와 비슷하면(CI가 0 포함) "fitting 데이터의 라벨 분포가 핵심"이라는 해석과 일치.

## 실험 2 — Seed 반복 (서버 GPU 4장, 약 9~10시간)

`scripts/run_seed_replication.py`. baseline, a_mixup, c_lora를 **seed 1, 2**로 다시 학습합니다(6회).

- split, 조건 정의, 학습 설정, 사전학습 가중치, 수치 코드(`train_tiny_calibration.py` 등)를 원래 연구 폴더에서 가져와 **해시로 동일성 확인**합니다. seed만 다릅니다.
- seed가 바뀌면 초기화·데이터 순서·원본 soft U target이 함께 바뀝니다(원래 학습 코드의 seed 정의).
- seed 우선 순서(seed 1의 세 조건 → seed 2)로 실행하여, 중간에 멈춰도 같은 seed의 baseline 비교가 가능합니다.
- 팀원 코드는 수정하지 않았습니다. `Study`의 실행·재개·산출물 해시 기능을 재사용합니다.

**판정 기준** (seed 0·1·2, 같은 seed baseline 대비 차이의 평균)
- Mixup 재현: expert_subset·TS에서 평균 ΔAUROC > 0 **그리고** 평균 ΔECE@10 < 0, 세 seed 중 2개 이상에서 같은 방향.
- LoRA 재현: expert_subset·TS에서 평균 ΔECE@10 < 0, 세 seed 중 2개 이상에서 같은 방향.
- "통계적으로 확인된 개선"은 위에 더해 seed 평균 Δ의 환자 bootstrap 95% CI가 0을 포함하지 않을 때만 주장합니다.
  CI는 평가 환자 표본의 불확실성만 반영하며 seed는 고정된 3개로 취급합니다(seed 간 SD는 별도 표기).
- 주 평가(report_holdout) 결과도 같은 형식으로 함께 보고합니다.

## 실행 (서버)

```bash
conda activate evax
cd ~/workspace/DigitalHealthCareTeam1 && git pull
OUT=~/workspace/outputs/chexpert

# 실험 1: 기존 결과만으로 분석 (CPU 몇 분)
python scripts/analyze_calibration_shift.py \
  --study-dir $OUT/tiny_study_seed0 --study-dir $OUT/tiny_study_lora_mixup_ignore \
  --output-dir $OUT/calibration_shift_seed0

# 실험 2: seed 반복 (GPU 4장 사용 — 다른 학습과 겹치지 않게 조율)
python scripts/run_seed_replication.py --dry-run
nohup python scripts/run_seed_replication.py > $OUT/tiny_seed_replication.launch.log 2>&1 < /dev/null &
cat $OUT/tiny_seed_replication/STATUS.md        # 진행 상황

# 실험 2 완료 후: seed 포함 분석
python scripts/analyze_calibration_shift.py \
  --study-dir $OUT/tiny_study_seed0 --study-dir $OUT/tiny_seed_replication \
  --output-dir $OUT/calibration_shift_seeds
```

- 중단 후에는 같은 명령으로 재개합니다. 실행 중에는 `git pull`이나 코드 수정을 하지 않습니다(코드 해시 검사로 중단됨).
- 결과: `<output-dir>/report.md`(요약 표), `analysis.json`(전체 수치). 집계 지표만 담고 환자 ID·경로·예측값은 없습니다.
- 로컬 검증: `python -m unittest discover -s tests -p "test_calibration_shift.py"` (합성 데이터, 11개).
