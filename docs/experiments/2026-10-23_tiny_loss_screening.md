# Tiny 재학습 — BCE vs label smoothing

## 실험 내용

- 목적: 동일한 학습 조건에서 label smoothing이 AUROC를 유지하면서 calibration을 개선하는지 탐색.
- 시작점: 공개 Tiny **MIM 사전학습** 가중치. CheXpert fine-tuned 체크포인트에서 시작하지 않음.
- 학습 범위: patch embedding 고정, 나머지 백본·분류 헤드 학습.
- 환자 분할: train 172,066장 / 58,082명, dev 9,152장 / 3,226명, calibration 9,809장 / 3,226명, test 202장 / 200명. 네 split 환자 중복 없음.
- seed 0, 실제 전체 순회 5 epoch, GPU 4개 × batch 128 = 총 batch 512, accumulation 1.
- AdamW, 고정 LR 0.001, layer decay 0.55, weight decay 0.05, drop path 0.2, gradient clip norm 1.0.
- 불확실 라벨: Atelectasis/Edema는 per-class `[0.7,0.8]`, 나머지는 `[0.2,0.3]`의 고정 soft target. 미언급은 0. 두 조건에서 같은 seed·정책 사용.
- BCE: `BCEWithLogitsLoss(z, y)`. Label smoothing: `y' = 0.95*y + 0.025` 후 같은 BCE.
- best 체크포인트는 dev AUROC(동률 시 dev NLL)로 선택. test는 선택에 사용하지 않음.
- TS 적용 없음. 결과는 저장된 test 예측으로 계산.

## 결과

| 지표 | BCE | Label smoothing | 차이 (smoothing − BCE) | 좋은 방향 |
|---|---:|---:|---:|---|
| mAUROC | 0.880511 | 0.880569 | 0.000058 | 높음 |
| Brier | 0.134473 | 0.133090 | -0.001383 | 낮음 |
| NLL | 0.425243 | 0.416338 | -0.008906 | 낮음 |
| ECE@5 | 0.094718 | 0.088163 | -0.006555 | 낮음 |
| Adaptive ECE@5 | 0.100306 | 0.099239 | -0.001067 | 낮음 |
| ECE@10 | 0.110574 | 0.115392 | 0.004818 | 낮음 |
| Adaptive ECE@10 | 0.112625 | 0.111854 | -0.000771 | 낮음 |

차이는 서버 비교 보고서에 출력된 값을 그대로 기록했습니다. 반올림된 두 표시값의 차이와 마지막 자리가 다를 수 있습니다.

제공된 BCE 질환별 결과:

| 질환 | AUROC | Brier | NLL | ECE@5 | AECE@5 | ECE@10 | AECE@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Atelectasis | 0.850919 | 0.171433 | 0.501952 | 0.089656 | 0.096727 | 0.120622 | 0.138296 |
| Cardiomegaly | 0.780804 | 0.208145 | 0.694013 | 0.206473 | 0.206473 | 0.206473 | 0.206473 |
| Consolidation | 0.905147 | 0.110884 | 0.334148 | 0.069793 | 0.103212 | 0.089098 | 0.103212 |
| Edema | 0.936905 | 0.081104 | 0.273934 | 0.054231 | 0.042982 | 0.071610 | 0.060998 |
| Pleural Effusion | 0.928782 | 0.100798 | 0.322168 | 0.053438 | 0.052136 | 0.065067 | 0.054144 |

Label smoothing의 질환별 원본 표는 이 채팅에 제공되지 않아 추가하지 않았습니다. 서버 원본 보고서에서 보완할 수 있습니다.

## 해석

이번 실험에서 label smoothing은 BCE 대비 AUROC를 거의 그대로 유지하면서 Brier·NLL을 개선했습니다. ECE@5는 개선됐지만 ECE@10은 악화되어, calibration 전반이 일관되게 좋아졌다고 주장할 수 없습니다.

공개 Tiny 평가 mAUROC 0.895816 대비 새 BCE는 약 **1.53 percentage points** 낮습니다. 데이터 분할·배치·샘플링·학습 일정·gradient clipping 등이 달라 원인은 분리되지 않았습니다. 이번 결과는 **자체 BCE 대비** 비교이며 공식 성능을 유지했다는 증거는 아닙니다.

## 한계

- seed 1개, test 202장. Bootstrap CI·통계적 유의성 검증 없음.
- BCE dev AUROC가 5 epoch까지 상승했으므로 충분히 수렴했다고 단정할 수 없음.
- TS 전 결과이며 TS 후에도 개선이 남는지 확인하지 않음.
- 기록은 사용자 제공 서버 로그 기준이며, 로그에 없는 값은 추정하지 않음.

## 산출물

서버 `tiny_comparison_20261023_095420/comparison/report.md`에 자동 비교 보고서가 있습니다. 조건별 `test/calibration/`에 지표·reliability diagram이 있습니다. 데이터·원시 예측·체크포인트는 Git에 포함하지 않습니다.
