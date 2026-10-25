# 공개 EVA-X Tiny·Small CheXpert 가중치 평가

## 실험 내용

- 목적: 공개 체크포인트의 판별력을 확인하고 calibration 문제를 측정.
- 가중치: `MapleF/eva_x`의 CheXpert fine-tuned Tiny·Small. 이번 평가에서 추가 학습·TS fitting 없음.
- 평가: CheXpert 공식 valid의 정면 이미지 202장, 환자 200명.
- 출력 순서: Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion.
- 지표: 질환별 양성 확률의 AUROC, Brier, 이진 NLL(자연로그), ECE·Adaptive ECE(bin 5/10). macro는 질환별 단순 평균.

## 결과

| 모델 | mAUROC | Brier | NLL | ECE@5 | AECE@5 | ECE@10 | AECE@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tiny | 0.895816 | 0.139745 | 0.442914 | 0.110588 | 0.121549 | 0.127809 | 0.128855 |
| Small | 0.901189 | 0.135177 | 0.432948 | 0.100122 | 0.106338 | 0.108714 | 0.109396 |

| 모델 | 질환 | AUROC | Brier | NLL | ECE@5 | AECE@5 | ECE@10 | AECE@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Tiny | Atelectasis | 0.860945 | 0.181008 | 0.519096 | 0.130120 | 0.150942 | 0.140025 | 0.150942 |
| Tiny | Cardiomegaly | 0.814060 | 0.229931 | 0.781427 | 0.239383 | 0.239383 | 0.239383 | 0.239383 |
| Tiny | Consolidation | 0.924632 | 0.109238 | 0.327315 | 0.071617 | 0.103657 | 0.109327 | 0.114166 |
| Tiny | Edema | 0.948512 | 0.077757 | 0.264491 | 0.063249 | 0.062224 | 0.067097 | 0.062224 |
| Tiny | Pleural Effusion | 0.930933 | 0.100790 | 0.322241 | 0.048570 | 0.051541 | 0.083213 | 0.077559 |
| Small | Atelectasis | 0.845984 | 0.162949 | 0.479377 | 0.080788 | 0.077889 | 0.093815 | 0.082786 |
| Small | Cardiomegaly | 0.838625 | 0.210291 | 0.725597 | 0.221508 | 0.221508 | 0.222615 | 0.221508 |
| Small | Consolidation | 0.943750 | 0.125969 | 0.389157 | 0.114255 | 0.133836 | 0.114255 | 0.142146 |
| Small | Edema | 0.936012 | 0.084979 | 0.277814 | 0.059054 | 0.059054 | 0.065222 | 0.059941 |
| Small | Pleural Effusion | 0.941576 | 0.091696 | 0.292794 | 0.025008 | 0.039402 | 0.047664 | 0.040601 |

## 해석

Small의 macro AUROC·확률 지표는 Tiny보다 좋았지만 모든 질환에서 개선된 것은 아닙니다. Consolidation은 Small의 AUROC가 높으면서 Brier·NLL·ECE가 나빠졌습니다. Cardiomegaly는 두 모델 모두 큰 calibration 오차를 보여 개선 대상으로 볼 수 있습니다.

Tiny의 reliability diagram에서 일부 질환은 실제 양성 빈도보다 낮은 위험을 예측했습니다. 전체 모델을 일괄적으로 과신한다고 결론 내릴 수 없으며 질환·확률 구간별 방향을 봐야 합니다.

## 한계

- 202장에 대한 point estimate이며 신뢰구간·여러 seed 검증 없음.
- 공개 모델의 학습 차이를 통제하지 않았으므로 모델 크기가 calibration 변화의 원인이라는 주장은 불가.
- TS 전 측정이며 개선 방법의 효과를 검증한 실험은 아님.
- 서버 출력 로그를 기록했으며 원시 파일의 독립 재계산은 하지 않음.

## 산출물

서버 평가 폴더 `eval_tiny_1023_0811`, `eval_small_1023_0850`의 `calibration/`에 지표 JSON·CSV와 reliability diagram이 있습니다. 원시 데이터·예측·가중치는 Git에 포함하지 않습니다.

출처: [공식 EVA-X 분류 문서](https://github.com/hustvl/EVA-X/tree/main/classification), [공개 가중치](https://huggingface.co/MapleF/eva_x/tree/main).
