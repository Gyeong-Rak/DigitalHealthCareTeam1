# 실험 기록

| 실험 | 상태 | 주요 결과 |
|---|---|---|
| [공개 Tiny·Small 가중치 평가](2026-10-23_public_baselines.md) | 완료 | mAUROC Tiny 0.895816 / Small 0.901189 |
| [Tiny 재학습: BCE vs label smoothing](2026-10-23_tiny_loss_screening.md) | 완료 | AUROC 거의 동일, smoothing의 Brier·NLL 개선, ECE는 bin에 따라 혼합 |
| PPT 전체 축 자동 실행 | 코드 준비·검증 단계 | 새 학습 결과는 아직 없음 |

완료 기록은 사용자가 제공한 서버 로그를 옮긴 것입니다. 이 작업 환경에서 서버의 원시 예측 파일을 직접 재검증한 결과는 아닙니다. 미제공 값은 추정해서 채우지 않습니다.

새 자동 실행은 `generated/<실험 폴더명>/`에 조건별 보고서와 전체 `README.md`를 생성합니다. 학습·dev 선택·test 평가·TS 계산까지 끝난 조건만 결과에 포함합니다. 실행 중에는 중간 보고서, 전체 종료 시에는 환자 bootstrap CI를 추가합니다. 생성 문서는 로컬 파일이며 Git commit/push는 자동 실행하지 않습니다.

문서의 기본 형식은 **실험 내용 → 결과 → 해석 → 한계 → 산출물**입니다. 팀 형식이 확정되면 [`_template.md`](_template.md)를 수정하고 실행 시 `--report-template docs/experiments/_template.md`로 지정할 수 있습니다. 실험 도중에는 사용한 템플릿을 바꾸지 않고 새 실행부터 적용합니다.

실행·상태 확인·재개 방법: [STUDY_GUIDE.md](../../STUDY_GUIDE.md).

구현 검증과 남은 검증 범위: [IMPLEMENTATION_REVIEW.md](IMPLEMENTATION_REVIEW.md).
