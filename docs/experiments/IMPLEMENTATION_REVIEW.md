# 자동 실험 코드 검토 기록 — 2026-10-25

## 변경 범위

기준 커밋 `ef80c5e`에서 작성한 코드의 로컬 검증 기록입니다. EVA-X 서브모듈과 데이터 파일을 변경하지 않았습니다. 병원 서버의 새 GPU 학습 결과는 아직 없습니다.

- PPT의 baseline·C/B/A 축 13조건과 dev로 선택하는 조합 1조건을 순차 실행.
- 사용자 요청대로 seed 0 하나. GPU당 batch 256, 최대 10 true epochs, 최소 5 이후 dev AUROC plateau 2회면 자동 종료.
- 모든 조건의 primary/secondary 분리된 TS 평가, 질환별·macro 지표, reliability diagram, 최종 환자 bootstrap.
- 조건별 학습·평가가 끝나면 `docs/experiments/generated/<실험>/`의 MD와 전체 진행표를 갱신.
- `STATUS.md` 한 파일에 핵심 진행·중간 지표·오류 요약·최근 주요 이벤트를 갱신. 원본 traceback은 별도 로그에 보관.
- 재시도, 독립 조건 계속 실행, 미완료 epoch 재개, 완료 산출물 hash 검증, 설정·코드 변경 감지, 실행 폴더 잠금.

## 검증 결과

| 검사 | 결과 |
|---|---|
| 일반 unittest | 35개 중 33개 통과, 선택적 분산 테스트 2개는 기본 실행에서 skip |
| 선택적 CPU 2-rank Gloo 테스트 | 위 2개를 별도 실행하여 모두 통과; checkpoint·rank RNG·조기 종료 broadcast 확인 |
| 실제 Tiny 구조의 scope/LoRA 검사 | 4종 모두 forward/backward, 초기 adapter 동등성, 학습 파라미터 gradient 확인 |
| 실제 MIM 가중치 + 실제 이미지 2장 CPU preflight | 13조건 전부 통과; optimizer step 없음 |
| Loss 수치 검사 | mask·focal gamma=0의 BCE 환원·LPR 식·SPA link 단조성/정상점·Mixup mask 확인 |
| TS·통계 검사 | 양의 T, fit NLL 비악화, 순위 보존, 환자 누출 거부, paired bootstrap 확인 |
| 전체 orchestration fixture | 검사→학습 모사→dev/calibration/test→실제 TS·지표→MD·bootstrap→조합→완료/재개 통과 |
| 장애 복구 fixture | 실제 실패 subprocess의 재시도·완료 skip·산출물 변조 감지·CLI 경로 재사용 확인 |
| 정적 검사 | Python compile, shell syntax, diff whitespace 검사 통과 |

## 해석 범위

CPU Gloo와 synthetic toy optimizer 테스트는 병원 GPU 4개/NCCL/AMP의 실제 학습 처리량·수치 안정성을 검증한 결과가 아닙니다. 실제 이미지를 이용한 로컬 검사는 optimizer 업데이트 없이 작은 forward/backward만 수행했습니다. 실제 서버에서는 각 조건 시작 전에 GPU preflight를 다시 실행합니다.

SPA+LPR·MDCA·soft U 조합은 다중 라벨에 맞춘 명시적 독립 구현입니다. 특히 SPA inverse link의 인쇄본 부호와 미분의 불일치, hyperparameter 및 soft-target 확장은 [가이드](../../STUDY_GUIDE.md)에 기록했습니다. 원 논문의 결과를 재현했다고 주장하지 않습니다.

과거 완료 실험 MD는 사용자 제공 서버 로그를 옮긴 기록입니다. 서버 원본 배열을 직접 읽어 재계산한 것과 구분합니다. 새 실험 결과와 섞지 않습니다.

원격 main을 fetch하여 확인했으며 이 검토 시점까지 동료의 새 docs 형식은 발견되지 않았습니다. 현재 공통 MD 템플릿을 분리해 두었으며, 팀 형식이 올라오면 해당 템플릿을 맞춘 후 배포할 수 있습니다. 자동 실행은 Git commit/push를 수행하지 않습니다.
