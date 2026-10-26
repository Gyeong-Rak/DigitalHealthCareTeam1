# 추가 조합: U-Ignore + LoRA + Mixup

원래 dev 자동 선택은 Full + BCE + U-Ignore로, 이미 실행한 `b_ignore`와 같은 설정이었습니다. 사용자는 중복 학습 대신 **U-Ignore + LoRA + Mixup**의 결합을 추가로 탐색하도록 요청했습니다. 이 문서는 실행 계획이며, 새 조합의 학습 결과는 아직 없습니다.

| 항목 | 설정 |
|---|---|
| 불확실 라벨 | U-Ignore: -1 원소의 학습 loss 제외 |
| 학습 범위 | LoRA rank 8, alpha 16 + classifier/pooling norm |
| 학습 방법 | Mixup alpha .2 + BCE, 유효 label mass 혼합 |
| 예산 | seed 0 하나, GPU당 batch 256, 최대 10 epoch·최소 5 이후 dev plateau 2회 조기 종료 |
| 평가 | 기존 split 그대로, 두 TS 프로토콜·질환별 지표·그림·환자 bootstrap |

기존 13개 조건의 완료 marker·예측·지표·학습 이력은 해시를 확인하고 새 출력 폴더로 복사합니다. 기존 checkpoint와 데이터 이미지는 복사하지 않습니다. **13개를 다시 학습하거나 추론하지 않고 새 조합만 한 번 학습**합니다. 원래 실행 폴더의 결과는 삭제하거나 고치지 않습니다. 완료한 원래 자동 조합이 있더라도 추가 실행에는 원래 13조건과 새 조합만 포함합니다.

새 출력 폴더 기본값:

```text
~/workspace/outputs/chexpert/tiny_study_lora_mixup_ignore/
```

`STATUS.md`와 `study_status.py --watch 10`으로 기존과 동일하게 확인합니다. MD는 `docs/experiments/generated/tiny_study_lora_mixup_ignore/`에 생성됩니다. 최종 비교에는 기존 baseline, LoRA, U-Ignore, Mixup 결과도 포함됩니다.

## 기존 실행 중단 후 전환

원래 driver가 실행 중인 checkout은 코드 hash로 고정되어 있습니다. **fetch → 새 종료 도구를 임시 파일로 꺼냄 → 원래 driver 종료·자식 정리 확인 → checkout 업데이트 → 새 실행** 순서를 지킵니다. 종료 도구는 source `.lock`과 `/proc`의 PID·스크립트·output 경로를 확인하며, 원래 조합/최종 요약 단계의 driver에만 SIGTERM을 보냅니다. 전체 Python 프로세스를 종료하는 `pkill`은 사용하지 않습니다.

```bash
conda activate evax
(
  set -e
  cd "$HOME/workspace/DigitalHealthCareTeam1"
  git fetch http://github.snuhub-cache-server/Gyeong-Rak/DigitalHealthCareTeam1.git main
  git show FETCH_HEAD:scripts/stop_calibration_study.py > /tmp/stop_dhc_study.py
  python /tmp/stop_dhc_study.py --output-dir "$HOME/workspace/outputs/chexpert/tiny_study_seed0"
  git merge --ff-only FETCH_HEAD
  nohup python scripts/run_calibration_followup.py \
    --source-dir "$HOME/workspace/outputs/chexpert/tiny_study_seed0" \
    --output-dir "$HOME/workspace/outputs/chexpert/tiny_study_lora_mixup_ignore" \
    > "$HOME/workspace/outputs/chexpert/tiny_study_lora_mixup_ignore.launch.log" 2>&1 < /dev/null &
)
```

중단 후 재개할 때는 **코드 업데이트 없이** 같은 follow-up 명령을 다시 실행합니다. 새 조합의 기존 checkpoint·RNG·optimizer 상태를 사용합니다. 원래 실행과 새 실행의 output 폴더를 같은 경로로 지정하지 않습니다. Follow-up은 원래 source lock도 유지하므로 원래 driver를 실수로 동시에 재개할 수 없습니다.

## 해석

이 조합은 test를 확인한 뒤 정한 **사후 탐색**입니다. 보고서에도 이를 명시하며, 원래 dev 자동 선택 기록은 `combination_selection.json`에 함께 보존합니다. 이번 결과가 좋아지더라도 가장 좋은 조합이라고 확정하거나 같은 test의 결과를 독립적인 최종 검증이라고 주장하지 않습니다. dev 모델 선택과 calibration/test 환자 분리는 그대로 유지합니다.
