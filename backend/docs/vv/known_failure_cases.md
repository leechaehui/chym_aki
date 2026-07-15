# Known Failure Cases (작업지시서 9 / 10)

검증 과정에서 확인된 한계/실패 케이스. "validation 없이 성능 주장 금지" 원칙에 따라
근거 수치와 함께 기록한다. (모델 수치 출처: `ai_model_validation_report.md`, holdout `test_final.csv` N=6433)

## 1. AI 모델 (AKI 2-stage)

| # | 케이스 | 근거 | 영향/대응 |
|---|---|---|---|
| M1 | **Stage2(중증도 LGBM) 변별력 낮음** | test AUROC **0.582** (우연 0.5 근접), AUPRC 0.264 | Stage1/2+3 구분은 *보조 지표*. 중증도 단독 임상판단 금지. 고recall(0.81)/저precision(0.28) — 과다경보 경향. |
| M2 | **Stage1 보정 불량(over-confident)** | ECE **0.179**, 예측확률 0.7–0.9 구간이 실제 위험 과대평가 | 운영 시 확률 원값 대신 위험등급(threshold 0.7) 사용. Platt/Isotonic 재보정 권장. |
| M3 | **결측 많은 환자 성능 저하** | 결측 적음 AUROC 0.950 vs 많음 0.843 (Δ0.107) | 결측 비율 높은 입력은 신뢰도 하향 표기 필요. |
| M4 | **신장(Cr/BUN) 피처 ablation 영향 미미** | 제거 시 AUROC Δ+0.004 | 모델이 소변량(제거 시 Δ0.149)에 크게 의존, Cr 신호 활용 적음 — 임상 직관과 부분 괴리. 피처 재설계 후보. |

## 2. 데이터/검증 범위 한계

| # | 케이스 | 내용 |
|---|---|---|
| D1 | **ICU/ER/ward subgroup 분석 불가** | 데이터셋에 careunit 컬럼 없음 → 4.1 subgroup 요건 부분 미충족. 성별·연령분위·결측수준으로 대체 검증. |
| D2 | **age 표준화로 절대 연령대 불가** | `age` z-score 화 → 절대 구간 대신 상대 분위(tercile)로 subgroup 분리. |
| D3 | **외부 검증(external validation) 부재** | 동일 코호트 holdout 만 평가. 타 기관/시점 일반화는 미검증. |

## 3. 시스템/아키텍처

| # | 케이스 | 내용 |
|---|---|---|
| S1 | **희소 EMR 입력은 규칙 기반으로 폴백** | 앱 환자 분석은 피처 커버리지 <60% → 학습 모델 대신 KDIGO 규칙(`RuleBasedAkiPredictor`). 학습 모델 성능치는 *표준화 48h 전체벡터*에서만 유효. |
| S2 | **`SELECT ... FOR UPDATE` 는 PG 에서만 실제 행잠금** | SQLite 는 쓰기 직렬화로 근사. 동시 병상 배정 경쟁은 운영(PostgreSQL)에서 검증 필요. |
| S3 | **clinical 단조성은 규칙 예측기 대상** | 배포 경로(희소 EMR) 기준 검증. 학습 모델 단조성은 별도(트리 모델 특성상 국소 비단조 가능). |

## 재현 방법

```bash
cd backend && set PYTHONPATH=. && set SEED_ON_STARTUP=false
.venv\Scripts\python.exe tests\vv_runners\run_aki_validation.py   # 모델 검증 리포트 갱신
.venv\Scripts\python.exe -m pytest tests\                         # 회귀(임상 단조성 포함)
```
