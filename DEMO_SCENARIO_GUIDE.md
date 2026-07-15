# AKI 대시보드 데모 시뮬레이터 — 시나리오 & 체크 가이드

`2026-07-03` 작업 정리. 데모 시뮬레이터(`backend/api/demo.py`)로 신장내과 대시보드 / ICU AKI 모니터링 화면을 실제 데이터 흐름 그대로 시연하는 방법과, 각 단계에서 확인해야 할 것들을 정리한다.

---

## 1. 시나리오 순서

프론트 우하단 **"데모 시뮬레이션"** 플로팅 버튼 → 패널에서 순서대로 클릭 (병리과 화면에서는 버튼 자체가 안 보임).

| 순서 | 버튼 | 하는 일 |
|---|---|---|
| ① | **시나리오 초기화 (Setup)** | 기존 데이터 전체 TRUNCATE 후 300명 신규 적재(정상 220 / Stage1 60 / Stage2-3 20). 0h 시점 수치로 세팅, 위험군은 최종 목표치의 15%만 미리 반영(초기 스프레드용). 모델 재추론 후 실제 위험도/등급 반영. `1시간 진행` 진행 위치·알람 dedup 캐시도 초기화됨. |
| ② | **1시간 진행 (Advance)** | 위험군(80명) 중 **10명씩** 롤링으로 악화 반영(누를 때마다 다음 배치로 이동, 총 8번 눌러야 위험군 전체 소진). 랩 수치·추이 포인트·KDIGO 알람(`_publish_lab`)까지 실제 파이프라인으로 갱신. |
| ③ | **오준현 환자 악화 트리거** | 시나리오 주인공(subject_id `10218191`)을 Cr 3.6 / 고칼륨혈증 / 무뇨 직전으로 강제 악화. 알림 배너(`chym.notifications`) + 알람 센터(`chym.alerts`) 둘 다 생성. 재트리거해도 중복 에러 없음(매번 새 알림 id 발급). |

순서는 **①→②(1~8회)→③** 이 기본 흐름. ②를 반복할수록 대시보드의 "주의 이상" 인원이 점점 늘어난다.

---

## 2. 데이터 아키텍처 요약

- **모델 실제 입력**(`chym.icu_features_scaled`): CSV(`test_final.csv`)의 스케일된 피처 그대로 사용 — 위험도 예측(`risk_score`/`pred`)은 이 테이블 기준.
- **표시용 임상 수치**(랩/추이그래프/알람엔진 입력): `test_final.csv`의 `creatinine_max`/`bun_max` 등은 **모델 학습용 RobustScaler 표준화 값**(음수 가능, stage별 차이 없음)이라 그대로 쓰면 안 됨 → `aki_stage`(정답 등급)에서 `_clinical_values_for_stage()`로 임상적으로 도출.
- **위험 등급(고위험/주의/안정) 판정**: `predictedStage`(모델의 실제 pred) 기준으로 전 화면 통일. `chym.patients.diagnosis`에도 `"AI 예측: {stage} (위험도 {score}%)"` 형식으로 모델의 실제 예측을 반영(정답 stage 아님).
- **알림 2계통**: `chym.notifications`(상단 배너, `notificationService`) / `chym.alerts`(알람 센터, KDIGO 파이프라인 `LAB_EVENT→AKI_EVENT→ALERT_EVENT`) — 서로 다른 테이블이라 둘 다 채워지는지 따로 확인 필요.

---

## 3. 단계별 체크 테이블

| # | 확인 대상 | 방법 | 정상 기준 |
|---|---|---|---|
| 1 | `chym.cohort` | `SELECT COUNT(*) FROM chym.cohort;` | 300 |
| 2 | `chym.patients` | `SELECT mimic_subject_id, ai_risk_score, diagnosis FROM chym.patients WHERE mimic_subject_id=10218191;` | ai_risk_score>0, diagnosis에 실제 예측 stage 텍스트 |
| 3 | `chym.icu_features_scaled` | ②를 누른 뒤 해당 배치 10명만 값이 바뀌었는지 | 나머지 290명은 불변 |
| 4 | `chym.patient_labs` | `SELECT * FROM chym.patient_labs WHERE patient_id='p-10218191';` | Cr/BUN/K가 양수·임상적으로 합리적인 값(음수 금지) |
| 5 | `chym.patient_trend_points` | `SELECT date, creatinine, egfr, bun FROM chym.patient_trend_points WHERE patient_id='p-10218191' ORDER BY date;` | setup(1개)→advance(2개)→trigger(3개), Cr이 단계적으로 상승 |
| 6 | `chym.patient_urine_points` | `SELECT value FROM chym.patient_urine_points WHERE patient_id='p-10218191';` | stage 높을수록 값 낮음(핍뇨) |
| 7 | `chym.notifications` | `SELECT COUNT(*) FROM chym.notifications;` | 트리거 후 ≥1건 |
| 8 | `chym.alerts` | `SELECT COUNT(*) FROM chym.alerts;` | advance/trigger 후 ≥1건(KDIGO 엔진이 CONFIRMED/SUSPECTED/PRE_AKI 판정한 만큼) |
| 9 | 화면 일치성 | 신장내과 대시보드 vs ICU AKI 모니터링 | 같은 환자 = 같은 위험점수·같은 등급·같은 정렬 순서 |
| 10 | 알람 센터 UI | `/nephrology/icu-aki` 우측 알람 센터 패널 | 위 8번과 건수 일치 |

---

## 4. 오늘 발견·수정한 주요 버그

- **분류 등급 역전**: `risk_score`(가중합)와 `pred`(등급 분류)가 다른 공식이라 "Stage1인데 고위험" 같은 모순 발생 → 정렬·뱃지를 `pred` 기준으로 통일(`icu_monitor_service.list_patients`, `riskLabel.ts`).
- **`getRiskLabel` 안정 등급 누락**: "고위험 아니면 전부 주의"로만 분기돼 있어 Non-AKI도 안정이 아니라 주의로 표시됨 → 3단계(고위험/주의/안정) 명시 분기로 수정. `AiRiskGauge.tsx`에 동일 버그 중복 존재 — 같이 수정.
- **대시보드 목록 truncation**: `chym.patients` 조회가 `admitted_at DESC` 정렬 + `limit=20`이라 고위험 환자가 페이지 밖으로 잘려나감 → `ai_risk_score DESC` 정렬로 변경.
- **알람 파이프라인 미연결**: `advance-hour`/`trigger-event`가 `chym.notifications`만 채우고 실제 KDIGO 알람 파이프라인(`chym.alerts`)은 전혀 안 태우고 있었음 → `_publish_lab()` 연결.
- **스케일된 CSV 값을 실제 임상 수치처럼 사용**: `creatinine_max` 등이 RobustScaler로 표준화된 값(음수 가능)인데 mg/dL인 것처럼 그대로 랩/추이/알람엔진에 주입 → `aki_stage` 기반 `_clinical_values_for_stage()`로 대체.
- **신장내과 대시보드 ↔ ICU AKI 모니터링 데이터 소스 불일치**: `GET /api/patients`가 `chym.patients` 대신 실제 MIMIC 전체 코호트에서 별도 샘플링(`_stratified_sample`)하던 경로가 있었음 → `chym.patients` 우선, 정렬도 동일 기준으로 통일.
- **이름 충돌**: 합성 이름 풀이 8,000가지뿐이라 300명 규모에서 충돌 빈발 → 48,000가지로 확장.
