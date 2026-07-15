# V&V 산출물 (Verification & Validation)

작업지시서 "9. Output 요구사항"의 필수 산출물 모음. 대부분 코드에서 자동 추출되어
재생성 가능하다(수기 문서 아님 → 코드와 항상 동기화).

## 산출물 인덱스

| 산출물 | 파일 | 생성 | 작업지시서 |
|---|---|---|---|
| module dependency graph | [module_dependency_graph.md](module_dependency_graph.md) | `tests/vv_runners/generate_vv_docs.py` | 9, 3.1 |
| database schema | [database_schema.md](database_schema.md) | `tests/vv_runners/generate_vv_docs.py` | 9 |
| API endpoint list | [api_endpoint_list.md](api_endpoint_list.md) | `tests/vv_runners/generate_vv_docs.py` | 9 |
| query execution plan summary | [query_execution_plan_summary.md](query_execution_plan_summary.md) | `tests/vv_runners/verify_queries.py` | 9, 3.2 |
| AI model validation report | [ai_model_validation_report.md](ai_model_validation_report.md) | `tests/vv_runners/run_aki_validation.py` | 9, 4 |
| AI Draft 파이프라인 실행 결과 | [ai_draft_pipeline_report.md](ai_draft_pipeline_report.md) | `tests/vv_runners/run_ai_draft_pipeline.py` | AI Draft 11 |
| known failure cases list | [known_failure_cases.md](known_failure_cases.md) | (수기 + 검증 근거) | 9, 10 |
| **종합 검증 보고서 (PDF)** | [VV_Backend_Test_Report.pdf](VV_Backend_Test_Report.pdf) | `tests/vv_runners/build_vv_report_pdf.py` | 종합 |

## 검증 인프라 (코드)

| 영역 | 위치 | 내용 |
|---|---|---|
| V&V 레이어 | `validator/` | metrics(AUROC/AUPRC/calibration/ECE), subgroup, clinical(단조성), timeseries, report, **soap_validator/ap_evidence_validator/risk_score_validator** |
| AI Draft 파이프라인 | `services/` | stt_service · soap_service(근거기반 SOAP) · problem_list_service · cdss_risk_service(rule+model hybrid) · ai_draft_service(오케스트레이터) |
| 테스트(pytest) | **리포 루트 `tests/test_*.py`** | unit(validator·model·SOAP·CDSS) + integration(API 엔드포인트·repository·AI Draft 파이프라인) + service(bed/bed_detail/patient/consult/admission/auth/notification/pathology/nephrology) + **RBAC 역할 매트릭스(관리자 슈퍼유저 포함)** + hallucination detection. **172 passed, 전체 95% 커버리지** |
| 검증 러너 | `tests/vv_runners/` | 산출물 생성 스크립트(generate_vv_docs·verify_queries·run_aki_validation·run_ai_draft_pipeline) — pytest 수집 대상 아님 |
| 트레이싱 | `core/logging.py` + `main.py` 미들웨어 | request_id 발급/전파, 지연시간 로깅, 표준 error_code |

> 참고: 테스트 폴더는 리포 루트(`C:\dev\chym_aki\tests`)에 있고, 앱 코드는 `backend/` 에 있다.
> `conftest.py` 가 `backend/` 를 sys.path 에 넣고 임시 SQLite 로 격리한다. `pytest.ini` 도 리포 루트.

## 전체 재생성 (1회) — 리포 루트에서 실행

```bash
cd C:\dev\chym_aki
set SEED_ON_STARTUP=false
backend\.venv\Scripts\python.exe tests\vv_runners\generate_vv_docs.py    # graph / schema / endpoints
set DATABASE_URL=postgresql+psycopg2://chym:chym@localhost:5432/chym_aki  # verify_queries 는 PG 전용
backend\.venv\Scripts\python.exe tests\vv_runners\verify_queries.py      # query plan (PostgreSQL EXPLAIN ANALYZE)
backend\.venv\Scripts\python.exe tests\vv_runners\run_aki_validation.py  # AI model validation
backend\.venv\Scripts\python.exe tests\vv_runners\run_ai_draft_pipeline.py # AI Draft 파이프라인 실행결과
backend\.venv\Scripts\python.exe -m pytest                               # V&V 테스트(러너 제외)
```

## 작업지시서 충족 현황

| 섹션 | 항목 | 상태 | 근거 |
|---|---|---|---|
| 2.1 | 레이어 분리(api/service/domain/repo/ai/validator/tests) | ✅ | `validator/`, `tests/` 신설(나머지 기존) |
| 3.1 | SRP / 레이어 경계 | ✅ | dependency graph 위반 0(예외 1 명시) |
| 3.2 | EXPLAIN / N+1 / 인덱스 | ✅ | query plan 전부 PASS, `ix_patients_admitted_at` 추가 |
| 3.3 | error code 표준화 / request_id 로깅 | ✅ | `core/logging.py`, `DomainError.error_code` |
| 4.1 | AUROC/AUPRC/calibration/subgroup/missing | ✅ | validation report (ICU subgroup은 데이터 한계 D1) |
| 4.2 | Cr/eGFR/소변량 반영, 약물 ablation | ✅ | clinical 단조성 PASS, feature ablation |
| 4.3 | time-series(sliding window/onset/누수) | ✅ | timeseries 검증 PASS(누수 0, 라벨 정합 100%) |
| 7 | unit/integration/clinical 테스트 | ✅ | 172 passed (전체 95%; 신장내과·병리과 API 100%) |
| 7(RBAC) | 권한별(role) 접근제어 테스트 | ✅ | `tests/test_rbac.py` 29건 — 역할 매트릭스 + **관리자 슈퍼유저 전수** + 인증 게이트(core/deps 97%) |
| AI Draft | STT→SOAP(근거)→Problem→CDSS→Timeline, hallucination 차단·검증 강제 | ✅ | [ai_draft_pipeline_report.md](ai_draft_pipeline_report.md), soap/ap/risk validator, 422 차단 |
| 9 | 필수 출력물 | ✅ | 본 디렉터리 |
| 10 | 금지사항(무검증 성능주장/raw SQL/timeline 없는 AKI 등) | ✅ | 전부 근거 수치 동반, raw SQL 0, timeline 기반 검증 |
