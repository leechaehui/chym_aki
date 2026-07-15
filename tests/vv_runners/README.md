# tests/vv_runners — V&V 산출물 러너

`pytest` 단위/통합 테스트(`tests/test_*.py`)와 달리, 여기 스크립트는 **실행해서
`docs/vv/` 산출물(섹션 9)을 생성**하는 러너다. 파일명이 `test_*.py` 가 아니므로
pytest 수집 대상이 아니며, 별도로 직접 실행한다.

| 스크립트 | 생성물 | 작업지시서 |
|---|---|---|
| `generate_vv_docs.py` | module_dependency_graph / database_schema / api_endpoint_list | 9, 3.1 |
| `verify_queries.py` | query_execution_plan_summary (EXPLAIN, 인덱스/N+1) | 9, 3.2 |
| `run_aki_validation.py` | ai_model_validation_report.{md,json} (AUROC/AUPRC/calibration/subgroup/…) | 9, 4 |

## 실행

```bash
cd backend
set PYTHONPATH=.
set SEED_ON_STARTUP=false
.venv\Scripts\python.exe tests\vv_runners\generate_vv_docs.py
.venv\Scripts\python.exe tests\vv_runners\verify_queries.py
.venv\Scripts\python.exe tests\vv_runners\run_aki_validation.py
```

회귀 테스트(임상 단조성·메트릭·API 계약)는 `pytest tests\` 로 실행한다.
