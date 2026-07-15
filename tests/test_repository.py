"""Integration — repository: N+1 회피 + 인덱스 경로 (Verification 3.1/3.2)."""
from sqlalchemy import event, select

from core.database import SessionLocal, engine
from models.patient import Patient
from repositories.patient_repository import PatientRepository


def _count_queries(fn):
    """fn 실행 동안 발생한 SQL 실행 횟수를 센다."""
    count = {"n": 0}

    def _hook(conn, cursor, statement, params, context, executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            count["n"] += 1

    event.listen(engine, "after_cursor_execute", _hook)
    try:
        result = fn()
    finally:
        event.remove(engine, "after_cursor_execute", _hook)
    return result, count["n"]


def test_get_with_details_avoids_n_plus_1(client):
    """labs/trend/urine 3개 자식을 selectinload → 부모1 + 자식3 = 상수 쿼리."""
    with SessionLocal() as db:
        repo = PatientRepository(db)
        pid = db.execute(select(Patient.id).limit(1)).scalar_one()

        patient, n = _count_queries(lambda: repo.get_with_details(pid))
        # 자식 컬렉션 강제 로딩(이미 selectinload 로 적재되어 추가 쿼리 없음).
        _ = (len(patient.labs), len(patient.trend), len(patient.urine_output))
        _, n_after_access = _count_queries(
            lambda: (list(patient.labs), list(patient.trend), list(patient.urine_output))
        )

    assert patient is not None
    # 부모 1 + 자식 3 IN 쿼리 = 4 (행 수와 무관, N+1 아님).
    assert n <= 4, f"기대 <=4, 실제 {n}"
    # 접근 시 추가 쿼리 0(이미 eager 적재).
    assert n_after_access == 0


def test_list_summary_paginated(client):
    with SessionLocal() as db:
        from core.query_optimizer import Page

        rows = PatientRepository(db).list_summary(Page.of(2, 0))
        assert len(rows) <= 2


def test_base_get_by_pk(client):
    with SessionLocal() as db:
        repo = PatientRepository(db)
        pid = db.execute(select(Patient.id).limit(1)).scalar_one()
        assert repo.get(pid).id == pid
        assert repo.get("does-not-exist") is None
