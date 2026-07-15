"""MIMIC-IV 파생 임상 테이블 읽기 전용 접근.

cohort/cr_timeseries/urine_rate/baseline_creatinine 등 AKI 파이프라인 산출 테이블은
앱 전용 스키마(settings.app_schema = chym)에 적재돼 있다(이전에는 public 사용 → 통합 이동).
앱 ORM 과 무관한 임상 데이터 조회 전용이며, 모두 읽기 전용 — 쓰기/DDL 없음.

SQL 쿼리는 queries/mimic_queries.py 에 모듈화되어 있다.
스키마명은 신뢰 가능한 설정값(settings.app_schema)이라 f-string 으로 안전하게 주입한다.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd
from sqlalchemy import text

from core.config import settings
from core.database import engine
from queries.constants import FEATURE_COLS
from queries import mimic_queries as Q

# MIMIC 파생 테이블이 적재된 앱 스키마(= chym). 더 이상 public 을 참조하지 않는다.
MIMIC_SCHEMA = settings.app_schema


def mimic_table_exists(name: str) -> bool:
    """앱 스키마에 MIMIC 파생 테이블 존재 여부(가용성 판정용)."""
    with engine.connect() as c:
        return bool(c.execute(
            Q.sql_table_exists(), {"sch": MIMIC_SCHEMA, "n": name}
        ).first())


def icu_cohort_features(limit: int | None = None) -> pd.DataFrame:
    """ICU 코호트 raw 피처 + 인구학 + 정답 라벨.

    final_features_48h ⨝ cohort(careunit/los).
    반환 컬럼: stay_id, subject_id, age, gender, first_careunit, icu_los_hours,
              aki_label, aki_stage + FEATURE_COLS(raw).
    """
    feat = ", ".join(f"f.{c}" for c in FEATURE_COLS)
    sql = Q.sql_cohort_features(MIMIC_SCHEMA, feat, limit)
    return pd.read_sql(sql, engine)


def stay_creatinine_series(stay_id: int) -> list[float]:
    """해당 stay 의 Cr 시계열(시간순) — baseline/추세 추정용."""
    with engine.connect() as c:
        rows = c.execute(Q.sql_creatinine_series(MIMIC_SCHEMA), {"s": stay_id}).all()
    return [float(r[0]) for r in rows if r[0] is not None]


def creatinine_trend(stay_id: int) -> list[dict]:
    """해당 stay 의 크레아티닌 시계열 — [{hours_from_admit, creatinine}] (시간순).

    ICU 입실 후 경과시간(hours_from_icu_admit)을 x축으로 추세 그래프에 쓴다.
    """
    with engine.connect() as c:
        rows = c.execute(Q.sql_creatinine_trend(MIMIC_SCHEMA), {"s": stay_id}).all()
    return [{"hours_from_admit": float(h), "creatinine": float(v)} for h, v in rows]


def creatinine_trends_batch(stay_ids: list[int]) -> dict[int, list[dict]]:
    """여러 stay 의 Cr 추세를 한 번에 조회 (N+1 → 배치, 성능 튜닝).

    반환: {stay_id: [{hours_from_admit, creatinine}, ...]}
    """
    if not stay_ids:
        return {}
    result: dict[int, list[dict]] = defaultdict(list)
    with engine.connect() as c:
        rows = c.execute(Q.sql_creatinine_trends_batch(MIMIC_SCHEMA), {"ids": stay_ids}).all()
    for sid, h, v in rows:
        result[int(sid)].append({"hours_from_admit": float(h), "creatinine": float(v)})
    return dict(result)


def urine_rate_trend(stay_id: int) -> list[dict]:
    """해당 stay 의 시간당 소변량 시계열 — [{hours_from_admit, urine_rate_ml_kg_h}].

    urine_rate 의 hour_bucket 과 cohort.icu_intime 차이로 입실 후 경과시간을 산출한다.
    """
    with engine.connect() as c:
        rows = c.execute(Q.sql_urine_rate_trend(MIMIC_SCHEMA), {"s": stay_id}).all()
    return [{"hours_from_admit": float(h), "urine_rate_ml_kg_h": float(v)} for h, v in rows]


def urine_rate_trends_batch(stay_ids: list[int]) -> dict[int, list[dict]]:
    """여러 stay 의 소변량 추세를 한 번에 조회 (N+1 → 배치, 성능 튜닝).

    반환: {stay_id: [{hours_from_admit, urine_rate_ml_kg_h}, ...]}
    """
    if not stay_ids:
        return {}
    result: dict[int, list[dict]] = defaultdict(list)
    with engine.connect() as c:
        rows = c.execute(Q.sql_urine_rate_trends_batch(MIMIC_SCHEMA), {"ids": stay_ids}).all()
    for sid, h, v in rows:
        result[int(sid)].append({"hours_from_admit": float(h), "urine_rate_ml_kg_h": float(v)})
    return dict(result)


def cohort_record(stay_id: int) -> dict | None:
    """해당 stay 의 코호트 원본 행(인구학·입원·재원시간). 없으면 None."""
    with engine.connect() as c:
        r = c.execute(Q.sql_cohort_record(MIMIC_SCHEMA), {"s": stay_id}).mappings().first()
    return dict(r) if r else None


def cohort_records_batch(stay_ids: list[int]) -> dict[int, dict]:
    """여러 stay 의 코호트 행을 한 번에 조회 (N+1 → 배치, 성능 튜닝).

    반환: {stay_id: {subject_id, age, gender, ...}}
    """
    if not stay_ids:
        return {}
    with engine.connect() as c:
        rows = c.execute(Q.sql_cohort_records_batch(MIMIC_SCHEMA), {"ids": stay_ids}).mappings().all()
    return {int(r["stay_id"]): dict(r) for r in rows}


def stay_latest_urine_rate(stay_id: int) -> float | None:
    """가장 최근 시간당 소변량(mL/kg/h)."""
    with engine.connect() as c:
        r = c.execute(Q.sql_urine_latest(MIMIC_SCHEMA), {"s": stay_id}).first()
    return float(r[0]) if r and r[0] is not None else None


def stay_baseline(stay_id: int) -> dict | None:
    """precomputed baseline creatinine(있으면)."""
    with engine.connect() as c:
        r = c.execute(Q.sql_baseline(MIMIC_SCHEMA), {"s": stay_id}).first()
    return {"baseline_cr": float(r[0]), "source": r[1]} if r else None


def stay_demographics(stay_id: int) -> dict | None:
    """stay 의 나이/성별/careunit(LAB_EVENT 구성용)."""
    with engine.connect() as c:
        r = c.execute(Q.sql_demographics(MIMIC_SCHEMA), {"s": stay_id}).first()
    return {"age": int(r[0]) if r[0] is not None else None,
            "gender": r[1], "careunit": r[2]} if r else None


def list_stays(limit: int = 30, careunit: str | None = None) -> list[dict]:
    """인제스트 대상 stay 목록(코호트)."""
    sql_str = Q.sql_list_stays(MIMIC_SCHEMA, careunit)
    params: dict = {}
    if careunit:
        params["cu"] = f"%{careunit}%"
    params["lim"] = int(limit)
    with engine.connect() as c:
        rows = c.execute(text(sql_str), params).mappings().all()
    return [dict(r) for r in rows]
