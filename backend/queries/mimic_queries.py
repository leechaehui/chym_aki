"""MIMIC-IV 파생 임상 테이블 SQL 쿼리.

원본: repositories/mimic_repository.py 에 인라인으로 있던 SQL 을 추출.
모든 쿼리는 (schema: str) → sqlalchemy.text 형태.
─────────────────────────────────────────────────────────────
연관 백엔드     : repositories/mimic_repository.py
호출 서비스     : real_patient_service, icu_patient_detail_service, icu_monitor_service
대상 테이블     : chym.cohort, chym.cr_timeseries, chym.urine_rate,
                  chym.baseline_creatinine, chym.final_features_48h
─────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from sqlalchemy import text


# ── 테이블 존재 확인 ──────────────────────────────────────────────
def sql_table_exists() -> text:
    """앱 스키마에 MIMIC 파생 테이블 존재 여부 확인.

    호출처: mimic_repository.mimic_table_exists()
    """
    return text(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = :sch AND table_name = :n"
    )


# ── 코호트 피처 조회 (final_features_48h ⨝ cohort) ─────────────
def sql_cohort_features(schema: str, feature_select: str, limit: int | None = None) -> text:
    """ICU 코호트 raw 피처 + 인구학 + 정답 라벨.

    호출처: mimic_repository.icu_cohort_features()
    연관  : icu_monitor_service (모델 입력용)
    """
    sql = (
        f"SELECT f.stay_id, f.subject_id, f.age, f.gender, "
        f"c.first_careunit, c.icu_los_hours, f.aki_label, f.aki_stage, "
        f"{feature_select} "
        f"FROM {schema}.final_features_48h f "
        f"LEFT JOIN {schema}.cohort c ON c.stay_id = f.stay_id"
    )
    if limit:
        sql += f" LIMIT {int(limit)}"
    return text(sql)


# ── Cr 시계열 (시간순, 값만) ─────────────────────────────────────
def sql_creatinine_series(schema: str) -> text:
    """해당 stay 의 Cr 시계열(시간순) — baseline/추세 추정용.

    호출처: mimic_repository.stay_creatinine_series()
    """
    return text(
        f"SELECT creatinine FROM {schema}.cr_timeseries "
        "WHERE stay_id = :s ORDER BY charttime"
    )


# ── Cr 추세 (hours_from_admit, creatinine) ──────────────────────
def sql_creatinine_trend(schema: str) -> text:
    """해당 stay 의 크레아티닌 시계열 — [{hours_from_admit, creatinine}].

    호출처: mimic_repository.creatinine_trend()
    연관  : real_patient_service._build_one(), icu_patient_detail_service.trends()
    """
    return text(
        f"SELECT hours_from_icu_admit, creatinine FROM {schema}.cr_timeseries "
        "WHERE stay_id = :s AND creatinine IS NOT NULL ORDER BY charttime"
    )


# ── Cr 추세 배치 (stay_id IN (:ids)) ────────────────────────────
def sql_creatinine_trends_batch(schema: str) -> text:
    """여러 stay 의 Cr 추세를 한 번에 조회 (N+1 → 배치).

    호출처: mimic_repository.creatinine_trends_batch()
    연관  : real_patient_service.list_real_patients() 성능 튜닝
    """
    return text(
        f"SELECT stay_id, hours_from_icu_admit, creatinine FROM {schema}.cr_timeseries "
        "WHERE stay_id = ANY(:ids) AND creatinine IS NOT NULL ORDER BY stay_id, charttime"
    )


# ── 소변량 추세 ─────────────────────────────────────────────────
def sql_urine_rate_trend(schema: str) -> text:
    """해당 stay 의 시간당 소변량 시계열.

    호출처: mimic_repository.urine_rate_trend()
    연관  : real_patient_service._build_one(), icu_patient_detail_service.trends()
    """
    return text(
        "SELECT EXTRACT(EPOCH FROM (u.hour_bucket - c.icu_intime)) / 3600.0 AS hours, "
        "       u.urine_rate_ml_kg_h "
        f"FROM {schema}.urine_rate u "
        f"JOIN {schema}.cohort c ON c.stay_id = u.stay_id "
        "WHERE u.stay_id = :s AND u.urine_rate_ml_kg_h IS NOT NULL "
        "ORDER BY u.hour_bucket"
    )


# ── 소변량 추세 배치 ────────────────────────────────────────────
def sql_urine_rate_trends_batch(schema: str) -> text:
    """여러 stay 의 소변량 추세를 한 번에 조회.

    호출처: mimic_repository.urine_rate_trends_batch()
    연관  : real_patient_service.list_real_patients() 성능 튜닝
    """
    return text(
        "SELECT u.stay_id, "
        "       EXTRACT(EPOCH FROM (u.hour_bucket - c.icu_intime)) / 3600.0 AS hours, "
        "       u.urine_rate_ml_kg_h "
        f"FROM {schema}.urine_rate u "
        f"JOIN {schema}.cohort c ON c.stay_id = u.stay_id "
        "WHERE u.stay_id = ANY(:ids) AND u.urine_rate_ml_kg_h IS NOT NULL "
        "ORDER BY u.stay_id, u.hour_bucket"
    )


# ── 코호트 단일 행 ──────────────────────────────────────────────
def sql_cohort_record(schema: str) -> text:
    """해당 stay 의 코호트 원본 행(인구학·입원·재원시간).

    호출처: mimic_repository.cohort_record()
    연관  : real_patient_service, icu_patient_detail_service
    """
    return text(
        "SELECT subject_id, hadm_id, age, gender, first_careunit, "
        "       admittime, icu_intime, icu_outtime, icu_los_hours, admission_type "
        f"FROM {schema}.cohort WHERE stay_id = :s LIMIT 1"
    )


# ── 코호트 배치 조회 ────────────────────────────────────────────
def sql_cohort_records_batch(schema: str) -> text:
    """여러 stay 의 코호트 행을 한 번에 조회 (N+1 → 배치).

    호출처: mimic_repository.cohort_records_batch()
    연관  : real_patient_service.list_real_patients() 성능 튜닝
    """
    return text(
        "SELECT stay_id, subject_id, hadm_id, age, gender, first_careunit, "
        "       admittime, icu_intime, icu_outtime, icu_los_hours, admission_type "
        f"FROM {schema}.cohort WHERE stay_id = ANY(:ids)"
    )


# ── 최신 소변량 ─────────────────────────────────────────────────
def sql_urine_latest(schema: str) -> text:
    """가장 최근 시간당 소변량(mL/kg/h).

    호출처: mimic_repository.stay_latest_urine_rate()
    연관  : icu_patient_detail_service.summary()
    """
    return text(
        f"SELECT urine_rate_ml_kg_h FROM {schema}.urine_rate "
        "WHERE stay_id = :s AND urine_rate_ml_kg_h IS NOT NULL "
        "ORDER BY hour_bucket DESC LIMIT 1"
    )


# ── baseline Cr ──────────────────────────────────────────────────
def sql_baseline(schema: str) -> text:
    """precomputed baseline creatinine.

    호출처: mimic_repository.stay_baseline()
    연관  : icu_patient_detail_service.summary(), risk_history()
    """
    return text(
        f"SELECT baseline_cr, baseline_source FROM {schema}.baseline_creatinine "
        "WHERE stay_id = :s LIMIT 1"
    )


# ── 인구학 ───────────────────────────────────────────────────────
def sql_demographics(schema: str) -> text:
    """stay 의 나이/성별/careunit.

    호출처: mimic_repository.stay_demographics()
    """
    return text(
        f"SELECT age, gender, first_careunit FROM {schema}.cohort "
        "WHERE stay_id = :s LIMIT 1"
    )


# ── stay 목록 ────────────────────────────────────────────────────
def sql_list_stays(schema: str, careunit: str | None = None) -> str:
    """인제스트 대상 stay 목록(코호트).

    호출처: mimic_repository.list_stays()
    반환  : SQL 문자열 (동적 WHERE 때문에 text() 는 호출처에서 래핑)
    """
    sql = (
        "SELECT stay_id, subject_id, age, gender, first_careunit, icu_los_hours "
        f"FROM {schema}.cohort"
    )
    if careunit:
        sql += " WHERE first_careunit ILIKE :cu"
    sql += " ORDER BY stay_id LIMIT :lim"
    return sql
