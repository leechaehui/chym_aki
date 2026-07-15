"""신장내과 대시보드용 **실제 환자 어댑터** — 실 ICU 코호트를 프론트 Patient(PatientOut) 형태로 변환.

데모(mock) 환자 대신 실데이터를 흘려보내되 **프론트 구조는 그대로 둔다**(GET /patients 만 교체).
- 검사값(labs) = final_features_48h(raw 48h 집계: bun/na/k/hco3) + cr_timeseries(최신 Cr) + computed eGFR
- 추이(trend) = cr_timeseries(일자별 Cr·eGFR)
- 소변량(urine_output) = urine_rate(다운샘플)
- 위험점수(ai_risk_score) = 학습 모델 위험점수(0–100)
- 이름 = 합성 표시명(fake_korean_name), 진단 = 모델 예측 단계(실 진단명 없음)

모든 값은 결정론적 조회/계산이며 생성형 추론이 아니다.

SQL 쿼리: queries/patient_queries.py
공통 상수: queries/constants.py
"""
from __future__ import annotations

from datetime import timedelta

import pandas as pd

from core.config import settings
from core.database import engine
from core.fake_name import fake_korean_name
from queries.constants import LAB_REFERENCES, STAGE_LABELS, is_female
from queries.patient_queries import sql_raw_labs
from repositories import mimic_repository as mimic
from services import clinical_calculations as clinical
from services.icu_monitor_service import _predictions, prediction_for_stay

_TREND_POINTS = 12  # 추이/소변량 미니차트용 최대 점 수(다운샘플).


def _flag(value: float | None, lo: float | None, hi: float | None) -> str:
    if value is None:
        return "normal"
    if hi is not None and value > hi:
        return "high"
    if lo is not None and value < lo:
        return "low"
    return "normal"


def _lab(key: str, value: float) -> dict:
    """Lab 참조범위를 constants.LAB_REFERENCES 에서 자동 조회하여 dict 생성."""
    ref = LAB_REFERENCES.get(key, {})
    label = ref.get("label", key.upper())
    unit = ref.get("unit", "")
    lo = ref.get("ref_low")
    hi = ref.get("ref_high")
    return {
        "key": key, "label": label, "value": round(float(value), 2), "unit": unit,
        "ref_low": lo, "ref_high": hi, "flag": _flag(value, lo, hi),
    }


def _raw_labs(stay_ids: list[int]) -> dict[int, dict]:
    """final_features_48h 의 raw 집계(bun/na/k/hco3)를 stay 별로 일괄 조회."""
    if not stay_ids:
        return {}
    schema = settings.app_schema
    with engine.connect() as c:
        rows = c.execute(sql_raw_labs(schema), {"ids": stay_ids}).mappings().all()
    return {int(r["stay_id"]): dict(r) for r in rows}


def _downsample(points: list, n: int = _TREND_POINTS) -> list:
    """긴 시계열을 균등 간격으로 n 점까지 줄인다(미니차트 가독성)."""
    if len(points) <= n:
        return points
    step = len(points) / n
    return [points[int(i * step)] for i in range(n)]


def _build_one(row, cohort: dict, raw: dict, cr_series: list[dict], uo_series: list[dict]) -> dict:
    stay_id = int(row["stay_id"])
    subject_id = int(row["subject_id"])
    age = cohort.get("age")
    gender = cohort.get("gender")
    female = is_female(gender)

    cr_now = cr_series[-1]["creatinine"] if cr_series else None
    egfr_now = clinical.estimate_egfr_ckd_epi_2021(cr_now, age, female) if cr_now is not None else None
    bun, na, k, hco3 = raw.get("bun_max"), raw.get("sodium_min"), raw.get("potassium_max"), raw.get("bicarbonate_min")

    labs: list[dict] = []
    if cr_now is not None:
        labs.append(_lab("cr", cr_now))
    if egfr_now is not None:
        labs.append(_lab("egfr", round(egfr_now, 1)))
    if bun is not None:
        labs.append(_lab("bun", bun))
    if na is not None:
        labs.append(_lab("na", na))
    if k is not None:
        labs.append(_lab("k", k))
    if hco3 is not None:
        labs.append(_lab("hco3", hco3))

    intime = pd.to_datetime(cohort.get("icu_intime"), errors="coerce")

    def _date(hours) -> str:
        if pd.isna(intime):
            return "1970-01-01"
        return (intime + timedelta(hours=float(hours))).strftime("%Y-%m-%d")

    trend: list[dict] = []
    for p in _downsample(cr_series):
        cr = float(p["creatinine"])
        eg = clinical.estimate_egfr_ckd_epi_2021(cr, age, female) or 0.0
        trend.append({
            "date": _date(p["hours_from_admit"]),
            "creatinine": round(cr, 2),
            "egfr": round(float(eg), 1),
            "bun": float(bun) if bun is not None else 0.0,
        })

    urine = [
        {"date": _date(p["hours_from_admit"]), "value": round(float(p["urine_rate_ml_kg_h"]), 2)}
        for p in _downsample(uo_series)
    ]

    pnum = int(row["pred"])
    return {
        "id": str(stay_id),
        "mrn": f"ICU-{stay_id}",
        "name": fake_korean_name(subject_id),
        "sex": "M" if (gender or "").upper().startswith("M") else "F",
        "age": int(age) if age is not None else 0,
        "diagnosis": f"{STAGE_LABELS[pnum]} (모델 예측)",
        "admitted_at": intime.isoformat() if not pd.isna(intime) else "",
        "attending": "ICU 담당",
        "room": cohort.get("first_careunit") or "—",
        "ai_risk_score": int(row["risk_score"]),
        "predicted_stage": STAGE_LABELS[pnum],
        "labs": labs,
        "trend": trend,
        "urine_output": urine,
    }


def list_real_patients(limit: int = 20, offset: int = 0) -> list[dict]:
    """실 ICU 코호트를 IcuMonitorService.list_patients()와 동일 기준(등급→점수)으로 정렬해 Patient 형태로 반환.

    과거엔 위험밴드별로 골고루 섞어 뽑는 별도 샘플링(_stratified_sample)을 썼는데, 그러면 같은 코호트를
    보는 "ICU AKI 모니터링" 화면(IcuMonitorService.list_patients, [pred, risk_score] 정렬)과
    이 화면(신장내과 대시보드)이 서로 다른 환자/다른 순서를 보여줘서 두 화면 숫자가 어긋났다.
    반드시 동일 정렬을 써서 두 화면이 같은 환자를 같은 순서·같은 점수로 보여주게 한다.

    [성능 튜닝] 환자별 개별 쿼리(N+1) → 배치 쿼리로 전환.
    Before: cohort_record() × 20 + creatinine_trend() × 20 + urine_rate_trend() × 20 = 60 쿼리
    After:  cohort_records_batch() + creatinine_trends_batch() + urine_rate_trends_batch() = 3 쿼리
    """
    df = _predictions().sort_values(["pred", "risk_score"], ascending=[False, False]).iloc[offset: offset + limit]
    stay_ids = [int(s) for s in df["stay_id"].tolist()]

    # 배치 조회 (60 → 3+1 쿼리)
    raws = _raw_labs(stay_ids)
    cohorts = mimic.cohort_records_batch(stay_ids)
    cr_trends = mimic.creatinine_trends_batch(stay_ids)
    uo_trends = mimic.urine_rate_trends_batch(stay_ids)

    out: list[dict] = []
    for _, row in df.iterrows():
        sid = int(row["stay_id"])
        cohort = cohorts.get(sid)
        if cohort is None:
            continue
        out.append(_build_one(
            row, cohort,
            raws.get(sid, {}),
            cr_trends.get(sid, []),
            uo_trends.get(sid, []),
        ))
    return out


def get_real_patient(stay_id: int) -> dict | None:
    """단일 실 환자(Patient 형태). 없으면 None."""
    row = prediction_for_stay(stay_id)
    cohort = mimic.cohort_record(stay_id)
    if row is None or cohort is None:
        return None
    cr_series = mimic.creatinine_trend(stay_id)
    uo_series = mimic.urine_rate_trend(stay_id)
    return _build_one(row, cohort, _raw_labs([stay_id]).get(stay_id, {}), cr_series, uo_series)
