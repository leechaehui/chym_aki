"""MIMIC 자동 Concept 추출 — stay_id → ConceptVector.

기존 ICU 서비스(검증된 임상 추출)를 재사용:
  - IcuPatientDetailService.summary : age·gender·egfr·stage_num·urine_rate
  - mimic_repository.creatinine_trend: Cr 시계열 → trend slope(ΔCr/day)
MIMIC ICU 에 없는 concept(proteinuria·a1c·DM/HTN·etiology)은 None/unknown(보수적, 저가중).
→ 사용자가 ICU 환자를 고르면 자동으로 Clinical Concept 가 채워진다.
"""
from __future__ import annotations

import numpy as np

from repositories import mimic_repository as mimic
from services.icu_patient_detail_service import IcuPatientDetailService

from .builder import build_mimic_concept
from .schema import ConceptVector


def _trend_slope(series: list[dict]) -> float | None:
    """Cr 시계열 → 하루당 변화율(ΔCr/day). 2점 미만이면 None."""
    pts = [(p.get("hours_from_admit"), p.get("creatinine")) for p in series
           if p.get("hours_from_admit") is not None and p.get("creatinine") is not None]
    if len(pts) < 2:
        return None
    h = np.array([p[0] for p in pts], float)
    c = np.array([p[1] for p in pts], float)
    if h.max() - h.min() < 1e-6:
        return None
    slope_per_hour = float(np.polyfit(h, c, 1)[0])
    return round(slope_per_hour * 24.0, 4)        # ΔCr/day


def _oliguria(urine_rate_ml_kg_h: float | None) -> int | None:
    if urine_rate_ml_kg_h is None:
        return None
    if urine_rate_ml_kg_h < 0.1:
        return 2                                  # anuria
    if urine_rate_ml_kg_h < 0.5:
        return 1                                  # oliguria
    return 0


def build_concept_from_stay(stay_id: int) -> ConceptVector | None:
    """MIMIC ICU stay → ConceptVector. 코호트 미존재 시 None."""
    svc = IcuPatientDetailService()
    s = svc.summary(stay_id)
    if s is None:
        return None
    slope = _trend_slope(mimic.creatinine_trend(stay_id))
    return build_mimic_concept(
        kdigo_stage=s.get("stage_num"),
        cr_trend_slope=slope,
        oliguria=_oliguria(s.get("urine_rate_ml_kg_h")),
        egfr=s.get("egfr"),
        age=s.get("age"),
        sex=s.get("gender"),
        proteinuria_mg_g=None,     # MIMIC ICU 미보유(저가중 None)
        a1c_pct=None,
        diabetes=None,
        hypertension=None,
        etiology_hint="unknown",   # MIMIC etiology 불명확 → hint 미사용
    )
