"""Baseline Creatinine 추정 (지시서 §3, 3-layer + confidence).

baseline Cr 을 신뢰도와 함께 추정한다. EMR 없는 환경을 위한 3계층 fallback:
  Layer 1 (ideal)        : EMR 이력 존재 → median(과거 90일 Cr)          confidence 0.9
  Layer 2 (population)   : f(age, sex, CKD risk) 인구집단 사전값          confidence ~0.5
  Layer 3 (admission)    : min(입실 24–48h Cr) — 입실 대용값             confidence ~0.55

baseline confidence 는 guardrail(불확실성 패널티)과 KDIGO escalation 게이트(§5.4)에 쓰인다.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class Baseline:
    cr: float
    confidence: float
    source: str  # "emr_median" | "population_prior" | "admission_surrogate"
    reason: str


def population_prior(age: int | None, sex: str | None, ckd_risk: bool | None) -> float:
    """연령/성별/CKD 위험 기반 인구집단 baseline Cr(mg/dL) — 지시서 §3.1 Layer 2.

    20대 0.7–1.0 / 60대 0.8–1.2, 남성 +0.1, CKD 위험 ↑ → baseline ↑.
    """
    if age is None:
        base = 0.9
    elif age < 30:
        base = 0.85   # 20대 중앙 (0.7–1.0)
    elif age < 45:
        base = 0.9
    elif age < 60:
        base = 0.95
    else:
        base = 1.0    # 60대+ 중앙 (0.8–1.2)

    if (sex or "").upper().startswith("M"):
        base += 0.1   # 남성 근육량 보정
    if ckd_risk:
        base += 0.3   # CKD 위험군 상향
    return round(base, 2)


def estimate_baseline(
    *,
    prior_creatinines: list[float] | None,
    current_cr: float | None,
    age: int | None,
    sex: str | None = None,
    ckd_risk: bool | None = None,
) -> Baseline:
    """3-layer baseline 추정. 가용한 가장 신뢰도 높은 출처를 택한다."""
    priors = [c for c in (prior_creatinines or []) if c and c > 0]

    # Layer 1 — EMR 이력(과거 90일) median: 2점 이상이면 가장 신뢰.
    if len(priors) >= 2:
        cr = round(statistics.median(priors), 2)
        return Baseline(cr=cr, confidence=0.9, source="emr_median",
                        reason=f"과거 {len(priors)}개 Cr median {cr} (EMR 이력)")

    # Layer 3 — admission surrogate: 입실 초기 1점만 있으면 그 값(=min 대용).
    if len(priors) == 1:
        cr = round(min(priors), 2)
        return Baseline(cr=cr, confidence=0.55, source="admission_surrogate",
                        reason=f"입실 초기 Cr {cr} (24–48h 최저값 대용)")

    # Layer 2 — population prior: 과거값 전무.
    prior = population_prior(age, sex, ckd_risk)
    # 현재 Cr 이 인구사전값보다 낮으면(이미 낮은 baseline) 현재값을 보수적으로 채택.
    if current_cr and 0 < current_cr < prior:
        return Baseline(cr=round(current_cr, 2), confidence=0.5, source="admission_surrogate",
                        reason=f"현재 Cr {current_cr} 가 인구사전값보다 낮음 → 보수적 채택")
    return Baseline(cr=prior, confidence=0.5, source="population_prior",
                    reason=f"인구집단 사전값(age={age},sex={sex},ckd={bool(ckd_risk)})={prior}")
