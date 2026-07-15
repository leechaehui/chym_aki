"""AKI Safety Engine — LAB_EVENT consumer (지시서 전체).

Recall-first 이중 탐지 파이프라인:
  1. Baseline Estimator (3-layer + confidence)         services.baseline_estimator
  2. Sensitive AKI Detector (high recall, weak-signal 합산 + trend + clinical override)
  3. Guardrail Filter (FP 제어: plausibility/temporal/contradiction/baseline penalty)
  4. ROC-calibrated Decision (확률→zone, Youden operating point)  services.roc_calibration
  5. KDIGO Staging Engine                               services.kdigo_staging
  6. Final Decision Engine → AKI_EVENT(stage) emit

설계 원칙: 놓치지 않는 것 최우선(FN↓). sensitive_positive 는 guardrail 로 깎여도 최소 PRE_AKI 유지.
KDIGO 확정은 baseline 신뢰도 게이트를 통과해야 한다(과진단 억제, §5.4).
"""
from __future__ import annotations

import re

from core.event_bus import AKI_EVENT, event_bus
from core.logging import get_logger
from ai_draft.aki_model import get_aki_predictor
from models.base import new_id
from repositories import mimic_repository
from services import kdigo_staging, roc_calibration
from services.baseline_estimator import estimate_baseline

log = get_logger("chym.aki_engine")

# clinical override 키워드(지시서 §4.5) — 등장 시 강제 위험 상승.
_OVERRIDE_KEYWORDS = ("소변량 감소", "핍뇨", "무뇨", "부종", "Cr 상승", "크레아티닌 상승", "투석 고려", "투석")
_OVERRIDE_FLOOR = 0.6  # 키워드 시 확률 하한(최소 pre-AKI 진입).


def _f(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# camelCase(프론트/by_alias dump) ↔ snake_case 키 정규화 — 엔진은 snake 로 통일해 읽는다.
_KEY_ALIASES = {
    "urineOutput": "urine_output",
    "contrastExposure": "contrast_exposure",
    "ckdRisk": "ckd_risk",
}


def _normalize(data: dict) -> dict:
    out = dict(data)
    for camel, snake in _KEY_ALIASES.items():
        if camel in out and snake not in out:
            out[snake] = out[camel]
    return out


def _extract_features(data: dict, baseline_cr: float) -> dict:
    cr = _f(data.get("creatinine"))
    egfr = _f(data.get("egfr"))
    uo = _f(data.get("urine_output"))
    delta = (cr - baseline_cr) if (cr is not None) else None
    return {
        "creatinine_max": cr, "creatinine_min": baseline_cr, "baseline_creatinine": baseline_cr,
        "creatinine_delta": delta, "egfr": egfr, "urine_ml_kg_hr": uo,
        "oliguria_flag": 1 if (uo is not None and uo < 0.5) else 0,
    }


def _detect_trend(series: list[float]) -> tuple[bool, float]:
    """Cr 시계열의 상승 추세 감지(단일 값 아닌 기울기, §4.3). (rising, slope) 반환."""
    pts = [x for x in series if x and x > 0]
    if len(pts) < 2:
        return False, 0.0
    slope = pts[-1] - pts[0]
    rising = pts[-1] > pts[0] and pts[-1] >= pts[-2]  # 마지막 구간도 비감소
    return (rising and slope > 0.05), round(slope, 3)


def _fill_subject_id(patient_id: str | None, subject_id: int | None) -> int | None:
    if subject_id is not None:
        return subject_id
    if not patient_id:
        return None
    match = re.match(r"^stay-(\d+)$", patient_id, re.IGNORECASE)
    if not match:
        return None
    stay_id = int(match.group(1))
    cohort = mimic_repository.cohort_record(stay_id)
    if cohort and cohort.get("subject_id") is not None:
        return int(cohort["subject_id"])
    return None


def _sensitive_detector(*, cr, baseline_cr, uo, symptom, rising) -> tuple[float, bool, list[str]]:
    """Recall-first 민감 탐지 + weak-signal 합산(§4.1/4.4). (score, positive, reasons)."""
    reasons: list[str] = []
    ratio = (cr / baseline_cr) if (cr and baseline_cr and baseline_cr > 0) else None
    delta = (cr - baseline_cr) if (cr is not None and baseline_cr is not None) else None
    sym = (symptom or "").strip()
    has_keyword = any(k in sym for k in _OVERRIDE_KEYWORDS)

    # weak-signal 컴포넌트(KDIGO 보다 민감한 경계).
    cr_comp = 0.0
    if ratio is not None and ratio >= 1.2:
        cr_comp = 0.4; reasons.append(f"Cr {ratio:.2f}× (≥1.2, 민감)")
    elif delta is not None and delta >= 0.2:
        cr_comp = 0.3; reasons.append(f"Cr 증가 {delta:.2f} (≥0.2, 민감)")
    elif rising:
        cr_comp = 0.3; reasons.append("Cr 상승 추세")

    uo_comp = 0.0
    if uo is not None:
        if uo < 0.3:
            uo_comp = 0.5; reasons.append(f"UO {uo:.2f} (무뇨)")
        elif uo < 0.5:
            uo_comp = 0.35; reasons.append(f"UO {uo:.2f} (핍뇨)")
        elif uo < 1.0:
            uo_comp = 0.2; reasons.append(f"UO {uo:.2f} (경도 감소)")  # ANY decrease

    sym_comp = 0.3 if has_keyword else 0.0
    if has_keyword:
        reasons.append(f"임상 키워드 '{sym}'")

    score = min(1.0, cr_comp + uo_comp + sym_comp)
    # §4.1 recall-first OR: 약신호 하나라도 있으면 positive.
    positive = bool(cr_comp or (uo is not None and uo < 1.0) or has_keyword)
    return score, positive, reasons


def _guardrail(prob: float, *, data: dict, baseline_confidence: float, uo, cr, baseline_cr, rising) -> tuple[float, list[str]]:
    """FP 제어(§5) — 비신장성 원인/모순/불확실성에 패널티."""
    notes: list[str] = []
    p = prob
    sym = (data.get("symptom") or "")

    # 5.1 biological plausibility
    if data.get("dehydration"):
        p *= 0.7; notes.append("guardrail: 탈수(prerenal 가능) ×0.7")
    if data.get("contrast_exposure"):
        p *= 0.85; notes.append("guardrail: 조영제 노출 ×0.85")
    if data.get("diuretics"):
        p *= 0.8; notes.append("guardrail: 이뇨제(UO 해석 왜곡) ×0.8")

    # 5.2 temporal consistency — 추세 없는 단발 spike 면 강등.
    if cr is not None and baseline_cr and cr > baseline_cr * 1.5 and not rising:
        p *= 0.85; notes.append("guardrail: 단발 spike(추세 없음) ×0.85")

    # 5.3 contradiction — '소변량 정상' 언급 + 객관 핍뇨 충돌.
    if "소변량 정상" in sym and uo is not None and uo < 0.5:
        p *= 0.85; notes.append("guardrail: '소변량 정상' vs 핍뇨 모순 ×0.85")

    # 5.4 baseline uncertainty penalty
    if baseline_confidence < 0.7:
        penalty = (0.7 - baseline_confidence) * 0.3
        p = max(0.0, p - penalty); notes.append(f"guardrail: baseline 신뢰도 {baseline_confidence:.2f} → -{penalty:.2f}")

    return p, notes


def _final_decision(*, prob: float, threshold: float, kdigo: kdigo_staging.KdigoResult, sensitive_positive: bool) -> str:
    """최종 결정(§6.5 zone + §7 decision engine). 반환: CONFIRMED|SUSPECTED|PRE_AKI|NONE."""
    if kdigo.positive:
        return "CONFIRMED"
    if prob >= threshold:                       # ROC operating point 초과 → AKI likely
        return "SUSPECTED"
    if sensitive_positive or prob >= roc_calibration.PRE_AKI_ZONE:
        return "PRE_AKI"                         # recall-first: 약신호는 최소 pre-AKI
    return "NONE"


def process_lab_event(event: dict) -> None:
    """LAB_EVENT 구독 핸들러 — Safety 파이프라인 → AKI_EVENT 발행."""
    patient_id = event.get("patientId") or event.get("patient_id")
    subject_id = _fill_subject_id(
        patient_id,
        event.get("subjectId") if event.get("subjectId") is not None else event.get("subject_id"),
    )
    if subject_id is None:
        log.info("LAB_EVENT skipped: missing subject_id for patient=%s", patient_id)
        return
    data = _normalize(event.get("data") or {})
    age = event.get("age")
    priors = event.get("priorCreatinines") or event.get("prior_creatinines") or []
    cr = _f(data.get("creatinine"))
    uo = _f(data.get("urine_output"))

    # 1) Baseline (3-layer + confidence)
    baseline = estimate_baseline(
        prior_creatinines=priors, current_cr=cr, age=age,
        sex=data.get("sex"), ckd_risk=data.get("ckd_risk"),
    )

    # base 확률(학습/규칙 예측기) + 추세
    pred = get_aki_predictor().predict(_extract_features(data, baseline.cr))
    p_model = float(pred["risk_score"]) / 100.0
    rising, slope = _detect_trend([*[_f(x) for x in priors], cr])
    rationale = list(pred.get("rationale", []))

    # 2) Sensitive detector (recall-first)
    sens_score, sensitive_positive, sens_reasons = _sensitive_detector(
        cr=cr, baseline_cr=baseline.cr, uo=uo, symptom=data.get("symptom"), rising=rising,
    )
    prob = max(p_model, sens_score)
    rationale.extend(sens_reasons)

    # clinical override (§4.5) — 강제 위험 상승(하한).
    sym = (data.get("symptom") or "")
    if any(k in sym for k in _OVERRIDE_KEYWORDS):
        prob = max(prob, _OVERRIDE_FLOOR)
        rationale.append(f"clinical override: 키워드 → 확률 하한 {_OVERRIDE_FLOOR}")

    # 3) Guardrail (FP↓) — 단, sensitive_positive 는 이후 최소 PRE_AKI 보장.
    prob, guard_notes = _guardrail(
        prob, data=data, baseline_confidence=baseline.confidence, uo=uo, cr=cr,
        baseline_cr=baseline.cr, rising=rising,
    )
    prob = round(min(1.0, max(0.0, prob)), 4)
    rationale.extend(guard_notes)

    # 4) ROC-calibrated decision + 5) KDIGO staging
    threshold = roc_calibration.operating_threshold()
    z = roc_calibration.zone(prob)
    kdigo = kdigo_staging.stage(cr=cr, baseline_cr=baseline.cr,
                                baseline_confidence=baseline.confidence, uo=uo)
    rationale.extend(kdigo.criteria)
    if kdigo.baseline_gated:
        rationale.append("KDIGO Cr-배수 기준 보류(baseline 신뢰도<0.7)")

    # 6) Final decision
    stage = _final_decision(prob=prob, threshold=threshold, kdigo=kdigo, sensitive_positive=sensitive_positive)
    pre_aki = stage == "PRE_AKI"

    model_stage = f"KDIGO Stage {kdigo.stage}" if kdigo.positive else z

    aki_event = {
        "eventType": AKI_EVENT,
        "eventId": new_id("akievt"),
        "patientId": patient_id,
        "baseline": {"cr": baseline.cr, "confidence": baseline.confidence, "source": baseline.source, "reason": baseline.reason},
        "current": {"cr": cr},
        "aki_score": prob,
        "probability": prob,
        "zone": z,
        "threshold": threshold,
        "stage": stage,
        "kdigoStage": kdigo.stage,
        "kdigoPositive": kdigo.positive,
        "sensitivePositive": sensitive_positive,
        "pre_aki": pre_aki,
        "trend_detected": rising,
        "trendSlope": slope,
        "rationale": rationale,
        "modelStage": model_stage,
        "sourceEventId": event.get("eventId"),
        "timestamp": event.get("timestamp"),
    }
    if subject_id is not None:
        aki_event["subjectId"] = subject_id
    log.info("AKI_EVENT patient=%s decision=%s prob=%.2f zone=%s kdigo=%d th=%.2f",
             patient_id, stage, prob, z, kdigo.stage, threshold)
    event_bus.publish(AKI_EVENT, aki_event)
