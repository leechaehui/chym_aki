"""임상 타당성 검증 (작업지시서 4.2).

예측기가 '의학적으로 말이 되는' 방향성을 갖는지 계약 수준에서 검증한다.
검증 대상은 ai_draft.aki_model 의 AkiPredictor 계약(predict(features)->dict).

답해야 하는 질문(4.2):
  1. Cr/eGFR 변화가 AKI prediction 에 실제로 반영되는가?  → monotonic_creatinine / monotonic_egfr
  2. urine output trend 가 위험도 변화에 영향을 주는가?     → monotonic_urine
  3. 약물 정보 포함/미포함 성능 차이 정량화                  → (offline은 subgroup.feature_ablation)

각 검사는 '다른 조건 고정, 한 축만 변화' 시 위험점수의 단조성을 본다.
통과 기준은 strict 단조가 아니라 '비감소/비증가'(동률 허용)로, 임상 직관과 일치한다.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MonotonicityCheck:
    name: str
    description: str
    axis_values: list[float]
    risk_scores: list[int]
    passed: bool
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "check": self.name,
            "description": self.description,
            "axis_values": self.axis_values,
            "risk_scores": self.risk_scores,
            "passed": self.passed,
            "note": self.note,
        }


def _is_non_decreasing(xs: list[float]) -> bool:
    return all(b >= a for a, b in zip(xs, xs[1:]))


def _is_non_increasing(xs: list[float]) -> bool:
    return all(b <= a for a, b in zip(xs, xs[1:]))


def _base_features() -> dict:
    """경계가 아닌 '대표 환자' 기본 피처. 각 검사에서 한 축만 덮어쓴다."""
    return {
        "baseline_creatinine": 1.0,
        "creatinine_min": 1.0,
        "creatinine_max": 1.0,
        "creatinine_delta": 0.0,
        "egfr": 90.0,
        "urine_ml_kg_hr": 1.0,
        "oliguria_flag": 0,
        "potassium_max": 4.0,
        "bicarbonate_min": 24.0,
        "bun_max": 15.0,
        "age": 60,
    }


def check_creatinine_monotonic(predictor) -> MonotonicityCheck:
    """Cr_max 를 baseline 대비 1.0→3.5배로 올리면 위험점수가 비감소해야 한다."""
    ratios = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    scores = []
    for r in ratios:
        f = _base_features()
        f["creatinine_max"] = round(1.0 * r, 2)
        f["creatinine_delta"] = round(f["creatinine_max"] - 1.0, 2)
        scores.append(predictor.predict(f)["risk_score"])
    return MonotonicityCheck(
        name="creatinine_monotonic",
        description="Cr 상승(baseline 대비 배수↑) → 위험점수 비감소",
        axis_values=ratios,
        risk_scores=scores,
        passed=_is_non_decreasing([float(s) for s in scores]),
        note="KDIGO: Cr 1.5/2.0/3.0배는 Stage 1/2/3 기준",
    )


def check_egfr_monotonic(predictor) -> MonotonicityCheck:
    """eGFR 를 90→10 으로 낮추면 위험점수가 비감소해야 한다."""
    egfrs = [90.0, 60.0, 45.0, 30.0, 20.0, 10.0]
    scores = []
    for e in egfrs:
        f = _base_features()
        f["egfr"] = e
        scores.append(predictor.predict(f)["risk_score"])
    return MonotonicityCheck(
        name="egfr_monotonic",
        description="eGFR 저하 → 위험점수 비감소",
        axis_values=egfrs,
        risk_scores=scores,
        passed=_is_non_decreasing([float(s) for s in scores]),
        note="eGFR<30/<15 에서 가중",
    )


def check_urine_monotonic(predictor) -> MonotonicityCheck:
    """시간당 소변량을 1.5→0.1 mL/kg/h 로 낮추면 위험점수가 비감소해야 한다."""
    uos = [1.5, 1.0, 0.6, 0.5, 0.3, 0.1]
    scores = []
    for u in uos:
        f = _base_features()
        f["urine_ml_kg_hr"] = u
        f["oliguria_flag"] = 1 if u < 0.5 else 0
        scores.append(predictor.predict(f)["risk_score"])
    return MonotonicityCheck(
        name="urine_output_monotonic",
        description="소변량 감소(핍뇨/무뇨) → 위험점수 비감소",
        axis_values=uos,
        risk_scores=scores,
        passed=_is_non_decreasing([float(s) for s in scores]),
        note="KDIGO 소변량 기준 <0.5(핍뇨)/<0.3(무뇨)",
    )


def run_clinical_checks(predictor) -> list[MonotonicityCheck]:
    """4.2 임상 타당성 일괄 실행."""
    return [
        check_creatinine_monotonic(predictor),
        check_egfr_monotonic(predictor),
        check_urine_monotonic(predictor),
    ]
