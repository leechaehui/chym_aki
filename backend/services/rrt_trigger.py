"""RRT(신대체요법) 트리거 — 임상 의사결정 보조(예측 모델 아님).

핵심 철학: "투석을 예측하는 AI"가 아니라 **"투석이 논의되어야 하는 순간을 잡아주는
임상 트리거 엔진"**이다. KDIGO + ICU 규칙으로 RRT '논의/고려/긴급 평가' 단계를 분류할 뿐,
"RRT 필요/확률" 같은 단정은 절대 하지 않는다(최종 결정은 신장내과/ICU 의료진).

설계(GoF·SOLID):
  - `RrtSignals` : 판정 입력(가용 신호만; 미측정은 None — 절대 임의 보정/날조 금지).
  - `RrtRule`    : (조건 평가 → 트리거 단계 + 근거) 규칙 하나. Strategy/Chain 형태로 누적.
  - `assess_rrt_trigger` : 규칙들을 적용해 최고 단계를 산출(추세 가중 포함).

이 코호트(MIMIC-IV 스케일 피처)에는 전해질(K/Na)·산염기(pH/HCO3) 실수치가 없으므로
해당 기준은 평가하지 않고 `unavailable_criteria` 로 정직하게 표기한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class RrtTriggerLevel(IntEnum):
    """RRT 트리거 단계(높을수록 긴급)."""

    NONE = 0       # 임상 신호 없음
    DISCUSSION = 1  # 신장내과 논의 시작
    CONSIDERATION = 2  # 투석 가능성 실제 논의
    URGENT = 3     # 생명 위협, 즉시 평가


# 단계 → 라벨/행동(안전 문구만 사용 — "필요/확률" 금지).
_LEVEL_LABEL = {
    RrtTriggerLevel.NONE: "현재 RRT 관련 임상 신호 없음",
    RrtTriggerLevel.DISCUSSION: "RRT DISCUSSION RECOMMENDED",
    RrtTriggerLevel.CONSIDERATION: "RRT CONSIDERATION",
    RrtTriggerLevel.URGENT: "RRT URGENT / EMERGENT",
}
_LEVEL_ACTION = {
    RrtTriggerLevel.NONE: "정기 모니터링 유지",
    RrtTriggerLevel.DISCUSSION: "신장내과 협진(논의) 권장",
    RrtTriggerLevel.CONSIDERATION: "긴급 신장내과 평가 권장",
    RrtTriggerLevel.URGENT: "즉시 ICU/신장내과 평가 — RRT 개시 가능성 논의 필요",
}

_WARNING_NOTE = "RRT 필요 여부를 예측하지 않습니다. 본 신호는 임상 의사결정 보조 용도이며 최종 결정은 의료진이 합니다."
ENGINE_TAGLINE = (
    "이 시스템은 '투석을 예측하는 AI'가 아니라 "
    "'투석이 논의되어야 하는 순간을 잡아주는 임상 트리거 엔진'입니다."
)


@dataclass(frozen=True)
class RrtSignals:
    """RRT 트리거 판정 입력 — 가용 신호만 채우고 미측정은 None 유지."""

    # 신기능(Cr / eGFR) — 이 코호트에서 산출 가능
    creatinine_current: float | None = None
    creatinine_baseline: float | None = None
    creatinine_delta_24h: float | None = None
    creatinine_ratio_to_baseline: float | None = None
    egfr_current: float | None = None
    egfr_declining: bool = False
    # 소변량 — urine_rate 가 있을 때만
    urine_min_6h_ml_kg_h: float | None = None
    urine_min_12h_ml_kg_h: float | None = None
    anuria_hours: float | None = None
    # 미가용(코호트에 실수치 없음): 전해질·산염기·임상맥락
    potassium: float | None = None
    ph: float | None = None
    bicarbonate: float | None = None


@dataclass(frozen=True)
class RrtAssessment:
    """RRT 트리거 판정 결과(strict 출력)."""

    rrt_level: str
    label: str
    confidence: float
    key_drivers: list[str] = field(default_factory=list)
    trend_signal: str = "stable"  # stable | worsening | rapidly deteriorating
    clinical_action: str = ""
    warning_note: str = _WARNING_NOTE
    unavailable_criteria: list[str] = field(default_factory=list)


def _renal_drivers(s: RrtSignals) -> list[tuple[RrtTriggerLevel, str]]:
    """신기능 기준 → (단계, 근거) 목록."""
    found: list[tuple[RrtTriggerLevel, str]] = []
    if s.creatinine_delta_24h is not None and s.creatinine_delta_24h >= 0.3:
        found.append((RrtTriggerLevel.DISCUSSION, f"Cr 24h 상승 +{s.creatinine_delta_24h:.2f} mg/dL (≥0.3)"))
    if s.creatinine_ratio_to_baseline is not None:
        if s.creatinine_ratio_to_baseline >= 2.0:
            found.append((RrtTriggerLevel.CONSIDERATION, f"Cr 기저치 대비 {s.creatinine_ratio_to_baseline:.1f}배 (48h 내 2배↑)"))
        elif s.creatinine_ratio_to_baseline >= 1.5:
            found.append((RrtTriggerLevel.DISCUSSION, f"Cr 기저치 대비 {s.creatinine_ratio_to_baseline:.1f}배 (≥1.5)"))
    if s.egfr_declining:
        found.append((RrtTriggerLevel.DISCUSSION, "eGFR 지속 감소 추세"))
    return found


def _urine_drivers(s: RrtSignals) -> list[tuple[RrtTriggerLevel, str]]:
    """소변량 기준 → (단계, 근거) 목록."""
    found: list[tuple[RrtTriggerLevel, str]] = []
    if s.anuria_hours is not None and s.anuria_hours >= 12:
        found.append((RrtTriggerLevel.URGENT, f"무뇨 {s.anuria_hours:.0f}h (>12h)"))
    if s.urine_min_12h_ml_kg_h is not None and s.urine_min_12h_ml_kg_h < 0.3:
        found.append((RrtTriggerLevel.CONSIDERATION, f"핍뇨 <0.3 mL/kg/h ≥12h ({s.urine_min_12h_ml_kg_h:.2f})"))
    elif s.urine_min_6h_ml_kg_h is not None and s.urine_min_6h_ml_kg_h < 0.5:
        found.append((RrtTriggerLevel.DISCUSSION, f"핍뇨 <0.5 mL/kg/h ≥6h ({s.urine_min_6h_ml_kg_h:.2f})"))
    return found


def _trend_signal(s: RrtSignals) -> str:
    """추세 신호 — 단일값보다 '궤적/속도' 우선(지시서 §5).

    rapidly deteriorating 는 정적 임계(예: Cr 2배)만으로 판정하지 않고 **빠른 상승 속도**
    (24h ΔCr 큰 값)나 무뇨처럼 명확한 악화 궤적일 때만 둔다(단계 산정과 이중계산 방지).
    """
    fast_cr_rise = (s.creatinine_delta_24h is not None and s.creatinine_delta_24h >= 0.5)
    if (s.anuria_hours is not None and s.anuria_hours >= 12) or fast_cr_rise:
        return "rapidly deteriorating"
    if s.egfr_declining or (s.creatinine_delta_24h is not None and s.creatinine_delta_24h >= 0.3) \
       or (s.creatinine_ratio_to_baseline is not None and s.creatinine_ratio_to_baseline >= 1.5):
        return "worsening"
    return "stable"


def _unavailable_criteria(s: RrtSignals) -> list[str]:
    """이 코호트에서 평가하지 못한 기준(정직성 표기)."""
    missing = []
    if s.potassium is None:
        missing.append("전해질(K/Na) — 코호트 실수치 없음")
    if s.ph is None or s.bicarbonate is None:
        missing.append("산-염기(pH/HCO3) — 코호트 실수치 없음")
    missing.append("임상맥락(폐부종·요독증상·이뇨 반응) — 미수집")
    return missing


def assess_rrt_trigger(signals: RrtSignals) -> RrtAssessment:
    """가용 신호로 RRT 트리거 단계를 산출(예측 아님·트리거 분류)."""
    candidates = _renal_drivers(signals) + _urine_drivers(signals)
    level = max((lvl for lvl, _ in candidates), default=RrtTriggerLevel.NONE)
    drivers = [reason for lvl, reason in candidates if lvl == level] or \
              [reason for _, reason in candidates]

    trend = _trend_signal(signals)
    # 추세 가중(지시서 §5): 빠른 악화면 DISCUSSION→CONSIDERATION 한 단계만 상향.
    # URGENT 는 무뇨 같은 '경성(hard)' 기준에서만 직접 부여하고, 추세로는 자동 격상하지 않는다
    # (정적 기준이 이미 잡은 신호로 이중 격상하지 않도록).
    if trend == "rapidly deteriorating" and level == RrtTriggerLevel.DISCUSSION:
        level = RrtTriggerLevel.CONSIDERATION
        drivers = drivers + ["빠른 악화 추세로 단계 상향"]

    if not candidates:
        drivers = ["유의한 RRT 임상 신호 없음"]

    # 신뢰도: 평가 가능한 입력 비중(전해질/산염기 부재로 1.0 도달 불가).
    available = sum(x is not None for x in (
        signals.creatinine_current, signals.egfr_current,
        signals.urine_min_6h_ml_kg_h, signals.creatinine_baseline,
    ))
    confidence = round(min(0.85, 0.35 + 0.125 * available), 2)

    return RrtAssessment(
        rrt_level=RrtTriggerLevel(level).name,
        label=_LEVEL_LABEL[RrtTriggerLevel(level)],
        confidence=confidence,
        key_drivers=drivers,
        trend_signal=trend,
        clinical_action=_LEVEL_ACTION[RrtTriggerLevel(level)],
        warning_note=_WARNING_NOTE,
        unavailable_criteria=_unavailable_criteria(signals),
    )
