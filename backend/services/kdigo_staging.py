"""KDIGO AKI 병기 엔진 (지시서 §2, §7).

KDIGO 2012 진단/병기 기준을 결정론적으로 적용한다(설명 동반):
  Stage 1: Cr ≥1.5× baseline  OR  Cr 증가 ≥0.3 mg/dL(48h)  OR  UO <0.5 (6–12h)
  Stage 2: Cr ≥2.0× baseline                                OR  UO <0.5 (≥12h)
  Stage 3: Cr ≥3.0× baseline  OR  Cr ≥4.0 mg/dL              OR  UO <0.3 (≥24h)/무뇨

baseline 불확실성 게이트(지시서 §5.4): baseline confidence < 0.7 이면 Cr-배수 기반
escalation 을 보류한다(절대치/UO 기준은 baseline 무관하므로 유지) — 과진단 억제.
"""
from __future__ import annotations

from dataclasses import dataclass

_BASELINE_CONF_GATE = 0.7


@dataclass(frozen=True)
class KdigoResult:
    stage: int              # 0(non) ~ 3
    positive: bool          # stage >= 1
    criteria: list[str]     # 충족 기준(설명)
    baseline_gated: bool    # baseline 불확실로 Cr-배수 기준 보류 여부


def stage(
    *,
    cr: float | None,
    baseline_cr: float | None,
    baseline_confidence: float,
    uo: float | None,
) -> KdigoResult:
    """KDIGO 병기 산출. Cr-배수는 baseline 신뢰도 게이트를 통과해야 적용."""
    criteria: list[str] = []
    s = 0
    gated = False

    ratio = (cr / baseline_cr) if (cr and baseline_cr and baseline_cr > 0) else None
    delta = (cr - baseline_cr) if (cr is not None and baseline_cr is not None) else None
    use_ratio = baseline_confidence >= _BASELINE_CONF_GATE

    # --- Cr 배수 기준(baseline 신뢰도 게이트 적용) ---
    if ratio is not None:
        if not use_ratio:
            gated = True  # baseline 불확실 → 배수 기준 보류(절대치/UO 로만 판단)
        else:
            if ratio >= 3.0:
                s = max(s, 3); criteria.append(f"Cr {ratio:.1f}× baseline (≥3.0, Stage 3)")
            elif ratio >= 2.0:
                s = max(s, 2); criteria.append(f"Cr {ratio:.1f}× baseline (≥2.0, Stage 2)")
            elif ratio >= 1.5:
                s = max(s, 1); criteria.append(f"Cr {ratio:.1f}× baseline (≥1.5, Stage 1)")

    # --- Cr 증가/절대치(baseline 무관) ---
    if delta is not None and delta >= 0.3 and use_ratio:
        s = max(s, 1); criteria.append(f"Cr 증가 {delta:.1f} mg/dL (≥0.3, 48h, Stage 1)")
    if cr is not None and cr >= 4.0:
        s = max(s, 3); criteria.append(f"Cr 절대치 {cr:.1f} mg/dL (≥4.0, Stage 3)")

    # --- 소변량(baseline 무관) ---
    if uo is not None:
        if uo < 0.3:
            s = max(s, 3); criteria.append(f"무뇨/중증 핍뇨 {uo:.2f} mL/kg/h (<0.3, Stage 3)")
        elif uo < 0.5:
            s = max(s, 1); criteria.append(f"핍뇨 {uo:.2f} mL/kg/h (<0.5, Stage 1)")

    return KdigoResult(stage=s, positive=s >= 1, criteria=criteria, baseline_gated=gated)
