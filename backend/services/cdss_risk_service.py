"""CDSS Risk Score 서비스 — RULE + MODEL HYBRID (작업지시서 6).

risk_score = 0.4*creatinine_trend + 0.3*urine_output_drop
           + 0.2*diagnosis_risk_weight + 0.1*vitals_instability

금지(6.4)
- LLM 단독 risk score 생성 금지 → 고정 가중치 + 결정론적 컴포넌트.
- 설명 없는 score 금지 → 모든 컴포넌트가 value/weight/contribution/explanation 동반(breakdown).
- single-variable 판단 금지 → 4개 컴포넌트를 항상 산출(부재 시 0 + 사유).

hybrid: diagnosis_risk_weight 컴포넌트에 AKI 예측기(모델/룰) 확률을 주입 → rule+model 결합.

tiers(6.3)
  0.0–0.3 LOW    (log only)
  0.3–0.6 MEDIUM (toast alert)
  0.6–1.0 HIGH   (modal alert + nephrology trigger)
"""
from __future__ import annotations

WEIGHTS = {
    "creatinine_trend": 0.4,
    "urine_output_drop": 0.3,
    "diagnosis_risk_weight": 0.2,
    "vitals_instability": 0.1,
}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _lab(labs: list, key: str):
    for lab in labs:
        if getattr(lab, "key", None) == key:
            return getattr(lab, "value", None)
    return None


class CdssRiskService:
    """SOAP + Problem List + 검사추이/소변량 + AKI 예측을 결합한 위험 산출기."""

    def score(self, patient, soap: dict, problems: list, aki_result: dict) -> dict:
        labs = list(getattr(patient, "labs", []) or [])
        trend = list(getattr(patient, "trend", []) or [])
        urine = list(getattr(patient, "urine_output", []) or [])

        comps = [
            self._creatinine_trend(labs, trend),
            self._urine_output_drop(labs, urine),
            self._diagnosis_risk_weight(aki_result, problems),
            self._vitals_instability(labs),
        ]
        # 가중 합(0~1).
        score = _clamp01(sum(c["contribution"] for c in comps))
        tier, action, alert, trigger = self._tier(score)
        return {
            "risk_score": round(score, 4),
            "tier": tier,
            "alert": alert,            # modal | toast | log
            "action": action,
            "nephrology_trigger": trigger,
            "weights": WEIGHTS,
            "breakdown": comps,
            "explanation": "; ".join(f"{c['component']}={c['value']}({c['explanation']})" for c in comps),
        }

    # ---------- 컴포넌트(각각 value 0~1 + 설명) ----------
    def _component(self, name, value, explanation) -> dict:
        v = _clamp01(value)
        return {
            "component": name,
            "value": round(v, 4),
            "weight": WEIGHTS[name],
            "contribution": round(v * WEIGHTS[name], 4),
            "explanation": explanation,
        }

    def _creatinine_trend(self, labs, trend) -> dict:
        cr_now = _lab(labs, "cr")
        baseline = trend[0].creatinine if trend else None
        cr_max = max([t.creatinine for t in trend] + ([cr_now] if cr_now else []), default=None)
        if cr_max and baseline and baseline > 0:
            ratio = cr_max / baseline
            # KDIGO: 1.5/2.0/3.0배 → 0.5/0.75/1.0 근사.
            val = (ratio - 1.0) / 2.0
            return self._component("creatinine_trend", val,
                                   f"Cr {baseline}→{cr_max} ({ratio:.1f}배)")
        if cr_now:
            val = (cr_now - 1.0) / 3.0  # baseline 없을 때 절대값 근사.
            return self._component("creatinine_trend", val, f"Cr {cr_now} (baseline 미상)")
        return self._component("creatinine_trend", 0.0, "Cr 데이터 없음")

    def _urine_output_drop(self, labs, urine) -> dict:
        uo = urine[-1].value if urine else None
        if uo is None:
            return self._component("urine_output_drop", 0.0, "소변량 데이터 없음")
        if uo < 0.3:
            return self._component("urine_output_drop", 1.0, f"{uo} mL/kg/h (무뇨)")
        if uo < 0.5:
            return self._component("urine_output_drop", 0.7, f"{uo} mL/kg/h (핍뇨)")
        if uo < 1.0:
            return self._component("urine_output_drop", 0.3, f"{uo} mL/kg/h (경도 감소)")
        return self._component("urine_output_drop", 0.0, f"{uo} mL/kg/h (정상)")

    def _diagnosis_risk_weight(self, aki_result, problems) -> dict:
        # MODEL 부분: AKI 예측 확률(Stage2+3 가중).
        p23 = float(aki_result.get("p_stage2_plus", 0.0) or 0.0)
        p1 = float(aki_result.get("p_stage1", 0.0) or 0.0)
        val = p23 * 1.0 + p1 * 0.5
        n_prob = len(problems or [])
        val = min(1.0, val + 0.05 * n_prob)  # 문제 수 소폭 가산.
        return self._component(
            "diagnosis_risk_weight", val,
            f"AKI 모델 P(S2+3)={p23:.2f},P(S1)={p1:.2f}, problems={n_prob} [{aki_result.get('source')}]"
        )

    def _vitals_instability(self, labs) -> dict:
        k = _lab(labs, "k")
        hco3 = _lab(labs, "hco3")
        score, notes = 0.0, []
        if k is not None and k >= 6.0:
            score += 0.6
            notes.append(f"K {k}(고칼륨)")
        if hco3 is not None and hco3 < 18:
            score += 0.4
            notes.append(f"HCO3 {hco3}(산증)")
        if not notes:
            notes.append("불안정 지표 없음/데이터 제한")
        return self._component("vitals_instability", score, ", ".join(notes))

    # ---------- tier ----------
    def _tier(self, score: float):
        if score >= 0.6:
            return "HIGH", "modal alert + nephrology trigger", "modal", True
        if score >= 0.3:
            return "MEDIUM", "toast alert", "toast", False
        return "LOW", "log only", "log", False
