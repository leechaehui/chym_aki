"""SOAP / A·P evidence / Risk score 검증기 (작업지시서 8 validator/, 9 VALIDATION RULES).

세 검증기를 한 모듈에 둔다(작업지시서의 validator 구성):
- SoapValidator        : S/O/A/P 존재, empty 금지, A/P evidence 필수.
- ApEvidenceValidator  : A/P 의 모든 evidence-backed 진술 검사 + hallucination 탐지.
- RiskScoreValidator   : score 설명가능성/컴포넌트 breakdown/가중치 합/재현성.

모든 검증기는 ValidationOutcome(passed, errors, warnings)을 반환한다(예외 던지지 않음).
오케스트레이터가 결과를 모아 위반 시 차단/표기한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from services.soap_service import NOT_SUFFICIENT


@dataclass
class ValidationOutcome:
    name: str
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.passed = False
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def as_dict(self) -> dict:
        return {
            "validator": self.name,
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class SoapValidator:
    """S/O/A/P 구조·존재·근거 규칙(9 SOAP validation)."""

    def validate(self, soap: dict) -> ValidationOutcome:
        out = ValidationOutcome("soap_validator")
        for key in ("S", "O", "A", "P"):
            if key not in soap:
                out.fail(f"필수 섹션 누락: {key}")

        # S/O: 문자열·비어있지 않음.
        for key in ("S", "O"):
            val = soap.get(key)
            if not isinstance(val, str) or not val.strip():
                out.fail(f"{key} 가 비어있음(empty string 금지)")

        # A/P: dict + 내부 텍스트 존재 + (내용 있으면) evidence 필수.
        for key, text_key in (("A", "assessment"), ("P", "plan")):
            section = soap.get(key)
            if not isinstance(section, dict):
                out.fail(f"{key} 는 객체여야 함")
                continue
            text = section.get(text_key, "")
            if not text or not str(text).strip():
                out.fail(f"{key}.{text_key} 비어있음")
                continue
            if text != NOT_SUFFICIENT and not section.get("evidence"):
                out.fail(f"{key} 에 내용이 있으나 evidence 없음(RULE 1 위반)")
        return out


class ApEvidenceValidator:
    """A/P hallucination 탐지 — 모든 진술은 evidence 를 가져야 한다(4.2 RULE 1)."""

    def validate(self, soap: dict) -> ValidationOutcome:
        out = ValidationOutcome("ap_evidence_validator")
        for key, text_key in (("A", "assessment"), ("P", "plan")):
            section = soap.get(key, {})
            if not isinstance(section, dict):
                out.fail(f"{key} 구조 오류")
                continue
            text = section.get(text_key, "")
            statements = section.get("statements", [])
            if text == NOT_SUFFICIENT:
                # 근거 부족 fallback 은 정상(RULE 2). statements 비어야 일관.
                if statements:
                    out.warn(f"{key}: NOT_SUFFICIENT 인데 statements 존재")
                continue
            if not statements:
                out.fail(f"{key}: 내용이 있으나 statements(근거 매핑) 없음 → hallucination 의심")
            for i, st in enumerate(statements):
                if not st.get("evidence"):
                    out.fail(f"{key}.statements[{i}] '{st.get('statement','')[:30]}' 근거 없음(hallucination)")
        return out


class RiskScoreValidator:
    """risk score 설명가능성·재현성(9 CDSS validation, 6.4)."""

    TOLERANCE = 1e-6

    def validate(self, risk: dict) -> ValidationOutcome:
        out = ValidationOutcome("risk_score_validator")
        score = risk.get("risk_score")
        breakdown = risk.get("breakdown")
        weights = risk.get("weights")

        if not isinstance(score, (int, float)):
            out.fail("risk_score 가 수치가 아님")
            return out
        if not (0.0 <= score <= 1.0):
            out.fail(f"risk_score 범위 이탈: {score}")
        if not breakdown:
            out.fail("breakdown 없음(설명 없는 score 금지)")
            return out

        # 모든 컴포넌트가 설명/값/가중치를 가져야 함.
        for c in breakdown:
            if not c.get("explanation"):
                out.fail(f"컴포넌트 {c.get('component')} 설명 없음")
            for f in ("value", "weight", "contribution"):
                if f not in c:
                    out.fail(f"컴포넌트 {c.get('component')} {f} 누락")

        # single-variable 판단 금지 → 컴포넌트 4개 모두 존재.
        comp_names = {c.get("component") for c in breakdown}
        if weights and comp_names != set(weights):
            out.fail(f"컴포넌트 불일치: {comp_names} vs {set(weights or [])}")

        # 가중치 합 = 1.0.
        if weights and abs(sum(weights.values()) - 1.0) > 1e-6:
            out.fail(f"가중치 합 != 1.0 ({sum(weights.values())})")

        # 재현성: Σ contribution == score(클램프 고려).
        recomputed = min(1.0, max(0.0, sum(c.get("contribution", 0.0) for c in breakdown)))
        if abs(recomputed - score) > self.TOLERANCE:
            out.fail(f"score 비재현: breakdown 합 {recomputed} != {score}")
        return out


def run_all(soap: dict, risk: dict) -> dict:
    """세 검증기 일괄 실행 → 종합 리포트."""
    results = [
        SoapValidator().validate(soap),
        ApEvidenceValidator().validate(soap),
        RiskScoreValidator().validate(risk),
    ]
    return {
        "passed": all(r.passed for r in results),
        "validators": [r.as_dict() for r in results],
    }
