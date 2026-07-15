"""Validator — soap / ap_evidence / risk_score (작업지시서 8·9·10).

hallucination detection(rule-based): 근거 없는 A/P 진술을 검증기가 반드시 잡아낸다.
"""
from services.soap_service import NOT_SUFFICIENT
from validator.soap_validator import (
    ApEvidenceValidator,
    RiskScoreValidator,
    SoapValidator,
    run_all,
)


def _valid_soap():
    return {
        "S": "다리가 부어요",
        "O": "Creatinine 4.2 mg/dL",
        "A": {"assessment": "AKI 평가: Stage 3", "evidence": ["Cr 3.0배"],
              "statements": [{"statement": "AKI 평가: Stage 3", "evidence": ["Cr 3.0배"]}]},
        "P": {"plan": "신장내과 협진", "evidence": ["Cr 3.0배"],
              "statements": [{"statement": "신장내과 협진", "evidence": ["Cr 3.0배"]}]},
    }


def _valid_risk():
    weights = {"creatinine_trend": 0.4, "urine_output_drop": 0.3,
               "diagnosis_risk_weight": 0.2, "vitals_instability": 0.1}
    breakdown = [
        {"component": "creatinine_trend", "value": 1.0, "weight": 0.4, "contribution": 0.4, "explanation": "Cr↑"},
        {"component": "urine_output_drop", "value": 1.0, "weight": 0.3, "contribution": 0.3, "explanation": "무뇨"},
        {"component": "diagnosis_risk_weight", "value": 0.5, "weight": 0.2, "contribution": 0.1, "explanation": "model"},
        {"component": "vitals_instability", "value": 0.0, "weight": 0.1, "contribution": 0.0, "explanation": "없음"},
    ]
    return {"risk_score": 0.8, "weights": weights, "breakdown": breakdown}


def test_valid_soap_passes():
    assert SoapValidator().validate(_valid_soap()).passed


def test_missing_section_fails():
    soap = _valid_soap()
    del soap["P"]
    assert not SoapValidator().validate(soap).passed


def test_empty_string_fails():
    soap = _valid_soap()
    soap["S"] = "   "
    assert not SoapValidator().validate(soap).passed


def test_assessment_with_content_but_no_evidence_fails():
    soap = _valid_soap()
    soap["A"]["evidence"] = []
    assert not SoapValidator().validate(soap).passed


def test_not_sufficient_is_allowed():
    soap = _valid_soap()
    soap["A"] = {"assessment": NOT_SUFFICIENT, "evidence": [], "statements": []}
    assert SoapValidator().validate(soap).passed


def test_ap_evidence_validator_detects_hallucination():
    soap = _valid_soap()
    # 근거 없는 가짜 진술 주입 → hallucination.
    soap["A"]["statements"].append({"statement": "패혈증", "evidence": []})
    out = ApEvidenceValidator().validate(soap)
    assert not out.passed
    assert any("hallucination" in e for e in out.errors)


def test_risk_validator_passes_reproducible():
    assert RiskScoreValidator().validate(_valid_risk()).passed


def test_risk_validator_detects_non_reproducible_score():
    risk = _valid_risk()
    risk["risk_score"] = 0.99  # breakdown 합(0.8)과 불일치
    out = RiskScoreValidator().validate(risk)
    assert not out.passed
    assert any("재현" in e or "비재현" in e for e in out.errors)


def test_risk_validator_detects_bad_weights():
    risk = _valid_risk()
    risk["weights"] = {**risk["weights"], "creatinine_trend": 0.9}
    assert not RiskScoreValidator().validate(risk).passed


def test_risk_validator_requires_all_components():
    risk = _valid_risk()
    risk["breakdown"] = risk["breakdown"][:1]  # single-variable
    assert not RiskScoreValidator().validate(risk).passed


def test_run_all_aggregates():
    report = run_all(_valid_soap(), _valid_risk())
    assert report["passed"]
    assert len(report["validators"]) == 3
