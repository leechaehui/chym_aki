"""Service — SOAP strict controlled generation (작업지시서 4, 10).

검증 핵심: S/O 는 추출만, A/P 는 근거 필수, 근거 없으면 'Not sufficient information'.
"""
from nlp.rule_based_strategy import RuleBasedNlpStrategy
from services.soap_service import NOT_SUFFICIENT, SoapService


class _Lab:
    def __init__(self, key, value):
        self.key, self.value = key, value


class _Urine:
    def __init__(self, value):
        self.value = value


class _Patient:
    def __init__(self, **kw):
        self.diagnosis = kw.get("diagnosis", "당뇨병성 신증")
        self.sex = kw.get("sex", "M")
        self.age = kw.get("age", 65)
        self.labs = kw.get("labs", [])
        self.urine_output = kw.get("urine_output", [])
        self.trend = kw.get("trend", [])


def _aki(stage_num=2, rationale=None):
    return {
        "stage": "AKI Stage 2-3", "risk_label": "고위험", "stage_num": stage_num,
        "risk_score": 80, "p_stage1": 0.2, "p_stage2_plus": 0.6,
        "rationale": rationale or ["Cr 3.0배 상승 (KDIGO Stage 3 기준)"],
    }


def _symptoms(transcript):
    return RuleBasedNlpStrategy().extract_symptoms(transcript)


def test_subjective_is_extraction_not_invention():
    svc = SoapService()
    t = "다리가 붓고 소변이 줄었어요"
    soap = svc.generate(_Patient(), t, _symptoms(t), _aki())
    assert t in soap["S"]  # 전사 원문 보존
    # 환자가 말하지 않은 증상은 S 에 없어야 한다.
    assert "흉통" not in soap["S"]


def test_subjective_empty_transcript_no_symptoms_is_not_sufficient():
    svc = SoapService()
    soap = svc.generate(_Patient(), "", _symptoms(""), _aki(stage_num=0))
    assert soap["S"] == NOT_SUFFICIENT


def test_objective_only_contains_present_values():
    svc = SoapService()
    p = _Patient(labs=[_Lab("cr", 4.2), _Lab("k", 6.3)])
    soap = svc.generate(p, "", _symptoms(""), _aki())
    assert "Creatinine 4.2" in soap["O"]
    assert "K 6.3" in soap["O"]
    assert "eGFR" not in soap["O"]  # egfr 미제공 → 표기 안 함(추출만)


def test_assessment_requires_evidence():
    svc = SoapService()
    p = _Patient(labs=[_Lab("cr", 4.2)])
    t = "다리가 너무 부어요"
    soap = svc.generate(p, t, _symptoms(t), _aki())
    A = soap["A"]
    assert A["assessment"] != NOT_SUFFICIENT
    assert A["evidence"]  # 근거 존재
    # 모든 statement 가 evidence 를 가진다.
    assert all(st["evidence"] for st in A["statements"])


def test_assessment_no_evidence_falls_back():
    """전사 무증상 + AKI 없음 + 진단 없음 → A 는 Not sufficient information."""
    svc = SoapService()
    p = _Patient(diagnosis="")  # 진단 근거 제거
    soap = svc.generate(p, "별 말 없음", _symptoms("별 말 없음"), _aki(stage_num=0))
    assert soap["A"]["assessment"] == NOT_SUFFICIENT
    assert soap["A"]["evidence"] == []
    assert soap["A"]["statements"] == []


def test_plan_evidence_for_hyperkalemia():
    svc = SoapService()
    p = _Patient(labs=[_Lab("k", 6.5)])
    soap = svc.generate(p, "", _symptoms(""), _aki(stage_num=1))
    plan = soap["P"]
    assert "고칼륨" in plan["plan"]
    assert any("K 6.5" in e for e in plan["evidence"])


def test_no_hallucinated_symptom_in_assessment():
    """환자 말에 없는 증상(흉통)이 A 근거로 들어가면 안 된다."""
    svc = SoapService()
    p = _Patient(labs=[_Lab("cr", 4.2)])
    t = "소변이 줄었어요"  # 흉통 언급 없음
    soap = svc.generate(p, t, _symptoms(t), _aki())
    blob = str(soap["A"])
    assert "흉통" not in blob and "chest" not in blob
