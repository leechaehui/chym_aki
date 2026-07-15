"""Service — Problem List (작업지시서 5, 10): A 기반만, hallucination 금지."""
from services.problem_list_service import ProblemListService
from services.soap_service import NOT_SUFFICIENT


def _soap_with_a(statements):
    evidence = [e for st in statements for e in st["evidence"]]
    return {
        "S": "x", "O": "y",
        "A": {"assessment": " · ".join(s["statement"] for s in statements),
              "evidence": evidence, "statements": statements},
        "P": {"plan": "z", "evidence": ["e"], "statements": [{"statement": "z", "evidence": ["e"]}]},
    }


def test_derive_from_assessment_only():
    soap = _soap_with_a([
        {"statement": "AKI 평가: Stage 3", "evidence": ["Cr 3.0배"]},
        {"statement": "고칼륨혈증 가능성", "evidence": ["K 6.3"]},
    ])
    problems = ProblemListService().derive(soap)
    names = [p["problem"] for p in problems]
    assert "Acute kidney injury (N17)" in names
    assert "Hyperkalemia (E87.5)" in names
    assert all(p["source"] == "A" for p in problems)
    assert all(0.0 <= p["confidence"] <= 1.0 for p in problems)


def test_not_sufficient_assessment_yields_empty():
    soap = {"S": "x", "O": "y",
            "A": {"assessment": NOT_SUFFICIENT, "evidence": [], "statements": []},
            "P": {"plan": NOT_SUFFICIENT, "evidence": [], "statements": []}}
    assert ProblemListService().derive(soap) == []


def test_statement_without_evidence_is_not_promoted():
    """근거 없는 A 진술은 문제로 승격되지 않는다(hallucination 차단)."""
    soap = _soap_with_a([{"statement": "AKI 평가: Stage 3", "evidence": ["Cr 3.0배"]}])
    # 근거 없는 가짜 statement 주입.
    soap["A"]["statements"].append({"statement": "패혈증 추정", "evidence": []})
    problems = ProblemListService().derive(soap)
    assert all("패혈증" not in p["problem"] for p in problems)


def test_confidence_increases_with_evidence_count():
    one = _soap_with_a([{"statement": "AKI 평가", "evidence": ["a"]}])
    many = _soap_with_a([{"statement": "AKI 평가", "evidence": ["a", "b", "c"]}])
    c1 = ProblemListService().derive(one)[0]["confidence"]
    c3 = ProblemListService().derive(many)[0]["confidence"]
    assert c3 > c1
