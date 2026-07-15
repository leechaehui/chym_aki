"""Problem List 서비스 — SOAP A → 문제 목록 (작업지시서 5).

규칙
- **A(assessment) 기반으로만** 생성. O 단독 생성 금지.
- ICD-like abstraction 허용(고정 매핑) — 단, 외부 추측(external guessing) 금지.
- hallucinated condition 금지: A 의 evidence-backed statement 만 문제로 승격.

출력: [{"problem": str, "source": "A", "confidence": 0.0~1.0, "evidence": [...]}]
"""
from __future__ import annotations

from services.soap_service import NOT_SUFFICIENT

# 고정 ICD-like 추상화 사전(키워드 → 표준 문제명). 외부 추측 아님(폐쇄 매핑).
_ICD_ABSTRACTION = [
    ("AKI", "Acute kidney injury (N17)"),
    ("체액 과부하", "Fluid overload (E87.7)"),
    ("고칼륨", "Hyperkalemia (E87.5)"),
    ("산증", "Metabolic acidosis (E87.2)"),
    ("요독", "Uremic symptoms (R39.2)"),
]


def _abstract(statement: str) -> str:
    """A 진술 → ICD-like 표준 문제명(매칭 없으면 진술 핵심부 사용)."""
    for kw, label in _ICD_ABSTRACTION:
        if kw in statement:
            return label
    # 매핑 실패 시: "라벨: 내용" 이면 내용을, 아니면 '—' 앞부분을 쓴다(새 조건 추측 아님).
    if ":" in statement:
        core = statement.split(":", 1)[1]
    else:
        core = statement.split("—")[0]
    return core.strip()[:80]


class ProblemListService:
    def derive(self, soap: dict) -> list[dict]:
        a = soap.get("A", {}) or {}
        if a.get("assessment") in (None, "", NOT_SUFFICIENT):
            return []
        statements = a.get("statements", []) or []
        problems: list[dict] = []
        seen: set[str] = set()
        for st in statements:
            stmt = st.get("statement", "")
            evidence = st.get("evidence", []) or []
            if not evidence:  # 근거 없는 진술은 문제로 승격 금지
                continue
            problem = _abstract(stmt)
            if problem in seen:
                continue
            seen.add(problem)
            problems.append(
                {
                    "problem": problem,
                    "source": "A",
                    "confidence": _confidence(evidence),
                    "evidence": evidence,
                }
            )
        return problems


def _confidence(evidence: list) -> float:
    """근거 개수 기반 신뢰도(0.5~0.95). 근거가 많을수록 높음(상한 고정)."""
    n = len(evidence)
    return round(min(0.95, 0.5 + 0.15 * n), 2)
