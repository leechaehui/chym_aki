"""SOAP 서비스 — STRICT CONTROLLED GENERATION (작업지시서 4).

핵심 안전 규칙
- S / O : extraction only. 전사·환자 기록의 사실만 옮긴다(생성 금지).
- A / P : controlled inference only. **반드시 transcript/객관 근거(evidence)에 기반**.
          근거 없는 진술은 생성하지 않는다(hallucination 차단, RULE 1).
          근거 부족 시 "Not sufficient information"(RULE 2).
          new disease inference 금지·differential 제한(RULE 3) — 고정 임상 규칙셋에서만 생성.

출력(4.3):
  {"S": str, "O": str,
   "A": {"assessment": str, "evidence": [...], "statements": [{statement, evidence}]},
   "P": {"plan": str, "evidence": [...], "statements": [{statement, evidence}]}}

순수 로직(LLM 자유생성 없음) — 결정론적·재현가능.
"""
from __future__ import annotations

NOT_SUFFICIENT = "Not sufficient information"


def _lab(labs: list, key: str):
    for lab in labs:
        if getattr(lab, "key", None) == key:
            return getattr(lab, "value", None)
    return None


def _dedup(items: list[str]) -> list[str]:
    seen, out = set(), []
    for it in items:
        if it and it not in seen:
            seen.add(it)
            out.append(it)
    return out


class SoapService:
    """근거 기반 SOAP 구조화기. DB 의존 없음(입력을 받아 순수 변환)."""

    def generate(self, patient, transcript: str, symptoms: dict, aki_result: dict) -> dict:
        labs = list(getattr(patient, "labs", []) or [])
        cr = _lab(labs, "cr")
        egfr = _lab(labs, "egfr")
        k = _lab(labs, "k")
        bun = _lab(labs, "bun")
        hco3 = _lab(labs, "hco3")
        urine = list(getattr(patient, "urine_output", []) or [])
        uo = urine[-1].value if urine else None

        sym = symptoms or {}
        found = sym.get("symptoms", []) or []
        evidence_map = sym.get("evidence", {}) or {}

        S = self._subjective(transcript, found, evidence_map)
        O = self._objective(cr, egfr, k, bun, uo, aki_result)
        A = self._assessment(patient, aki_result, cr, found, evidence_map)
        P = self._plan(aki_result, k, hco3, uo, found, evidence_map)
        return {"S": S, "O": O, "A": A, "P": P}

    # ---------------- S: 주관적(추출만) ----------------
    def _subjective(self, transcript: str, found: list, evidence_map: dict) -> str:
        t = (transcript or "").strip()
        if t:
            if found:
                return f"{t} (추출 증상: {', '.join(found)})"
            return t
        if found:
            # 전사 없이 증상만 있는 경우 — 추출된 증상 라벨만 나열(생성 아님).
            return f"환자 호소(추출): {', '.join(found)}"
        return NOT_SUFFICIENT

    # ---------------- O: 객관적(추출만) ----------------
    def _objective(self, cr, egfr, k, bun, uo, aki_result: dict) -> str:
        parts: list[str] = []
        if cr is not None:
            parts.append(f"Creatinine {cr} mg/dL")
        if egfr is not None:
            parts.append(f"eGFR {egfr} mL/min")
        if k is not None:
            parts.append(f"K {k} mmol/L")
        if bun is not None:
            parts.append(f"BUN {bun} mg/dL")
        if uo is not None:
            parts.append(f"시간당 소변량 {uo} mL/kg/h")
        if aki_result.get("risk_score") is not None:
            parts.append(f"AKI 위험점수 {aki_result['risk_score']}점")
        return ". ".join(parts) if parts else NOT_SUFFICIENT

    # ---------------- A: 평가(근거 필수, controlled) ----------------
    def _assessment(self, patient, aki_result, cr, found, evidence_map) -> dict:
        statements: list[dict] = []

        # (1) 기저 진단 재기술 — 새 질병 추론 아님(환자 기록 근거).
        diagnosis = getattr(patient, "diagnosis", None)
        if diagnosis:
            statements.append(
                {"statement": f"기저 진단: {diagnosis}",
                 "evidence": [f"환자 기록 진단: {diagnosis}"]}
            )

        # (2) AKI 단계 — 모델/룰 근거(rationale)+객관 수치가 있을 때만.
        if aki_result.get("stage_num", 0) >= 1:
            ev = list(aki_result.get("rationale", [])[:3])
            if cr is not None:
                ev.append(f"Creatinine {cr} mg/dL")
            ev = _dedup(ev)
            if ev:  # evidence 없으면 생성 금지
                statements.append(
                    {"statement":
                         f"AKI 평가: {aki_result.get('stage')} ({aki_result.get('risk_label')})",
                     "evidence": ev}
                )

        # (3) 증상 기반 증후군 우려 — 고정 규칙셋(자유 진단 금지), transcript 근거 필수.
        if ("부종" in found or "핍뇨" in found or "무뇨" in found):
            ev = _dedup([evidence_map.get(s) for s in ("부종", "핍뇨", "무뇨") if s in found])
            if ev:
                statements.append(
                    {"statement": "체액 과부하/신기능 저하 가능성 — 임상 상관 필요",
                     "evidence": ev}
                )
        if "오심/구토" in found and aki_result.get("stage_num", 0) >= 1:
            ev = _dedup([evidence_map.get("오심/구토")])
            if ev:
                statements.append(
                    {"statement": "요독 증상 가능성(오심/구토) — AKI 맥락에서 평가",
                     "evidence": ev}
                )

        return self._pack(statements, key="assessment")

    # ---------------- P: 계획(근거 필수, controlled) ----------------
    def _plan(self, aki_result, k, hco3, uo, found, evidence_map) -> dict:
        statements: list[dict] = []
        stage = aki_result.get("stage_num", 0)
        aki_ev = _dedup(list(aki_result.get("rationale", [])[:2]))

        if stage >= 3 and aki_ev:
            statements.append(
                {"statement": "신독성 약물 점검/중단, 전해질 응급 교정, 신대체요법 적응증 확인, 신장내과 긴급 협진",
                 "evidence": aki_ev}
            )
        elif stage >= 1 and aki_ev:
            statements.append(
                {"statement": "수액·전해질 교정, 신독성 약물 점검, 시간당 소변량 모니터링, 추적 신기능 검사",
                 "evidence": aki_ev}
            )

        if k is not None and k >= 6.0:
            statements.append(
                {"statement": "고칼륨혈증 교정(칼슘/인슐린-포도당/케이엑살레이트) 및 ECG 모니터",
                 "evidence": [f"K {k} mmol/L"]}
            )
        if hco3 is not None and hco3 < 18:
            statements.append(
                {"statement": "대사성 산증 교정 고려(원인 평가 동반)",
                 "evidence": [f"HCO3 {hco3} mmol/L"]}
            )
        if uo is not None and uo < 0.5:
            ev = _dedup([f"시간당 소변량 {uo} mL/kg/h"]
                        + [evidence_map.get(s) for s in ("핍뇨", "무뇨") if s in found])
            statements.append(
                {"statement": "엄격한 수분 출납(I/O) 및 시간당 소변량 추적", "evidence": ev}
            )

        return self._pack(statements, key="plan")

    # ---------------- 공통 패킹 ----------------
    def _pack(self, statements: list[dict], *, key: str) -> dict:
        statements = [s for s in statements if s.get("evidence")]  # 근거 없는 진술 제거
        if not statements:
            return {key: NOT_SUFFICIENT, "evidence": [], "statements": []}
        text = " · ".join(s["statement"] for s in statements)
        evidence = _dedup([e for s in statements for e in s["evidence"]])
        return {key: text, "evidence": evidence, "statements": statements}
