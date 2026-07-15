"""SOAP 진료 초안 생성기.

책임: 전사(transcript) + NLP 증상 + 환자 검사값 + AKI 평가를 결합해
SOAP(주관/객관/평가/계획) 4섹션 초안을 구성한다.
순수 함수형 — DB/세션 의존 없음(서비스가 입력을 모아 전달).
"""
from __future__ import annotations


def _lab_value(labs: list, key: str):
    for lab in labs:
        if getattr(lab, "key", None) == key:
            return getattr(lab, "value", None)
    return None


def generate_soap(
    patient,
    transcript: str,
    symptoms: dict,
    aki_result: dict,
) -> dict:
    """SOAP dict 생성.

    patient : ORM Patient(labs 포함) 또는 동등 객체.
    transcript : 환자 대화/음성 전사.
    symptoms : NLP 추출 결과.
    aki_result : AKI 예측기 출력.
    """
    labs = list(getattr(patient, "labs", []) or [])
    cr = _lab_value(labs, "cr")
    egfr = _lab_value(labs, "egfr")
    k = _lab_value(labs, "k")

    sex_label = "남성" if getattr(patient, "sex", "M") == "M" else "여성"
    symptom_list = symptoms.get("symptoms", []) if symptoms else []
    symptom_text = ", ".join(symptom_list) if symptom_list else "특이 호소 없음"

    # S — 주관적: 전사 우선, 없으면 인적정보+추출 증상으로 요약.
    subjective = (transcript or "").strip() or (
        f"{getattr(patient, 'age', '-')}세 {sex_label}, "
        f"{getattr(patient, 'diagnosis', '')} 경과 관찰 중. 주요 호소: {symptom_text}."
    )
    if transcript and symptom_list:
        subjective += f" (추출 증상: {symptom_text})"

    # O — 객관적: 핵심 검사값.
    objective = (
        f"Creatinine {cr if cr is not None else '-'} mg/dL, "
        f"eGFR {egfr if egfr is not None else '-'} mL/min, "
        f"K {k if k is not None else '-'} mmol/L. "
        f"AKI 위험점수 {aki_result.get('risk_score')}점."
    )

    # A — 평가: AKI 단계 + 근거.
    rationale = "; ".join(aki_result.get("rationale", [])[:3])
    assessment = (
        f"{getattr(patient, 'diagnosis', '')}. "
        f"AKI 평가: {aki_result.get('stage')} ({aki_result.get('risk_label')}). "
        f"근거: {rationale}."
    )

    # P — 계획: 위험도 기반 권고.
    if aki_result.get("stage_num", 0) >= 3:
        plan = (
            "신독성 약물 즉시 점검·중단, 전해질(특히 K) 응급 교정, "
            "수액 반응 재평가 및 신대체요법(HD/CRRT) 적응증 확인, 신장내과 긴급 협진."
        )
    elif aki_result.get("stage_num", 0) >= 1:
        plan = (
            "수액·전해질 교정, 신독성 약물 점검, 시간당 소변량 모니터링, "
            "추적 신기능 검사 및 필요 시 협진 의뢰."
        )
    else:
        plan = "현 치료 유지, 신기능 추적 관찰, 위험인자 모니터링."

    return {
        "subjective": subjective,
        "objective": objective,
        "assessment": assessment,
        "plan": plan,
    }
