"""실제 ICU stay 기반 AI 진료 초안(SOAP) — 데모 환자 mock 없이 실데이터로 동일 파이프라인.

음성/텍스트 전사 → SOAP(근거기반) → Problem List → CDSS Risk → 검증. **영속화 없음**
(코호트는 읽기 전용이고 ICU stay 는 앱 patient/timeline 테이블에 없다).

기존 순수 서비스(SoapService/ProblemListService/CdssRiskService/validator/nlp)는 patient 객체의
최소 인터페이스(labs[].key/value, trend[].creatinine, urine_output[].value, age, diagnosis)만
요구하므로, 실제 stay 데이터로 만든 가벼운 어댑터로 그대로 재사용한다(코드 중복 없음).

AKI 결과는 **실제 학습 모델 예측**(prediction_for_stay)을 쓰고, 근거(rationale)는 실측 Cr/소변량
KDIGO 기준으로 생성 → SOAP A/P 의 evidence 가 정확한 KDIGO 기준을 가리킨다.
"""
from __future__ import annotations

from nlp.factory import create_nlp_strategy
from repositories import mimic_repository as mimic
from services import clinical_calculations as clinical
from services.cdss_risk_service import CdssRiskService
from services.icu_monitor_service import prediction_for_stay
from services.problem_list_service import ProblemListService
from services.soap_service import SoapService
from validator.soap_validator import run_all as run_soap_validation

_STAGE_LABELS = {0: "Non-AKI", 1: "AKI Stage 1", 2: "AKI Stage 2-3"}
_RISK_LABELS = {0: "안정", 1: "중등도", 2: "고위험"}


def _is_female(gender: str | None) -> bool:
    return bool(gender) and gender.strip().upper().startswith("F")


class _Lab:
    __slots__ = ("key", "value")

    def __init__(self, key: str, value) -> None:
        self.key, self.value = key, value


class _TrendPoint:
    __slots__ = ("creatinine",)

    def __init__(self, creatinine) -> None:
        self.creatinine = creatinine


class _UrinePoint:
    __slots__ = ("value",)

    def __init__(self, value) -> None:
        self.value = value


class _PatientAdapter:
    """soap/cdss/feature 서비스가 기대하는 최소 인터페이스만 노출하는 실-stay 어댑터."""

    def __init__(self, *, labs, trend, urine_output, age, diagnosis) -> None:
        self.labs = labs
        self.trend = trend
        self.urine_output = urine_output
        self.age = age
        self.diagnosis = diagnosis


def _kdigo_rationale(baseline, cr_max, uo) -> list[str]:
    """실측 Cr 배수/증가량·소변량으로 KDIGO 근거 문자열 생성(룰 예측기와 동일 표현)."""
    out: list[str] = []
    if cr_max and baseline and baseline > 0:
        ratio = cr_max / baseline
        if ratio >= 3.0:
            out.append(f"Cr {ratio:.1f}배 상승 (KDIGO Stage 3 기준)")
        elif ratio >= 2.0:
            out.append(f"Cr {ratio:.1f}배 상승 (Stage 2)")
        elif ratio >= 1.5:
            out.append(f"Cr {ratio:.1f}배 상승 (Stage 1)")
        delta = cr_max - baseline
        if delta >= 0.3:
            out.append(f"48h Cr 증가 {delta:.1f} mg/dL (≥0.3)")
    if uo is not None:
        if uo < 0.3:
            out.append(f"시간당 소변량 {uo:.2f} mL/kg/h (<0.3, 무뇨)")
        elif uo < 0.5:
            out.append(f"시간당 소변량 {uo:.2f} mL/kg/h (<0.5, 핍뇨)")
    return out


class IcuDraftService:
    """실제 ICU stay 의 SOAP 진료 초안 생성(순수 계산·읽기 전용)."""

    def __init__(self) -> None:
        self.nlp = create_nlp_strategy()
        self.soap = SoapService()
        self.problems = ProblemListService()
        self.cdss = CdssRiskService()

    def draft_for_stay(self, stay_id: int, transcript: str) -> dict | None:
        pred = prediction_for_stay(stay_id)
        cohort = mimic.cohort_record(stay_id)
        if pred is None or cohort is None:
            return None

        cr_series = [p["creatinine"] for p in mimic.creatinine_trend(stay_id)]
        baseline_row = mimic.stay_baseline(stay_id)
        baseline = (
            baseline_row["baseline_cr"] if baseline_row else (cr_series[0] if cr_series else None)
        )
        cr_now = cr_series[-1] if cr_series else None
        cr_max = max(cr_series) if cr_series else cr_now
        uo = mimic.stay_latest_urine_rate(stay_id)
        age = cohort.get("age")
        egfr = (
            clinical.estimate_egfr_ckd_epi_2021(cr_now, age, _is_female(cohort.get("gender")))
            if cr_now is not None
            else None
        )

        # 어댑터 — trend[0]=baseline(기저), urine 마지막=현재값(서비스 규약).
        trend = ([_TrendPoint(baseline)] if baseline is not None else []) + [
            _TrendPoint(c) for c in cr_series
        ]
        labs: list[_Lab] = []
        if cr_now is not None:
            labs.append(_Lab("cr", round(cr_now, 2)))
        if egfr is not None:
            labs.append(_Lab("egfr", round(egfr, 1)))
        patient = _PatientAdapter(
            labs=labs,
            trend=trend,
            urine_output=[_UrinePoint(round(uo, 2))] if uo is not None else [],
            age=age,
            diagnosis=None,  # ICU 코호트엔 확정 진단명이 없어 기저진단 생성 안 함(정직).
        )

        pnum = int(pred["pred"])
        aki_result = {
            "stage": _STAGE_LABELS[pnum],
            "stage_num": pnum,
            "risk_label": _RISK_LABELS[pnum],
            "risk_score": int(pred["risk_score"]),
            "p_aki": round(float(pred["p_aki"]), 4),
            "p_stage1": round(float(pred["p_stage1"]), 4),
            "p_stage2_plus": round(float(pred["p_stage2_plus"]), 4),
            "rationale": _kdigo_rationale(baseline, cr_max, uo),
            "source": "model",
        }

        symptoms = self.nlp.extract_symptoms(transcript)
        soap = self.soap.generate(patient, transcript, symptoms, aki_result)
        problems = self.problems.derive(soap)
        risk = self.cdss.score(patient, soap, problems, aki_result)
        validation = run_soap_validation(soap, risk)
        return {
            "id": f"stay-{stay_id}",
            "patient_id": str(stay_id),
            "transcript": transcript,
            "symptoms": symptoms,
            "soap": soap,
            "problem_list": problems,
            "risk": risk,
            "validation": validation,
            "status": "draft",
            "created_at": None,
        }
