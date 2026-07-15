"""ICU 환자 상세 뷰 조립 서비스 — Quick View / Trend / SHAP / Validation / Risk History.

ICU AKI 모니터의 한 환자를 깊게 들여다보는 화면(검색·퀵뷰·검증 리포트)에 필요한 데이터를
한 곳에서 조립한다. 데이터 출처가 여럿(코호트 예측 캐시 · 원시 시계열 DB · 모델 설명 · 보정)이라
각 출처는 전용 모듈에 위임하고(SRP), 이 서비스는 '조립과 표현 포맷'만 책임진다(Facade).

의존(모두 단방향):
  - icu_monitor_service     : 코호트 예측 캐시 + 모델/스케일 피처 접근
  - mimic_repository        : 원시 Cr/소변량 시계열 + 코호트 행
  - clinical_calculations   : eGFR · KDIGO 단계 · 위험점수 proxy
  - model_explainer         : 환자별 SHAP 기여도
  - calibration_report      : 모델 신뢰도 곡선(ECE 대체)
"""
from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from repositories import mimic_repository as mimic
from services import clinical_calculations as clinical
from services.calibration_report import get_calibration_report
from services.icu_monitor_service import _predictions, prediction_for_stay
from services.model_explainer import TwoStageExplainer
from services.rrt_trigger import RrtSignals, assess_rrt_trigger
from queries.constants import STAGE_LABELS, RISK_TIERS, is_female
from queries.patient_queries import sql_raw_bun_for_stay
from core.config import settings
from core.database import engine


def _is_female(gender: str | None) -> bool:
    return is_female(gender)


def _format_elapsed(hours: float | None) -> str | None:
    """입실 후 경과시간 → 사람이 읽는 문구. 24h 미만은 시:분, 이상은 일 단위."""
    if hours is None:
        return None
    if hours < 24:
        whole_hours = int(hours)
        minutes = int(round((hours - whole_hours) * 60))
        return f"입실 후 {whole_hours}시간 {minutes}분 경과"
    return f"입실 후 {int(hours // 24)}일 경과"


class IcuPatientDetailService:
    """ICU 한 환자의 상세 뷰를 조립한다(읽기 전용·상태 없음)."""

    def __init__(self) -> None:
        self._explainer = TwoStageExplainer()

    # ---- 검색 -----------------------------------------------------------------
    def search(self, query: str, limit: int = 10) -> list[dict]:
        """환자 이름 · stay_id · subject_id · 병동(careunit)으로 환자를 찾는다(위험순).

        이름은 합성 표시명(display_name, 프론트와 동일)으로 매칭 → 화면에 보이는 이름을
        그대로 입력해 조회할 수 있다. 숫자는 stay/subject ID, 그 외 텍스트는 이름·병동.
        """
        df = _predictions()
        query = (query or "").strip()
        if not query:
            return []
        as_text = query.lower()
        numeric = query if query.isdigit() else None
        # 텍스트: 합성 이름 또는 병동명 부분일치(정규식 아님 — 괄호/한글 안전).
        mask = (
            df["display_name"].str.contains(query, na=False, regex=False)
            | df["careunit"].str.lower().str.contains(as_text, na=False, regex=False)
        )
        if numeric is not None:
            mask = mask | df["stay_id"].astype(str).str.contains(numeric) \
                        | df["subject_id"].astype(str).str.contains(numeric)
        hits = df[mask].sort_values("risk_score", ascending=False).head(limit)
        return [self._search_row(r) for _, r in hits.iterrows()]

    @staticmethod
    def _search_row(r: pd.Series) -> dict:
        return {
            "stay_id": int(r["stay_id"]),
            "subject_id": int(r["subject_id"]),
            "age": int(r["age"]) if pd.notna(r["age"]) else None,
            "gender": r["gender"],
            "careunit": r["careunit"],
            "risk_score": int(r["risk_score"]),
            "risk": RISK_TIERS[int(r["pred"])],
            "stage": STAGE_LABELS[int(r["pred"])],
        }

    # ---- Quick View 요약 ------------------------------------------------------
    def summary(self, stay_id: int) -> dict | None:
        """퀵뷰용 환자 요약 — 인구학·입원·AKI 상태·신장 기능(가용 값만, 결측은 None)."""
        prediction = prediction_for_stay(stay_id)
        cohort = mimic.cohort_record(stay_id)
        if prediction is None or cohort is None:
            return None

        creatinine_series = mimic.creatinine_trend(stay_id)
        latest_creatinine = creatinine_series[-1]["creatinine"] if creatinine_series else None
        baseline = mimic.stay_baseline(stay_id)
        latest_egfr = clinical.estimate_egfr_ckd_epi_2021(
            latest_creatinine, cohort.get("age"), _is_female(cohort.get("gender"))
        ) if latest_creatinine is not None else None
        latest_urine_rate = mimic.stay_latest_urine_rate(stay_id)
        icu_los_hours = (
            float(cohort["icu_los_hours"]) if cohort.get("icu_los_hours") is not None else None
        )
        pred = int(prediction["pred"])

        # BUN: final_features_48h 에서 raw bun_max 조회 (기존에 None 하드코딩이던 것을 수정)
        bun_raw = None
        try:
            with engine.connect() as c:
                bun_row = c.execute(
                    sql_raw_bun_for_stay(settings.app_schema), {"s": stay_id}
                ).first()
            if bun_row and bun_row[0] is not None:
                bun_raw = round(float(bun_row[0]), 1)
        except Exception:
            pass  # 테이블 미존재 시 None 유지

        return {
            "stay_id": stay_id,
            "subject_id": int(prediction["subject_id"]),
            # 기본 정보 (체중/키/BMI 는 MIMIC 코호트에 없어 None — 화면에서 '비측정' 처리)
            "age": int(prediction["age"]) if pd.notna(prediction["age"]) else None,
            "gender": prediction["gender"],
            "weight_kg": None,
            "height_cm": None,
            "bmi": None,
            # 입원 정보
            "careunit": cohort["first_careunit"],
            "bed": None,  # MIMIC 비식별: 병실/베드 정보 없음
            "admission_type": cohort.get("admission_type"),
            "icu_los_hours": round(icu_los_hours, 1) if icu_los_hours is not None else None,
            "elapsed_text": _format_elapsed(icu_los_hours),
            "predict_at": prediction.get("predict_at") if pd.notna(prediction.get("predict_at")) else None,
            # AKI 상태
            "stage": STAGE_LABELS[pred],
            "stage_num": pred,
            "risk_score": int(prediction["risk_score"]),
            "risk": RISK_TIERS[pred],
            "p_stage2_plus": round(float(prediction["p_stage2_plus"]), 4),
            # 신장 기능 — BUN 은 final_features_48h.bun_max 에서 Raw 조회
            "creatinine": round(latest_creatinine, 2) if latest_creatinine is not None else None,
            "egfr": latest_egfr,
            "bun": bun_raw,
            "baseline_creatinine": round(baseline["baseline_cr"], 2) if baseline else None,
            "urine_rate_ml_kg_h": round(latest_urine_rate, 2) if latest_urine_rate is not None else None,
        }

    # ---- 추세 그래프 ----------------------------------------------------------
    def trends(self, stay_id: int) -> dict:
        """환자별 Cr · eGFR · 소변량 시계열(입실 후 경과시간 x축)."""
        cohort = mimic.cohort_record(stay_id)
        age = cohort.get("age") if cohort else None
        female = _is_female(cohort.get("gender")) if cohort else False

        creatinine_series = mimic.creatinine_trend(stay_id)
        creatinine_points = [
            {"hours": round(p["hours_from_admit"], 1), "value": round(p["creatinine"], 2)}
            for p in creatinine_series
        ]
        egfr_points = []
        for p in creatinine_series:
            egfr = clinical.estimate_egfr_ckd_epi_2021(p["creatinine"], age, female)
            if egfr is not None:
                egfr_points.append({"hours": round(p["hours_from_admit"], 1), "value": egfr})
        urine_points = [
            {"hours": round(p["hours_from_admit"], 1), "value": round(p["urine_rate_ml_kg_h"], 2)}
            for p in mimic.urine_rate_trend(stay_id)
        ]
        return {
            "stay_id": stay_id,
            "creatinine": creatinine_points,
            "egfr": egfr_points,
            "urine_output": urine_points,
        }

    # ---- SHAP 설명 ------------------------------------------------------------
    def shap(self, stay_id: int, limit: int = 8) -> dict:
        """환자별 상위 기여 피처(SHAP-style). 모델이 본 스케일 피처 기준 정확 계산."""
        features = [asdict(c) for c in self._explainer.top_features(stay_id, limit=limit)]
        return {"stay_id": stay_id, "features": features, "shap_features": features}

    # ---- 검증 리포트 ----------------------------------------------------------
    def validation(self, stay_id: int) -> dict | None:
        """환자별 예측 vs 실제 + 모델 신뢰도 곡선(ECE 대체)."""
        prediction = prediction_for_stay(stay_id)
        if prediction is None:
            return None
        calibration = get_calibration_report()
        pred = int(prediction["pred"])
        return {
            "stay_id": stay_id,
            "predicted_stage": STAGE_LABELS[pred],
            "predicted_stage_num": pred,
            "p_aki": round(float(prediction["p_aki"]), 4),
            "p_stage2_plus": round(float(prediction["p_stage2_plus"]), 4),
            "risk_score": int(prediction["risk_score"]),
            "actual_label": int(prediction["actual_label"]),
            "actual_stage": int(prediction["actual_stage"]),
            "correct": (pred >= 1) == (int(prediction["actual_label"]) == 1),
            "calibration": {
                "bins": [asdict(b) for b in calibration.bins],
                "ece": calibration.expected_calibration_error,
                "n": calibration.n,
            },
        }

    # ---- RRT 트리거(임상 의사결정 보조) --------------------------------------
    def rrt_assessment(self, stay_id: int) -> dict | None:
        """KDIGO+ICU 규칙 기반 RRT 트리거 단계(예측 아님). 가용 신호(Cr/eGFR/소변량)만 사용."""
        cohort = mimic.cohort_record(stay_id)
        creatinine_series = mimic.creatinine_trend(stay_id)
        if cohort is None or not creatinine_series:
            return None
        signals = self._build_rrt_signals(stay_id, cohort, creatinine_series)
        assessment = assess_rrt_trigger(signals)
        return {
            "stay_id": stay_id,
            "rrt_level": assessment.rrt_level,
            "label": assessment.label,
            "confidence": assessment.confidence,
            "key_drivers": assessment.key_drivers,
            "trend_signal": assessment.trend_signal,
            "clinical_action": assessment.clinical_action,
            "warning_note": assessment.warning_note,
            "unavailable_criteria": assessment.unavailable_criteria,
        }

    def _build_rrt_signals(self, stay_id, cohort, creatinine_series) -> RrtSignals:
        """원시 Cr/소변량 시계열 → RRT 트리거 입력 신호(미측정은 None)."""
        age = cohort.get("age")
        female = _is_female(cohort.get("gender"))
        baseline = mimic.stay_baseline(stay_id)
        baseline_cr = baseline["baseline_cr"] if baseline else creatinine_series[0]["creatinine"]

        current_cr = creatinine_series[-1]["creatinine"]
        last_hours = creatinine_series[-1]["hours_from_admit"]
        # 24h 전 시점에 가장 가까운 Cr 과의 차이.
        prior_24h = min(
            creatinine_series,
            key=lambda p: abs(p["hours_from_admit"] - (last_hours - 24)),
        )["creatinine"]
        delta_24h = round(current_cr - prior_24h, 2)
        ratio = round(current_cr / baseline_cr, 2) if baseline_cr else None

        egfr_now = clinical.estimate_egfr_ckd_epi_2021(current_cr, age, female)
        egfr_first = clinical.estimate_egfr_ckd_epi_2021(creatinine_series[0]["creatinine"], age, female)
        egfr_declining = bool(egfr_now and egfr_first and egfr_now < egfr_first * 0.85)

        urine = mimic.urine_rate_trend(stay_id)
        urine_min_6h = urine_min_12h = anuria_hours = None
        if urine:
            end = urine[-1]["hours_from_admit"]
            last_6h = [u["urine_rate_ml_kg_h"] for u in urine if u["hours_from_admit"] >= end - 6]
            last_12h = [u["urine_rate_ml_kg_h"] for u in urine if u["hours_from_admit"] >= end - 12]
            urine_min_6h = round(min(last_6h), 2) if last_6h else None
            urine_min_12h = round(min(last_12h), 2) if last_12h else None
            # 끝에서 연속된 무뇨(rate<0.1) 시간 길이.
            anuria = [u for u in urine if u["urine_rate_ml_kg_h"] < 0.1]
            if anuria and urine[-1]["urine_rate_ml_kg_h"] < 0.1:
                anuria_hours = round(end - anuria[0]["hours_from_admit"], 1)

        return RrtSignals(
            creatinine_current=round(current_cr, 2),
            creatinine_baseline=round(baseline_cr, 2) if baseline_cr else None,
            creatinine_delta_24h=delta_24h,
            creatinine_ratio_to_baseline=ratio,
            egfr_current=egfr_now,
            egfr_declining=egfr_declining,
            urine_min_6h_ml_kg_h=urine_min_6h,
            urine_min_12h_ml_kg_h=urine_min_12h,
            anuria_hours=anuria_hours,
        )

    # ---- 위험도 시계열(정직성: Cr 파생 + 모델 1점) ----------------------------
    def risk_history(self, stay_id: int) -> dict:
        """위험도 추세 — Cr/기저치 기반 KDIGO 추정선 + 실제 모델 예측 1점(명확 구분).

        모델은 입실 후 48h 단일 윈도우로 1회만 예측하므로 시점별 모델 출력은 존재하지 않는다.
        따라서 추세선은 Cr 기반 KDIGO '추정'(source='cr_kdigo')으로 그리고, 진짜 모델 출력은
        단일 점(source='model')으로 겹쳐 표시한다 — 둘은 라벨로 분명히 구분한다.
        """
        baseline = mimic.stay_baseline(stay_id)
        baseline_cr = baseline["baseline_cr"] if baseline else None
        creatinine_series = mimic.creatinine_trend(stay_id)

        trajectory = []
        if baseline_cr and baseline_cr > 0:
            for p in creatinine_series:
                ratio = p["creatinine"] / baseline_cr
                stage = clinical.kdigo_stage_from_creatinine_ratio(ratio)
                trajectory.append({
                    "hours": round(p["hours_from_admit"], 1),
                    "risk_score": clinical.risk_score_from_kdigo_stage(stage),
                    "kdigo_stage": stage,
                    "source": "cr_kdigo",
                })

        prediction = prediction_for_stay(stay_id)
        model_point = None
        if prediction is not None:
            # 모델 입력 윈도우(입실+48h)에 해당하는 실제 모델 위험점수 1점.
            last_hours = creatinine_series[-1]["hours_from_admit"] if creatinine_series else 48.0
            model_point = {
                "hours": round(min(48.0, last_hours), 1),
                "risk_score": int(prediction["risk_score"]),
                "source": "model",
            }
        return {
            "stay_id": stay_id,
            "baseline_creatinine": round(baseline_cr, 2) if baseline_cr else None,
            "trajectory": trajectory,
            "model_point": model_point,
            "disclaimer": "추세선은 Cr/기저치 기반 KDIGO 추정이며, 실제 모델 출력은 단일 점(48h)뿐입니다.",
        }
