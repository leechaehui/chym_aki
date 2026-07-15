"""프로젝트 전체 공통 상수 — 단일 소스(Single Source of Truth).

AKI 스테이지 라벨, 위험 등급, Lab 참조범위, 모델 피처 컬럼 등
여러 서비스/리포지토리에서 중복 정의되던 상수를 한곳에 모은다.
변경 시 이 파일만 수정하면 전체에 반영된다.
"""
from __future__ import annotations


# ── AKI 스테이지 라벨 ────────────────────────────────────────────
#  모델 예측(pred) 또는 KDIGO 스테이지 → 사용자 표시 문구.
#  사용처: icu_monitor_service, real_patient_service, icu_patient_detail_service
STAGE_LABELS: dict[int, str] = {
    0: "Non-AKI",
    1: "AKI Stage 1",
    2: "AKI Stage 2-3",
}

# ── 위험 등급 (tier/label) ───────────────────────────────────────
#  사용처: icu_monitor_service._to_out, icu_patient_detail_service.summary
RISK_TIERS: dict[int, str] = {0: "low", 1: "moderate", 2: "high"}
RISK_LABELS: dict[int, str] = {0: "안정", 1: "중등도", 2: "고위험"}

# ── Lab 참조범위 (의료 표준) ─────────────────────────────────────
#  사용처: real_patient_service._lab() 호출 시 하드코딩 제거용.
#  key → {label, unit, ref_low, ref_high}
LAB_REFERENCES: dict[str, dict] = {
    "cr":   {"label": "Creatinine", "unit": "mg/dL",  "ref_low": 0.7,  "ref_high": 1.3},
    "egfr": {"label": "eGFR",      "unit": "mL/min", "ref_low": 60,   "ref_high": None},
    "bun":  {"label": "BUN",       "unit": "mg/dL",  "ref_low": 8,    "ref_high": 20},
    "na":   {"label": "Na",        "unit": "mmol/L", "ref_low": 135,  "ref_high": 145},
    "k":    {"label": "K",         "unit": "mmol/L", "ref_low": 3.5,  "ref_high": 5.1},
    "hco3": {"label": "HCO3",     "unit": "mmol/L", "ref_low": 22,   "ref_high": 29},
}

# ── 모델 입력 35 피처 (transform_info / 모델 feature_cols 와 동일 순서) ──
#  사용처: mimic_repository.FEATURE_COLS, icu_monitor_service
FEATURE_COLS: list[str] = [
    "map_mean", "map_min", "map_below65_hours", "sbp_min", "sbp_mean", "shock_index_mean",
    "hr_max", "hr_mean", "rr_max", "rr_mean", "temp_max", "temp_mean",
    "urine_output_sum", "urine_output_6h", "oliguria_flag",
    "creatinine_min", "creatinine_max", "creatinine_delta", "bun_max", "bun_cr_ratio",
    "lactate_max", "lactate_mean", "vasopressor_flag", "vasopressor_hours", "norepi_dose_max",
    "potassium_max", "potassium_mean", "bicarbonate_min", "bicarbonate_mean",
    "sodium_min", "sodium_max", "hemoglobin_min", "hemoglobin_mean", "spo2_min", "spo2_mean",
]


def is_female(gender: str | None) -> bool:
    """성별 문자열 → 여성 여부 판정 (공통 유틸).

    사용처: real_patient_service, icu_patient_detail_service, clinical_calculations
    """
    return bool(gender) and gender.strip().upper().startswith("F")
