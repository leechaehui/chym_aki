"""순수 임상 계산식 모음 (상태 없음·부작용 없음).

신장 기능/AKI 평가에 쓰는 표준 공식을 한 곳에 모은다. 모든 함수는 입력만으로 결과가
결정되는 순수 함수라 단위 테스트가 쉽고, 호출부(서비스)는 '무엇을 계산하는지'를
함수명만으로 알 수 있다.

- eGFR: CKD-EPI 2021 (인종 보정 제거 버전, 현행 KDIGO 권고).
- AKI 단계: KDIGO 혈청 크레아티닌 기준(기저치 대비 배수).
- 위험 점수: KDIGO 단계 → 0~100 점수(시계열 추세 표시용 proxy).
"""
from __future__ import annotations

# CKD-EPI 2021 성별 상수(κ=정규화 분모, α=저(低)크레아티닌 지수).
_CKD_EPI_KAPPA = {"female": 0.7, "male": 0.9}
_CKD_EPI_ALPHA = {"female": -0.241, "male": -0.302}
_CKD_EPI_FEMALE_MULTIPLIER = 1.012
_CKD_EPI_AGE_BASE = 0.9938


def estimate_egfr_ckd_epi_2021(
    creatinine_mg_dl: float, age_years: int, is_female: bool
) -> float | None:
    """CKD-EPI 2021(인종 비보정) 추정 사구체여과율(mL/min/1.73m²).

    유효하지 않은 입력(크레아티닌 ≤ 0 등)이면 None.
    """
    if creatinine_mg_dl is None or creatinine_mg_dl <= 0 or age_years is None:
        return None
    sex = "female" if is_female else "male"
    kappa = _CKD_EPI_KAPPA[sex]
    alpha = _CKD_EPI_ALPHA[sex]
    cr_over_kappa = creatinine_mg_dl / kappa
    low_term = min(cr_over_kappa, 1.0) ** alpha
    high_term = max(cr_over_kappa, 1.0) ** -1.200
    egfr = 142.0 * low_term * high_term * (_CKD_EPI_AGE_BASE ** age_years)
    if is_female:
        egfr *= _CKD_EPI_FEMALE_MULTIPLIER
    return round(egfr, 1)


def kdigo_stage_from_creatinine_ratio(current_over_baseline: float) -> int:
    """기저치 대비 크레아티닌 배수 → KDIGO AKI 단계(0~3).

    Stage1 1.5–1.9배 · Stage2 2.0–2.9배 · Stage3 ≥3.0배. (소변량 기준은 별도)
    """
    if current_over_baseline is None or current_over_baseline < 1.5:
        return 0
    if current_over_baseline < 2.0:
        return 1
    if current_over_baseline < 3.0:
        return 2
    return 3


# KDIGO 단계별 대표 위험 점수(시계열 추세 표시용 proxy — 실제 모델 출력 아님).
_RISK_SCORE_BY_KDIGO_STAGE = {0: 12, 1: 43, 2: 72, 3: 90}


def risk_score_from_kdigo_stage(stage: int) -> int:
    """KDIGO 단계 → 0~100 위험 점수 proxy(Cr 추세선 표시용)."""
    return _RISK_SCORE_BY_KDIGO_STAGE.get(stage, 12)
