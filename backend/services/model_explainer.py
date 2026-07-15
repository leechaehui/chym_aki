"""환자별 피처 기여도(SHAP-style) 설명 — 2-Stage 모델용.

목적: "이 환자의 위험 예측에 어떤 피처가 얼마나 기여했는가"를 환자마다 다르게 계산한다.
정직성: 각 단계는 모델 종류에 맞는 '정확한' 기여도를 쓴다(근사·고정값 금지).
  - Stage1 LogisticRegression: 기여도 = 계수(coef) × 스케일 피처값  (로그오즈 기여, 정확)
  - Stage2 LGBM: LightGBM 네이티브 pred_contrib (TreeSHAP, 정확)

설계(GoF):
  - Strategy:  `PatientFeatureExplainer` 인터페이스 + 단계별 구현 2종.
  - Facade:    `TwoStageExplainer` 가 두 전략을 합쳐 상위 기여 피처를 돌려준다.
한 환자의 스케일 피처 1행만 있으면 동작하며, 모델은 캐시본을 재사용한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from services.icu_monitor_service import model_bundles, scaled_features_for_stay

# 모델 내부 피처명 → 임상의가 읽을 한글 라벨. 표/막대 차트 표시에 사용.
FEATURE_LABELS_KO: dict[str, str] = {
    "creatinine_max": "Cr 최고치(48h)",
    "creatinine_min": "Cr 최저치(48h)",
    "creatinine_delta": "Cr 변화량(48h)",
    "bun_max": "BUN 최고치",
    "bun_cr_ratio": "BUN/Cr 비",
    "urine_output_sum": "소변량 총합(48h)",
    "urine_output_6h": "최근 6h 소변량",
    "oliguria_flag": "핍뇨 발생",
    "map_mean": "평균 동맥압",
    "map_min": "최저 동맥압",
    "map_below65_hours": "MAP<65 지속시간",
    "sbp_min": "최저 수축기압",
    "sbp_mean": "평균 수축기압",
    "shock_index_mean": "평균 쇼크지수",
    "hr_max": "최고 심박수",
    "hr_mean": "평균 심박수",
    "rr_max": "최고 호흡수",
    "rr_mean": "평균 호흡수",
    "temp_max": "최고 체온",
    "temp_mean": "평균 체온",
    "spo2_min": "최저 SpO₂",
    "spo2_mean": "평균 SpO₂",
    "lactate_max": "최고 젖산",
    "lactate_mean": "평균 젖산",
    "vasopressor_flag": "혈압상승제 사용",
    "vasopressor_hours": "혈압상승제 시간",
    "norepi_dose_max": "노르에피 최대용량",
    "potassium_max": "최고 칼륨",
    "potassium_mean": "평균 칼륨",
    "bicarbonate_min": "최저 중탄산",
    "bicarbonate_mean": "평균 중탄산",
    "sodium_min": "최저 나트륨",
    "sodium_max": "최고 나트륨",
    "hemoglobin_min": "최저 헤모글로빈",
    "hemoglobin_mean": "평균 헤모글로빈",
}


def label_for(feature: str) -> str:
    """피처 내부명 → 한글 라벨(미정의 시 원본명)."""
    return FEATURE_LABELS_KO.get(feature, feature)


def _unit_for(feature: str) -> str:
    units = {
        "creatinine_max": "mg/dL",
        "creatinine_min": "mg/dL",
        "creatinine_delta": "mg/dL",
        "bun_max": "mg/dL",
        "bun_cr_ratio": "",
        "urine_output_sum": "mL",
        "urine_output_6h": "mL/kg/h",
        "oliguria_flag": "",
        "map_mean": "mmHg",
        "map_min": "mmHg",
        "map_below65_hours": "h",
        "sbp_min": "mmHg",
        "sbp_mean": "mmHg",
        "shock_index_mean": "",
        "hr_max": "/min",
        "hr_mean": "/min",
        "rr_max": "/min",
        "rr_mean": "/min",
        "temp_max": "°C",
        "temp_mean": "°C",
        "spo2_min": "%",
        "spo2_mean": "%",
        "lactate_max": "mmol/L",
        "lactate_mean": "mmol/L",
        "vasopressor_flag": "",
        "vasopressor_hours": "h",
        "norepi_dose_max": "mcg/kg/min",
        "potassium_max": "mEq/L",
        "potassium_mean": "mEq/L",
        "bicarbonate_min": "mEq/L",
        "bicarbonate_mean": "mEq/L",
        "sodium_min": "mEq/L",
        "sodium_max": "mEq/L",
        "hemoglobin_min": "g/dL",
        "hemoglobin_mean": "g/dL",
    }
    return units.get(feature, "")


@dataclass(frozen=True)
class FeatureContribution:
    """한 피처가 위험 예측에 기여한 정도(부호 = 위험 ↑/↓)."""

    feature: str
    label: str
    contribution: float   # >0 위험을 높임, <0 낮춤
    scaled_value: float    # 모델이 본 표준화 값(참고용)
    value: float | None = None
    unit: str = ""
    shap_value: float | None = None


class PatientFeatureExplainer(Protocol):
    """단계별 기여도 계산 전략."""

    def explain(self, scaled_features: dict[str, float]) -> list[FeatureContribution]:
        ...


class LinearModelExplainer:
    """Stage1 LogisticRegression — 기여도 = 계수 × 표준화 피처값(정확한 로그오즈 기여)."""

    def __init__(self, model, feature_cols: list[str]) -> None:
        self._coefficients = dict(zip(feature_cols, model.coef_[0]))
        self._feature_cols = feature_cols

    def explain(self, scaled_features: dict[str, float]) -> list[FeatureContribution]:
        contributions = []
        for feature in self._feature_cols:
            value = scaled_features.get(feature, 0.0)
            contributions.append(
                FeatureContribution(
                    feature=feature,
                    label=label_for(feature),
                    contribution=float(self._coefficients[feature] * value),
                    scaled_value=float(value),
                    value=float(value),
                    unit=_unit_for(feature),
                    shap_value=float(self._coefficients[feature] * value),
                )
            )
        return contributions


class TreeModelExplainer:
    """Stage2 LGBM — LightGBM 네이티브 pred_contrib(TreeSHAP). 마지막 열은 base value 라 제외."""

    def __init__(self, model, feature_cols: list[str]) -> None:
        self._booster = model.booster_
        self._feature_cols = feature_cols

    def explain(self, scaled_features: dict[str, float]) -> list[FeatureContribution]:
        row = np.array([[scaled_features.get(f, 0.0) for f in self._feature_cols]], dtype=float)
        shap_values = self._booster.predict(row, pred_contrib=True)[0]  # length = n_features + 1
        return [
            FeatureContribution(
                feature=feature,
                label=label_for(feature),
                contribution=float(shap_values[i]),
                scaled_value=float(scaled_features.get(feature, 0.0)),
                value=float(scaled_features.get(feature, 0.0)),
                unit=_unit_for(feature),
                shap_value=float(shap_values[i]),
            )
            for i, feature in enumerate(self._feature_cols)
        ]


class TwoStageExplainer:
    """Facade — 두 단계 설명을 합쳐 환자별 상위 기여 피처를 돌려준다.

    Stage1(AKI 발생)·Stage2(중증도)는 기여 스케일이 달라 단순 합산하지 않고,
    각 단계 기여도를 그 단계 내 절대크기 합으로 정규화(0~1)한 뒤 평균낸다 →
    "전체 위험에 대한 상대적 영향력"으로 해석 가능.
    """

    def __init__(self) -> None:
        stage1, stage2 = model_bundles()
        self._stage1 = LinearModelExplainer(stage1["model"], list(stage1["feature_cols"]))
        self._stage2 = TreeModelExplainer(stage2["model"], list(stage2["feature_cols"]))

    @staticmethod
    def _normalize(contributions: list[FeatureContribution]) -> dict[str, float]:
        total = sum(abs(c.contribution) for c in contributions) or 1.0
        return {c.feature: c.contribution / total for c in contributions}

    def top_features(self, stay_id: int, limit: int = 8) -> list[FeatureContribution]:
        scaled = scaled_features_for_stay(stay_id)
        if scaled is None:
            return []
        stage1 = self._stage1.explain(scaled)
        stage2 = self._stage2.explain(scaled)
        s1_norm = self._normalize(stage1)
        s2_norm = self._normalize(stage2)
        by_feature = {c.feature: c for c in stage1}
        merged = []
        for feature, contrib in by_feature.items():
            blended = (s1_norm.get(feature, 0.0) + s2_norm.get(feature, 0.0)) / 2.0
            merged.append(
                FeatureContribution(
                    feature=feature,
                    label=contrib.label,
                    contribution=round(blended, 4),
                    scaled_value=round(contrib.scaled_value, 3),
                    value=contrib.value,
                    unit=contrib.unit,
                    shap_value=contrib.shap_value,
                )
            )
        merged.sort(key=lambda c: abs(c.contribution), reverse=True)
        return merged[:limit]
