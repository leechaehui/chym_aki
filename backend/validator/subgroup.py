"""Subgroup analysis + missing-data sensitivity (작업지시서 4.1).

- subgroup_metrics: 임의의 그룹 키별로 AUROC/AUPRC/유병률을 분리 산출.
- missing_sensitivity: *_missing 플래그(데이터셋 제공)를 이용해
  '결측이 많은 환자'와 '결측이 적은 환자'의 성능 차이를 정량화한다.
- feature_ablation: 특정 피처를 결측 처리(0)했을 때의 성능 저하를 측정한다
  (예: 약물 정보 포함/미포함 성능 차이 — 4.2).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from validator.metrics import auprc, auroc


@dataclass(frozen=True)
class SubgroupResult:
    name: str
    n: int
    positives: int
    prevalence: float
    auroc: float
    auprc: float

    def as_dict(self) -> dict:
        return {
            "subgroup": self.name,
            "n": self.n,
            "positives": self.positives,
            "prevalence": round(self.prevalence, 4),
            "auroc": round(self.auroc, 4) if self.auroc == self.auroc else None,
            "auprc": round(self.auprc, 4) if self.auprc == self.auprc else None,
        }


def subgroup_metrics(y_true, prob, groups, min_n: int = 30) -> list[SubgroupResult]:
    """groups(범주형 배열) 값별로 성능을 분리 산출. min_n 미만 그룹은 제외."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    g = np.asarray(groups)
    out: list[SubgroupResult] = []
    for val in sorted(set(g.tolist()), key=str):
        mask = g == val
        n = int(mask.sum())
        if n < min_n:
            continue
        ys, ps = y[mask], p[mask]
        out.append(
            SubgroupResult(
                name=str(val),
                n=n,
                positives=int(ys.sum()),
                prevalence=float(ys.mean()),
                auroc=auroc(ys, ps),
                auprc=auprc(ys, ps),
            )
        )
    return out


def age_band(age) -> str:
    """연령 → 임상 표준 구간 라벨(원시 연령 입력용)."""
    a = float(age)
    if a < 40:
        return "<40"
    if a < 60:
        return "40-59"
    if a < 75:
        return "60-74"
    return ">=75"


def quantile_bands(values, labels=("Q1(낮음)", "Q2(중간)", "Q3(높음)")) -> np.ndarray:
    """연속값을 분위수 구간 라벨로 변환.

    데이터셋의 age 가 표준화(z-score)되어 절대 구간이 무의미할 때,
    상대적 분위(예: 연령 하위/중위/상위 1/3)로 subgroup 을 나눈다.
    """
    v = np.asarray(values, dtype=float)
    n = len(labels)
    edges = np.quantile(v, np.linspace(0, 1, n + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    out = np.empty(len(v), dtype=object)
    for i in range(n):
        mask = (v >= edges[i]) & (v < edges[i + 1] if i < n - 1 else v <= edges[i + 1])
        out[mask] = labels[i]
    return out


@dataclass(frozen=True)
class MissingSensitivityResult:
    overall_auroc: float
    low_missing_auroc: float
    high_missing_auroc: float
    median_missing_count: float
    delta: float  # low - high (양수면 결측 많을수록 성능 저하)

    def as_dict(self) -> dict:
        return {
            "overall_auroc": round(self.overall_auroc, 4),
            "low_missing_auroc": round(self.low_missing_auroc, 4),
            "high_missing_auroc": round(self.high_missing_auroc, 4),
            "median_missing_count": self.median_missing_count,
            "delta_low_minus_high": round(self.delta, 4),
        }


def missing_sensitivity(y_true, prob, missing_counts) -> MissingSensitivityResult:
    """환자별 결측 피처 수 기준 상/하위로 나눠 성능 차이를 본다."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    m = np.asarray(missing_counts, dtype=float)
    med = float(np.median(m))
    low = m <= med   # 결측 적음
    high = m > med   # 결측 많음
    a_low = auroc(y[low], p[low]) if low.sum() > 0 else float("nan")
    a_high = auroc(y[high], p[high]) if high.sum() > 0 else float("nan")
    return MissingSensitivityResult(
        overall_auroc=auroc(y, p),
        low_missing_auroc=a_low,
        high_missing_auroc=a_high,
        median_missing_count=med,
        delta=(a_low - a_high),
    )


@dataclass(frozen=True)
class AblationResult:
    feature_group: str
    base_auroc: float
    ablated_auroc: float
    auroc_drop: float

    def as_dict(self) -> dict:
        return {
            "feature_group": self.feature_group,
            "base_auroc": round(self.base_auroc, 4),
            "ablated_auroc": round(self.ablated_auroc, 4),
            "auroc_drop": round(self.auroc_drop, 4),
        }


def feature_ablation(
    predict_fn,
    X: np.ndarray,
    y_true,
    feature_index: dict[str, int],
    groups: dict[str, list[str]],
) -> list[AblationResult]:
    """피처 그룹을 0(결측)으로 만들고 AUROC 저하를 측정.

    predict_fn(X) -> prob 배열. groups 예: {"약물": ["vasopressor_flag", ...]}.
    '약물 정보 포함/미포함 성능 차이 정량화'(4.2)를 직접 답한다.
    """
    y = np.asarray(y_true, dtype=int)
    base = auroc(y, predict_fn(X))
    out: list[AblationResult] = []
    for gname, cols in groups.items():
        Xa = X.copy()
        for c in cols:
            if c in feature_index:
                Xa[:, feature_index[c]] = 0.0
        ablated = auroc(y, predict_fn(Xa))
        out.append(
            AblationResult(
                feature_group=gname,
                base_auroc=base,
                ablated_auroc=ablated,
                auroc_drop=base - ablated,
            )
        )
    return out
