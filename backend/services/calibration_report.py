"""신뢰도 곡선(Calibration / Reliability diagram) — ECE 숫자를 그래프로 대체.

"예측 확률 80% 구간의 환자들이 실제로 몇 % AKI였는가"를 구간(bin)별로 집계한다.
완벽히 보정된 모델이면 예측확률 ≈ 실제발생률(대각선). 코호트 예측표(p_aki vs 실제 라벨)
한 곳에서 계산하며 프로세스당 1회 캐시한다.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass


@dataclass(frozen=True)
class ReliabilityBin:
    """확률 구간 하나의 보정 상태."""

    predicted_mean: float   # 구간 내 평균 예측확률 (x축)
    observed_rate: float    # 구간 내 실제 AKI 발생률 (y축)
    count: int              # 구간 표본 수


@dataclass(frozen=True)
class CalibrationReport:
    bins: list[ReliabilityBin]
    expected_calibration_error: float   # ECE — 그래프의 평균 이탈(참고 수치)
    n: int


@functools.lru_cache(maxsize=1)
def get_calibration_report(n_bins: int = 10) -> CalibrationReport:
    """코호트 전체에 대한 신뢰도 곡선(캐시). 코호트 미가용 시 빈 보고서."""
    try:
        import numpy as np

        from services.icu_monitor_service import _predictions, is_available

        if not is_available():
            raise RuntimeError("cohort unavailable")

        df = _predictions()
        predicted = df["p_aki"].to_numpy(dtype=float)
        observed = (df["actual_label"].to_numpy() >= 1).astype(int)

        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_index = np.clip(np.digitize(predicted, edges[1:-1]), 0, n_bins - 1)

        bins: list[ReliabilityBin] = []
        weighted_gap = 0.0
        for b in range(n_bins):
            mask = bin_index == b
            count = int(mask.sum())
            if count == 0:
                continue
            predicted_mean = float(predicted[mask].mean())
            observed_rate = float(observed[mask].mean())
            bins.append(ReliabilityBin(
                predicted_mean=round(predicted_mean, 4),
                observed_rate=round(observed_rate, 4),
                count=count,
            ))
            weighted_gap += count * abs(predicted_mean - observed_rate)

        total = int(len(predicted))
        ece = round(weighted_gap / total, 4) if total else 0.0
        return CalibrationReport(bins=bins, expected_calibration_error=ece, n=total)
    except Exception:  # noqa: BLE001 — 보정 곡선 실패는 빈 보고서로 폴백
        return CalibrationReport(bins=[], expected_calibration_error=0.0, n=0)
