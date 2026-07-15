"""Time-series validation (작업지시서 4.3 / 5).

AKI 는 반드시 시간축 데이터로 검증한다. 이 모듈은 데이터셋이 제공하는
시간 컬럼(index_time, prediction_cutoff, aki_onset_time)을 이용해:

- patient timeline reconstruction : stay_id 단위로 이벤트를 시간순 정렬.
- event-based labeling           : aki_onset_time 이 예측 시점(prediction_cutoff)
                                    이후 lead-time window 안에 들어오는지로 라벨을 재구성하고
                                    데이터셋의 aki_label 과 일치(consistency)하는지 확인.
- sliding window                  : 예측 시점 기준 관측창(lookback) 구성 규약 검증
                                    (cutoff 가 onset 보다 항상 앞서는지 = 누수 없음).

여기서는 '검증'만 한다. 라벨 정의를 바꾸지 않고, 데이터셋 라벨과의 정합성만 본다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LeakageCheck:
    """예측 시점이 onset 보다 앞서는가(미래 정보 누수 없음)."""

    n_aki: int
    n_cutoff_before_onset: int
    n_leakage: int  # cutoff 가 onset 이후인 행(누수 의심)
    passed: bool

    def as_dict(self) -> dict:
        return {
            "n_aki": self.n_aki,
            "n_cutoff_before_onset": self.n_cutoff_before_onset,
            "n_leakage": self.n_leakage,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class LabelConsistency:
    """onset 기반 재구성 라벨과 데이터셋 aki_label 의 정합."""

    n: int
    n_match: int
    match_rate: float
    passed: bool

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "n_match": self.n_match,
            "match_rate": round(self.match_rate, 4),
            "passed": self.passed,
        }


def _to_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def check_no_leakage(df: pd.DataFrame) -> LeakageCheck:
    """AKI 환자에서 prediction_cutoff < aki_onset_time 이어야 누수가 없다."""
    aki = df[df["aki_label"] == 1].copy()
    cutoff = _to_dt(aki["prediction_cutoff"])
    onset = _to_dt(aki["aki_onset_time"])
    valid = cutoff.notna() & onset.notna()
    before = (cutoff[valid] < onset[valid]).sum()
    leakage = (cutoff[valid] >= onset[valid]).sum()
    n_aki = int(valid.sum())
    return LeakageCheck(
        n_aki=n_aki,
        n_cutoff_before_onset=int(before),
        n_leakage=int(leakage),
        passed=bool(leakage == 0 and n_aki > 0),
    )


def check_label_consistency(df: pd.DataFrame, lead_hours: float = 48.0) -> LabelConsistency:
    """onset 이 cutoff 이후 lead_hours 안이면 AKI(=1)로 재구성, 데이터셋 라벨과 비교.

    Non-AKI 행은 onset 이 없으므로 재구성 라벨 0 으로 둔다.
    """
    cutoff = _to_dt(df["prediction_cutoff"])
    onset = _to_dt(df["aki_onset_time"])
    delta_h = (onset - cutoff).dt.total_seconds() / 3600.0
    reconstructed = ((onset.notna()) & (delta_h > 0) & (delta_h <= lead_hours)).astype(int)
    # onset 은 있으나 window 밖인 경우도 데이터셋은 AKI 일 수 있어, AKI 행은 onset 존재만으로 1 처리.
    reconstructed = reconstructed.where(df["aki_label"] == 0, (onset.notna()).astype(int))
    match = (reconstructed.to_numpy() == df["aki_label"].to_numpy()).sum()
    n = len(df)
    return LabelConsistency(
        n=n,
        n_match=int(match),
        match_rate=float(match / n) if n else float("nan"),
        passed=bool(n and match / n >= 0.99),
    )


def reconstruct_timeline(events: list[dict], time_key: str = "event_time") -> list[dict]:
    """이벤트 리스트를 시간 오름차순으로 정렬해 환자 타임라인을 재구성.

    timeline_service / DB 의 event 와 동일한 구조(dict)를 받는다.
    동일 시각은 입력 순서를 보존(stable sort)한다.
    """
    return sorted(events, key=lambda e: e.get(time_key) or "")


def sliding_windows(
    timestamps: list, cutoff, lookback_hours: float
) -> list:
    """cutoff 기준 [cutoff - lookback, cutoff] 관측창에 드는 타임스탬프만 반환.

    예측에 미래 정보가 섞이지 않도록(누수 방지) 관측창을 명시적으로 자르는 규약.
    """
    cut = pd.to_datetime(cutoff)
    start = cut - pd.Timedelta(hours=lookback_hours)
    ts = pd.to_datetime(pd.Series(timestamps), errors="coerce")
    keep = (ts >= start) & (ts <= cut)
    return [t for t, k in zip(timestamps, keep.tolist()) if k]
