"""Validation — time-series 재구성/누수/슬라이딩윈도 (작업지시서 4.3 / 5)."""
import pandas as pd

from validator.timeseries import (
    check_label_consistency,
    check_no_leakage,
    reconstruct_timeline,
    sliding_windows,
)


def _df():
    return pd.DataFrame(
        {
            "aki_label": [1, 1, 0, 0],
            "prediction_cutoff": [
                "2026-01-01T00:00:00",
                "2026-01-02T00:00:00",
                "2026-01-03T00:00:00",
                "2026-01-04T00:00:00",
            ],
            "aki_onset_time": [
                "2026-01-01T12:00:00",  # cutoff 이후 12h → 누수 없음
                "2026-01-02T06:00:00",
                None,
                None,
            ],
        }
    )


def test_no_leakage_when_cutoff_precedes_onset():
    chk = check_no_leakage(_df())
    assert chk.passed
    assert chk.n_leakage == 0
    assert chk.n_aki == 2


def test_leakage_detected_when_cutoff_after_onset():
    df = _df()
    df.loc[0, "aki_onset_time"] = "2025-12-31T00:00:00"  # cutoff 이전(미래정보 누수)
    chk = check_no_leakage(df)
    assert not chk.passed
    assert chk.n_leakage == 1


def test_label_consistency_matches_dataset():
    chk = check_label_consistency(_df(), lead_hours=48)
    assert chk.passed
    assert chk.match_rate == 1.0


def test_reconstruct_timeline_sorts_by_time():
    events = [
        {"event_time": "2026-01-03"},
        {"event_time": "2026-01-01"},
        {"event_time": "2026-01-02"},
    ]
    out = reconstruct_timeline(events)
    assert [e["event_time"] for e in out] == ["2026-01-01", "2026-01-02", "2026-01-03"]


def test_sliding_window_clips_to_lookback():
    ts = ["2026-01-01T00:00", "2026-01-01T20:00", "2026-01-02T00:00"]
    kept = sliding_windows(ts, cutoff="2026-01-02T00:00", lookback_hours=12)
    # cutoff-12h = 2026-01-01T12:00 이후만 남는다.
    assert kept == ["2026-01-01T20:00", "2026-01-02T00:00"]
