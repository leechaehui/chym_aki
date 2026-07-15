"""Unit — validator.metrics (Verification: 검증 모듈 자체의 정확성)."""
import numpy as np

from validator.metrics import (
    auprc,
    auroc,
    binary_metrics,
    brier_score,
    calibration_table,
    expected_calibration_error,
)


def test_auroc_perfect_separation():
    y = [0, 0, 1, 1]
    p = [0.1, 0.2, 0.8, 0.9]
    assert auroc(y, p) == 1.0


def test_auroc_inverted_is_zero():
    y = [0, 0, 1, 1]
    p = [0.9, 0.8, 0.2, 0.1]
    assert auroc(y, p) == 0.0


def test_auroc_random_is_half():
    y = [0, 1, 0, 1]
    p = [0.5, 0.5, 0.5, 0.5]
    assert abs(auroc(y, p) - 0.5) < 1e-9


def test_auroc_numpy_fallback_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = rng.random(500)
    from validator.metrics import _auroc_np

    assert abs(_auroc_np(np.array(y), np.array(p)) - auroc(y, p)) < 1e-6


def test_auprc_at_least_prevalence():
    y = [0, 0, 0, 1]
    p = [0.2, 0.3, 0.4, 0.9]
    assert auprc(y, p) == 1.0  # 완전 분리


def test_brier_perfect_is_zero():
    assert brier_score([0, 1], [0.0, 1.0]) == 0.0


def test_calibration_table_bins_and_ece():
    y = [0] * 50 + [1] * 50
    p = [0.05] * 50 + [0.95] * 50
    bins = calibration_table(y, p, n_bins=10)
    # 두 개 bin 만 채워짐(0.0-0.1, 0.9-1.0).
    assert len(bins) == 2
    ece = expected_calibration_error(bins, len(y))
    # 잘 보정된 경우 ECE 작음.
    assert ece < 0.1


def test_binary_metrics_contract():
    y = [0, 0, 1, 1, 1]
    p = [0.1, 0.4, 0.6, 0.8, 0.9]
    m = binary_metrics(y, p)
    assert m.n == 5
    assert m.positives == 3
    assert 0.0 <= m.auroc <= 1.0
    assert 0.0 <= m.auprc <= 1.0
    d = m.as_dict()
    assert {"auroc", "auprc", "brier", "ece", "calibration"} <= set(d)
