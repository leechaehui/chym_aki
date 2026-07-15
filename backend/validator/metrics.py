"""분류 성능 지표 — 순수 함수(외부 상태 없음).

작업지시서 4.1 요구사항을 충족한다:
- AUROC, AUPRC 모두 출력
- calibration curve(신뢰도-관측빈도) 확인
- Brier score / ECE(Expected Calibration Error) 로 보정 수준 정량화

sklearn 이 있으면 AUROC/AUPRC 는 sklearn 으로 계산(검증된 구현),
없으면 numpy 기반 폴백을 사용해 의존성 없이도 동작한다(에어갭 대비).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CalibrationBin:
    """보정 곡선 한 구간(확률 bin)."""

    lower: float
    upper: float
    count: int
    mean_predicted: float
    observed_rate: float


@dataclass(frozen=True)
class BinaryMetrics:
    """이진 분류 성능 요약."""

    n: int
    positives: int
    prevalence: float
    auroc: float
    auprc: float
    brier: float
    ece: float
    calibration: list[CalibrationBin]

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "positives": self.positives,
            "prevalence": round(self.prevalence, 4),
            "auroc": round(self.auroc, 4),
            "auprc": round(self.auprc, 4),
            "brier": round(self.brier, 4),
            "ece": round(self.ece, 4),
            "calibration": [
                {
                    "bin": f"[{b.lower:.1f},{b.upper:.1f})",
                    "count": b.count,
                    "mean_predicted": round(b.mean_predicted, 4),
                    "observed_rate": round(b.observed_rate, 4),
                }
                for b in self.calibration
            ],
        }


def _auroc_np(y: np.ndarray, p: np.ndarray) -> float:
    """AUROC = Mann-Whitney U 통계(순위 기반). sklearn 폴백."""
    pos = p[y == 1]
    neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    # 동점 평균 순위 보정.
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    avg_rank = cum - (counts - 1) / 2.0
    ranks = avg_rank[inv]
    sum_pos = ranks[y == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _auprc_np(y: np.ndarray, p: np.ndarray) -> float:
    """Average Precision(PR 곡선 아래 면적, 계단 합). sklearn 폴백."""
    order = np.argsort(-p, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / (tp + fp)
    total_pos = y.sum()
    if total_pos == 0:
        return float("nan")
    recall = tp / total_pos
    # recall 증가분 가중 precision 합(sklearn average_precision_score 정의).
    rec_prev = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - rec_prev) * precision))


def auroc(y_true, prob) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    try:
        from sklearn.metrics import roc_auc_score

        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, p))
    except Exception:
        return _auroc_np(y, p)


def auprc(y_true, prob) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    try:
        from sklearn.metrics import average_precision_score

        if len(np.unique(y)) < 2:
            return float("nan")
        return float(average_precision_score(y, p))
    except Exception:
        return _auprc_np(y, p)


def calibration_table(y_true, prob, n_bins: int = 10) -> list[CalibrationBin]:
    """예측확률을 동일 폭 bin 으로 나눠 (평균예측, 관측빈도)를 산출."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[CalibrationBin] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        bins.append(
            CalibrationBin(
                lower=float(lo),
                upper=float(hi),
                count=cnt,
                mean_predicted=float(p[mask].mean()),
                observed_rate=float(y[mask].mean()),
            )
        )
    return bins


def expected_calibration_error(bins: list[CalibrationBin], n_total: int) -> float:
    """ECE = Σ (bin 가중치) · |평균예측 - 관측빈도|."""
    if n_total == 0:
        return float("nan")
    return float(
        sum(b.count / n_total * abs(b.mean_predicted - b.observed_rate) for b in bins)
    )


def brier_score(y_true, prob) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)
    return float(np.mean((p - y) ** 2))


def binary_metrics(y_true, prob, n_bins: int = 10) -> BinaryMetrics:
    """이진 분류 성능 일괄 산출(AUROC/AUPRC/Brier/ECE/calibration)."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    bins = calibration_table(y, p, n_bins)
    return BinaryMetrics(
        n=len(y),
        positives=int(y.sum()),
        prevalence=float(y.mean()) if len(y) else float("nan"),
        auroc=auroc(y, p),
        auprc=auprc(y, p),
        brier=brier_score(y, p),
        ece=expected_calibration_error(bins, len(y)),
        calibration=bins,
    )
