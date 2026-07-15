"""OOD Detector — concept space Mahalanobis 거리로 분포 밖 환자 차단.

분포 밖(=KPMP 임상 phenotype 범위 밖)이면 retrieval 하지 않고
"No reliable pathology reference found." 를 반환(아무 프로토타입이나 반환 방지).

중요: MIMIC 전용 concept 차원(Cr trend·Oliguria)은 KPMP 에 분산이 없어(상수) 포함 시
모든 MIMIC 가 OOD 가 된다 → **양쪽 공통 임상 차원(SHARED_DIMS)만**으로 거리를 계산한다.
"""
from __future__ import annotations

import numpy as np
from sqlalchemy.orm import Session

import models.retrieval as R
from ..concept.schema import DIM

# 공유 임상 차원(전체 16 중 cr_trend(1)·oliguria(2) 제외 = MIMIC 전용).
SHARED_DIMS = [i for i in range(DIM) if i not in (1, 2)]


def _mahalanobis(x: np.ndarray, mean: np.ndarray, prec: np.ndarray) -> float:
    d = x - mean
    return float(np.sqrt(max(0.0, d @ prec @ d)))


class OodDetector:
    """concept_distribution(평균·역공분산·임계) 로부터 OOD 판정. DB 1행 싱글턴."""

    def __init__(self, mean: np.ndarray, prec: np.ndarray, threshold: float):
        self.mean, self.prec, self.threshold = mean, prec, threshold

    def score(self, concept_vec: np.ndarray) -> tuple[float, bool]:
        x = np.asarray(concept_vec, float)[SHARED_DIMS]
        d = _mahalanobis(x, self.mean, self.prec)
        return round(d, 4), bool(d > self.threshold)

    @classmethod
    def load(cls, db: Session) -> "OodDetector | None":
        row = db.get(R.ConceptDistribution, 1)
        if row is None or row.ood_threshold is None:
            return None
        mean = np.asarray(row.mean_vec, float)
        prec = np.asarray(row.cov_inv["precision"], float)
        return cls(mean, prec, float(row.ood_threshold))


def fit_distribution(member_vecs: np.ndarray, *, percentile: float = 97.5,
                     shrink: float = 0.1) -> dict:
    """KPMP 멤버 concept 벡터(weighted, 전체 16d) → 분포 파라미터.

    SHARED_DIMS 부분공간에서 평균·(수축)공분산·역행렬·임계(분위수) 산출.
    shrink: 공분산 대각 수축(소표본·저분산 차원 안정화, 특이 방지).
    """
    X = np.asarray(member_vecs, float)[:, SHARED_DIMS]
    mean = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    cov = (1 - shrink) * cov + shrink * np.eye(cov.shape[0]) * np.trace(cov) / cov.shape[0]
    prec = np.linalg.pinv(cov)
    dists = np.array([_mahalanobis(x, mean, prec) for x in X])
    threshold = float(np.percentile(dists, percentile))
    return {
        "mean": mean.tolist(),
        "precision": prec.tolist(),
        "threshold": round(threshold, 4),
        "n_ref": int(len(X)),
        "ref_dist_summary": {
            "p50": round(float(np.percentile(dists, 50)), 3),
            "p95": round(float(np.percentile(dists, 95)), 3),
            "max": round(float(dists.max()), 3),
        },
    }
