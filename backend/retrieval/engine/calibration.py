"""⑦ Calibration — raw cosine similarity → 보정 confidence.

Ground Truth 가 없으므로(cross-cohort) similarity 의 **경험적 분포(ECDF)**를 isotonic 으로
적합해, "이 매칭이 전형적 매칭 대비 얼마나 강한가"를 confidence 로 표현한다.
코사인은 0.85~0.95 로 뭉치므로 raw 값은 비정보적 → 분포 기준 재척도(monotone)한다.
concept completeness 를 곱해 결측 많은 입력의 confidence 를 낮춘다(보수적).

저장: concept_distribution.cov_inv["calibration"] = {"x": breakpoints, "y": calibrated}.
"""
from __future__ import annotations

import numpy as np
from sqlalchemy.orm import Session

import models.retrieval as R


class Calibrator:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        # 단조 증가 보정맵(x=similarity, y=confidence). 범위 밖은 clip.
        self._x, self._y = np.asarray(x, float), np.asarray(y, float)

    def calibrate(self, similarity: float, *, completeness: float = 1.0) -> float:
        base = float(np.interp(similarity, self._x, self._y))
        return float(np.clip(base * (0.5 + 0.5 * completeness), 0.0, 1.0))

    @classmethod
    def fit(cls, reference_sims: list[float]) -> "Calibrator":
        from sklearn.isotonic import IsotonicRegression
        s = np.sort(np.asarray(reference_sims, float))
        ecdf = (np.arange(1, len(s) + 1)) / len(s)        # 경험적 CDF
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        yhat = iso.fit_transform(s, ecdf)
        # 중복 x 제거(np.interp 용 단조 grid)
        xu, idx = np.unique(s, return_index=True)
        return cls(xu, yhat[idx])

    def to_dict(self) -> dict:
        return {"x": self._x.tolist(), "y": self._y.tolist()}

    @classmethod
    def load(cls, db: Session) -> "Calibrator | None":
        row = db.get(R.ConceptDistribution, 1)
        cal = (row.cov_inv or {}).get("calibration") if row else None
        if not cal:
            return None
        return cls(np.asarray(cal["x"]), np.asarray(cal["y"]))
