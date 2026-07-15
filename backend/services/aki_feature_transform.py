"""AKI 피처 트랜스폼 — 학습 시 전처리(06_transform.py)를 추론에 그대로 재현.

순서(학습과 동일):
  1) log1p(LOG1P_COLS)
  2) sqrt(SQRT_COLS, 음수 clip)
  3) median 대치(변환 후 train median, transform_info['medians'])
  4) RobustScaler.transform (transform_info['scaler'], feature_cols 순서)

raw `final_features_48h`(app_schema) DataFrame → 모델 입력 행렬(numpy). transform_info.pkl 캐시.
"""
from __future__ import annotations

import functools
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import BACKEND_DIR

# AKI/transform/transform_info.pkl (백엔드 기준 상위 repo 의 AKI 폴더).
_TRANSFORM_INFO = BACKEND_DIR.parent / "AKI" / "transform" / "transform_info.pkl"


@functools.lru_cache(maxsize=1)
def _info() -> dict:
    with open(_TRANSFORM_INFO, "rb") as f:
        return pickle.load(f)


def available() -> bool:
    return _TRANSFORM_INFO.exists()


def feature_cols() -> list[str]:
    return list(_info()["feature_cols"])


def transform(df: pd.DataFrame) -> np.ndarray:
    """raw 피처 DataFrame → 표준화된 모델 입력 행렬(feature_cols 순서)."""
    info = _info()
    cols = info["feature_cols"]
    X = df.reindex(columns=cols).copy().astype(float)

    for c in info.get("log1p_cols", []):
        X[c] = np.log1p(X[c])
    for c in info.get("sqrt_cols", []):
        X[c] = np.sqrt(X[c].clip(lower=0))

    # 변환 후 train median 으로 결측 대치(학습과 동일 값).
    X = X.fillna(value=info["medians"])
    # 남은 결측(혹시 medians 에 없는 컬럼)은 0.
    X = X.fillna(0.0)

    return info["scaler"].transform(X[cols])
