"""ROC 기반 임계값 보정 (지시서 §6).

static threshold 금지 — AKI 확률(0~1)의 operating point 를 **실제 라벨 코호트의 ROC**에서
Youden Index(J = sensitivity + specificity - 1) 최대 지점으로 산출한다.

- 코호트(MIMIC-IV ICU, services.icu_monitor_service)의 모델 확률 p_aki vs 실제 AKI 라벨로 보정.
- 코호트/sklearn 미가용 시 safety-first 기본값 0.6 으로 폴백(지시서 §6.3 초기값).
- 3-zone(§6.5): Normal[0,0.5) · Pre-AKI[0.5,0.75) · AKI-likely[0.75,1.0].
프로세스당 1회 계산 후 캐시.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

from core.logging import get_logger

log = get_logger("chym.roc")

# safety-first 초기 운영 임계값(지시서 §6.3).
_DEFAULT_THRESHOLD = 0.6
# 3-zone 경계(지시서 §6.5).
PRE_AKI_ZONE = 0.5
AKI_LIKELY_ZONE = 0.75


@dataclass(frozen=True)
class Calibration:
    threshold: float          # Youden-최적 operating point
    sensitivity: float        # 해당 지점 recall
    specificity: float
    youden_j: float
    n: int                    # 보정 표본 수
    source: str               # "roc_youden" | "default"


@functools.lru_cache(maxsize=1)
def get_calibration() -> Calibration:
    """ROC/Youden 보정값(캐시). 실패 시 안전 기본값."""
    try:
        import numpy as np
        from sklearn.metrics import roc_curve

        from services.icu_monitor_service import _predictions, is_available

        if not is_available():
            raise RuntimeError("cohort unavailable")

        df = _predictions()
        y = (df["actual_label"] >= 1).astype(int)
        score = df["p_aki"]
        fpr, tpr, thresholds = roc_curve(y, score)
        j = tpr - fpr
        i = int(np.argmax(j))
        # roc_curve 의 첫 threshold 는 inf 일 수 있으므로 클램프.
        th = float(min(max(thresholds[i], 0.01), 0.99))
        cal = Calibration(
            threshold=round(th, 3),
            sensitivity=round(float(tpr[i]), 3),
            specificity=round(float(1 - fpr[i]), 3),
            youden_j=round(float(j[i]), 3),
            n=int(len(df)),
            source="roc_youden",
        )
        log.info("ROC calibration: threshold=%.3f sens=%.2f spec=%.2f (n=%d)",
                 cal.threshold, cal.sensitivity, cal.specificity, cal.n)
        return cal
    except Exception:  # noqa: BLE001 — 보정 실패는 안전 기본값으로 폴백
        log.warning("ROC calibration failed → default threshold %.2f", _DEFAULT_THRESHOLD, exc_info=True)
        return Calibration(
            threshold=_DEFAULT_THRESHOLD, sensitivity=0.0, specificity=0.0,
            youden_j=0.0, n=0, source="default",
        )


def operating_threshold() -> float:
    """현재 운영 임계값(Youden-최적 또는 기본값)."""
    return get_calibration().threshold


def zone(prob: float) -> str:
    """3-zone 분류(지시서 §6.5)."""
    if prob >= AKI_LIKELY_ZONE:
        return "AKI_LIKELY"
    if prob >= PRE_AKI_ZONE:
        return "PRE_AKI"
    return "NORMAL"
