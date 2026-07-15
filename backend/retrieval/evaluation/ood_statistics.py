"""OOD Statistics — OOD score 분포(히스토그램·분위수) 및 임계 통계.

"분포 밖 환자를 실제로 걸러내는가"를 정량화. similarity 분포도 함께 본다.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def _quantiles(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    a = np.asarray(xs, float)
    return {
        "n": len(a), "min": round(float(a.min()), 4), "max": round(float(a.max()), 4),
        "mean": round(float(a.mean()), 4),
        "p25": round(float(np.percentile(a, 25)), 4),
        "p50": round(float(np.percentile(a, 50)), 4),
        "p95": round(float(np.percentile(a, 95)), 4),
    }


def ood_statistics(logs: Iterable[dict], bins: int = 10) -> dict:
    """반환: OOD score 분위수 + top1 similarity 분위수 + 히스토그램."""
    rows = list(logs)
    ood = [r["ood_score"] for r in rows if r.get("ood_score") is not None]
    top1 = [r["hits"][0]["similarity"] for r in rows if r.get("hits")]
    hist, edges = (np.histogram([s for s in top1], bins=bins, range=(0, 1))
                   if top1 else (np.array([]), np.array([])))
    return {
        "ood_score": _quantiles(ood),
        "top1_similarity": _quantiles(top1),
        "similarity_histogram": {
            "counts": hist.tolist(), "bin_edges": [round(float(e), 2) for e in edges]},
    }
