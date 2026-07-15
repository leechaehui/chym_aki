"""⑦ Calibration 적합 — KPMP 멤버 self-retrieval top-1 similarity 분포로 isotonic 보정맵 적합.

reference 분포 = 각 KPMP 멤버 concept 로 검색했을 때의 top-1 유사도(=전형적 매칭 강도).
→ concept_distribution.cov_inv["calibration"] 에 저장. RetrievalService 가 confidence 산출에 사용.

사용: (team venv) PYTHONPATH=. python scripts/fit_calibration.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from core.database import SessionLocal
import models.retrieval as R
from retrieval.concept.builder import load_kpmp_concepts
from retrieval.engine.calibration import Calibrator
from retrieval.engine.retriever import HybridRetriever

ART = Path(__file__).resolve().parent.parent.parent / "Pathology_model" / "artifacts"


def main() -> None:
    concepts = load_kpmp_concepts(ART)
    db = SessionLocal()
    try:
        r = HybridRetriever(db, k_final=1)
        sims = [hits[0].similarity for c in concepts.values()
                if (hits := r.retrieve(c))]
        print(f"[calib] reference top-1 sims: n={len(sims)} "
              f"min={min(sims):.3f} p50={np.percentile(sims,50):.3f} max={max(sims):.3f}")

        cal = Calibrator.fit(sims)
        row = db.get(R.ConceptDistribution, 1)
        if row is None:
            print("[calib] concept_distribution 없음 — fit_ood 먼저 실행 필요"); return
        cov = dict(row.cov_inv or {})
        cov["calibration"] = cal.to_dict()
        row.cov_inv = cov
        db.commit()
        # 예시: raw sim → calibrated
        for s in (0.80, 0.88, 0.92, 0.96):
            print(f"  sim {s:.2f} → confidence {cal.calibrate(s):.3f}")
        print("[calib] 보정맵 저장 완료(concept_distribution.cov_inv.calibration).")
    finally:
        db.close()


if __name__ == "__main__":
    main()
