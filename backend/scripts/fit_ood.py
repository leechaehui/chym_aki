"""⑥ OOD 분포 적합 — KPMP concept 분포(평균·역공분산·임계) → concept_distribution 적재.

사용: (team venv) PYTHONPATH=. python scripts/fit_ood.py [--percentile 97.5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from core.database import SessionLocal
import models.retrieval as R
from retrieval.concept.builder import load_kpmp_concepts
from retrieval.ood.detector import fit_distribution

ART = Path(__file__).resolve().parent.parent.parent / "Pathology_model" / "artifacts"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--percentile", type=float, default=97.5)
    args = ap.parse_args()

    concepts = load_kpmp_concepts(ART)
    vecs = np.stack([c.to_vector() for c in concepts.values()])
    params = fit_distribution(vecs, percentile=args.percentile)
    print(f"[ood] n_ref={params['n_ref']} threshold(p{args.percentile})={params['threshold']} "
          f"ref_dist={params['ref_dist_summary']}")

    db = SessionLocal()
    try:
        row = db.get(R.ConceptDistribution, 1)
        if row is None:
            row = R.ConceptDistribution(id=1)
            db.add(row)
        row.mean_vec = params["mean"]
        row.cov_inv = {"precision": params["precision"],
                       "ref_dist": params["ref_dist_summary"]}
        row.ood_threshold = params["threshold"]
        row.n_ref = params["n_ref"]
        db.commit()
        print("[ood] concept_distribution 적재 완료(id=1).")
    finally:
        db.close()


if __name__ == "__main__":
    main()
