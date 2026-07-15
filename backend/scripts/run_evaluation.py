"""⑦ Retrieval Evaluation 리포트 — Ground Truth 없는 cross-cohort 평가(실데이터).

쿼리셋 = 실 KPMP 멤버 concept(in-distribution) + 합성 OOD 케이스.
각 쿼리에 OOD 게이트 + Hybrid Retrieve 를 돌려 로그 레코드를 만들고,
evaluation 모듈(abstain/OOD/prototype 분포/consistency)로 집계한다.
산출: results/retrieval_evaluation.json + 콘솔 요약.

사용: (team venv) PYTHONPATH=. python scripts/run_evaluation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.database import SessionLocal
from retrieval.concept.builder import build_mimic_concept, load_kpmp_concepts
from retrieval.engine.retriever import HybridRetriever
from retrieval.evaluation.abstain_rate import abstain_rate
from retrieval.evaluation.ood_statistics import ood_statistics
from retrieval.evaluation.prototype_distribution import prototype_distribution
from retrieval.evaluation.retrieval_consistency import neighbor_consistency
from retrieval.ood.detector import OodDetector

ART = Path(__file__).resolve().parent.parent.parent / "Pathology_model" / "artifacts"
OUT = Path(__file__).resolve().parent.parent.parent / "Pathology_model" / "results"

# 합성 OOD 케이스(KPMP 분포 밖이어야 abstain 되어야 정상).
OOD_CASES = [
    dict(kdigo_stage=0, egfr=140, proteinuria_mg_g=10, a1c_pct=5.0, age=8, sex="Male",
         diabetes=False, hypertension=False),           # 건강 소아
    dict(kdigo_stage=0, egfr=130, proteinuria_mg_g=5, a1c_pct=4.8, age=12, sex="Female",
         diabetes=False, hypertension=False),           # 건강 소아2
]


def main() -> None:
    concepts = load_kpmp_concepts(ART)
    db = SessionLocal()
    try:
        ood = OodDetector.load(db)
        retr = HybridRetriever(db, k_final=5)
        records: list[dict] = []

        def run(concept, tag):
            score, is_ood = ood.score(concept.to_vector()) if ood else (None, False)
            hits = [] if is_ood else retr.retrieve(concept)
            records.append({
                "tag": tag, "is_ood": is_ood, "ood_score": score,
                "hits": [{"prototype_id": h.prototype_id, "similarity": h.similarity} for h in hits],
                "concept_vec": concept.to_vector().tolist(),
                "top1": hits[0].prototype_id if hits else None,
            })

        # in-distribution: 실 KPMP 멤버 concept
        for c in concepts.values():
            run(c, "kpmp")
        # OOD: 합성
        for kw in OOD_CASES:
            run(build_mimic_concept(etiology_hint="unknown", **kw), "ood_synth")

        report = {
            "n_queries": len(records),
            "n_in_distribution": sum(r["tag"] == "kpmp" for r in records),
            "n_ood_synthetic": sum(r["tag"] == "ood_synth" for r in records),
            "abstain_rate": abstain_rate(records),
            "ood_statistics": ood_statistics(records),
            "prototype_distribution": prototype_distribution(records),
            "retrieval_consistency": neighbor_consistency(records, k=5),
            "ood_synth_blocked": sum(r["is_ood"] for r in records if r["tag"] == "ood_synth"),
        }
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "retrieval_evaluation.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

        # 콘솔 요약
        print("=== Retrieval Evaluation (실데이터) ===")
        print(f"쿼리 {report['n_queries']}건 (in-dist {report['n_in_distribution']} + OOD합성 {report['n_ood_synthetic']})")
        print(f"abstain_rate     : {report['abstain_rate']['abstain_rate']} "
              f"({report['abstain_rate']['n_abstain']}/{report['abstain_rate']['n']})")
        print(f"OOD 합성 차단    : {report['ood_synth_blocked']}/{report['n_ood_synthetic']}")
        os_ = report["ood_statistics"]
        print(f"top1 similarity  : p50={os_['top1_similarity'].get('p50')} "
              f"min={os_['top1_similarity'].get('min')} max={os_['top1_similarity'].get('max')}")
        print(f"OOD score        : p50={os_['ood_score'].get('p50')} p95={os_['ood_score'].get('p95')}")
        pd_ = report["prototype_distribution"]
        print(f"prototype 사용   : {pd_['n_prototypes_used']}개, entropy_norm={pd_['entropy_norm']} (1=균등)")
        print(f"retrieval 일관성 : consistency@5={report['retrieval_consistency']['consistency_at_k']}")
        print(f"\n-> {OUT/'retrieval_evaluation.json'}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
