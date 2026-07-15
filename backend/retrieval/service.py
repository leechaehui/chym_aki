"""RetrievalService — Concept 빌드 → (OOD 게이트) → Hybrid Retrieve → 로깅 → Evidence.

OOD 는 ⑥에서 주입(현재 placeholder: 항상 pass). Calibration 은 ⑦에서(현재 confidence=similarity).
"""
from __future__ import annotations

from sqlalchemy.orm import Session

import models.retrieval as R
from schemas.retrieval import (
    ConceptOut, OodOut, RepresentativeWsiOut, RetrievalHitOut, RetrievalQueryIn,
    RetrievalResultOut,
)

from .concept.builder import build_mimic_concept
from .engine.retriever import HybridRetriever

REFERENCE_NOTICE = "참조용 — 진단이 아닙니다. 임상 phenotype 이 유사한 KPMP 프로토타입과 대표 WSI 입니다."


class RetrievalService:
    def __init__(self, db: Session, *, ood_detector=None, calibrator=None):
        self.db = db
        self.ood = ood_detector          # ⑥에서 주입
        self.calibrator = calibrator     # ⑦에서 주입

    def query(self, body: RetrievalQueryIn) -> RetrievalResultOut:
        # stay_id 지정 시 MIMIC ICU 에서 concept 자동 추출(없으면 수동 입력값).
        concept = None
        if body.stay_id is not None:
            from .concept.mimic_adapter import build_concept_from_stay
            concept = build_concept_from_stay(body.stay_id)
        if concept is None:
            concept = build_mimic_concept(
                kdigo_stage=body.kdigo_stage, cr_trend_slope=body.cr_trend_slope,
                oliguria=body.oliguria, egfr=body.egfr,
                proteinuria_mg_g=body.proteinuria_mg_g, a1c_pct=body.a1c_pct,
                age=body.age, sex=body.sex, diabetes=body.diabetes,
                hypertension=body.hypertension, etiology_hint=body.etiology_hint,
            )

        # OOD 게이트(⑥ 전까지 pass)
        if self.ood is not None:
            score, is_ood = self.ood.score(concept.to_vector())
        else:
            score, is_ood = None, False
        ood_out = OodOut(score=score, is_ood=is_ood,
                         message=("신뢰할 만한 병리 참조를 찾지 못했습니다 (임상 분포 밖)." if is_ood else None))

        hits_out: list[RetrievalHitOut] = []
        if not is_ood:
            completeness = concept.completeness()
            for h in HybridRetriever(self.db, k_final=body.k).retrieve(concept):
                conf = (self.calibrator.calibrate(h.similarity, completeness=completeness)
                        if self.calibrator else h.similarity)
                w = h.representative_wsi
                hits_out.append(RetrievalHitOut(
                    prototype_id=h.prototype_id, label=h.label, similarity=h.similarity,
                    confidence=round(conf, 4), n_members=h.n_members, is_rare=h.is_rare,
                    kdigo_band=h.kdigo_band, etiology_hint=h.etiology_hint,
                    egfr_mean=h.egfr_mean,
                    representative_wsi=(RepresentativeWsiOut(
                        slide_id=w.slide_id, stain=w.stain, source=w.source) if w else None),
                ))

        concept_out = ConceptOut(
            kdigo_stage=concept.kdigo_stage, kdigo_band=concept.kdigo_band(),
            egfr=concept.egfr, proteinuria=concept.proteinuria, a1c=concept.a1c,
            age=concept.age, etiology_hint=concept.etiology_hint,
            completeness=concept.completeness())

        # audit + evaluation/calibration 소스
        self.db.add(R.RetrievalLog(
            subject_id=body.subject_id, concept_json=concept_out.model_dump(),
            ood_score=score, is_ood=is_ood,
            hits_json=[{"prototype_id": h.prototype_id, "similarity": h.similarity,
                        "confidence": h.confidence} for h in hits_out],
            raw_conf=(hits_out[0].similarity if hits_out else None),
            calibrated_conf=(hits_out[0].confidence if hits_out else None)))
        self.db.commit()

        return RetrievalResultOut(concept=concept_out, ood=ood_out, hits=hits_out,
                                  reference_notice=REFERENCE_NOTICE)
