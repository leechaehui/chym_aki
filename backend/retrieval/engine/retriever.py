"""Hybrid Retriever — 3-stage (규칙 prefilter → metadata 코사인 → 대표 WSI).

Frozen flow: Clinical Concept → Rule Filter → Prototype Candidate → metadata_vec cosine
            → Top-K → Representative WSI.
OOD 게이트는 ⑥에서 retrieve 앞단에 붙인다(현재는 retriever 자체에 미포함 — 우선순위상 ③ 먼저).
출력은 Evidence 전용(진단/reasoning 없음).
"""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from ..concept.schema import ConceptVector
from ..repository import PrototypeRepository


@dataclass(frozen=True)
class RepresentativeWsi:
    slide_id: str
    stain: str | None
    source: str


@dataclass(frozen=True)
class RetrievalHit:
    prototype_id: str
    label: str
    similarity: float
    n_members: int
    is_rare: bool
    kdigo_band: str | None
    etiology_hint: str | None
    egfr_mean: float | None
    representative_wsi: RepresentativeWsi | None


class HybridRetriever:
    def __init__(self, db: Session, *, k_filter: int = 50, k_final: int = 5):
        self.repo = PrototypeRepository(db)
        self.k_filter, self.k_final = k_filter, k_final

    def retrieve(self, concept: ConceptVector) -> list[RetrievalHit]:
        cands = self.repo.filter_candidates(
            kdigo_band=concept.kdigo_band(), etiology=concept.etiology_hint,
            limit=self.k_filter)
        ranked = self.repo.rank_by_metadata_similarity(
            concept_vec=concept.to_vector(), candidates=cands, limit=self.k_final)

        hits: list[RetrievalHit] = []
        for r in ranked:
            p = r.prototype
            w = self.repo.representative_wsi(p.id)
            hits.append(RetrievalHit(
                prototype_id=p.id, label=p.label, similarity=round(r.similarity, 4),
                n_members=p.n_members, is_rare=p.is_rare,
                kdigo_band=p.kdigo_band, etiology_hint=p.etiology_hint,
                egfr_mean=p.egfr_mean,
                representative_wsi=(RepresentativeWsi(w.slide_id, w.stain, w.source)
                                   if w else None),
            ))
        return hits
