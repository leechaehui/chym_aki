"""Prototype Repository — 후보 규칙 필터 + metadata_vec 코사인 순위 + 대표 WSI.

현재 metadata_vec 는 double precision[](fallback) → 코사인은 Python 으로 계산(수십 prototype 규모).
pgvector 설치 시 rank_by_metadata_similarity 를 `metadata_vec <=> :qv` 로 교체만 하면 된다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sqlalchemy.orm import Session

import models.retrieval as R

# KDIGO band 인접(규칙 prefilter 의 ±1 허용).
_ADJ = {"0": ["0", "1"], "1": ["0", "1", "2-3"], "2-3": ["1", "2-3"],
        "unknown": ["0", "1", "2-3", "unknown"]}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


@dataclass(frozen=True)
class RankedPrototype:
    prototype: R.Prototype
    similarity: float


class PrototypeRepository:
    def __init__(self, db: Session):
        self.db = db

    # 1차: 임상 규칙 prefilter (kdigo band ±1, etiology lenient)
    def filter_candidates(
        self, *, kdigo_band: str, etiology: str, limit: int = 50,
    ) -> list[R.Prototype]:
        bands = _ADJ.get(kdigo_band, ["0", "1", "2-3", "unknown"]) + ["unknown"]
        q = self.db.query(R.Prototype).filter(
            (R.Prototype.kdigo_band.in_(bands)) | (R.Prototype.kdigo_band.is_(None)))
        cands = q.all()
        # etiology lenient: 일치 OR prototype unknown OR concept unknown.
        if etiology != "unknown":
            cands = [p for p in cands
                     if p.etiology_hint in (etiology, "unknown", None)]
        # 너무 적으면(과필터) 전체로 폴백 — 코사인이 순위를 책임진다.
        if len(cands) < 3:
            cands = self.db.query(R.Prototype).all()
        return cands[:limit] if limit else cands

    # 2차: concept-vec ↔ metadata_vec 코사인 순위 (후보 한정)
    def rank_by_metadata_similarity(
        self, *, concept_vec: np.ndarray, candidates: list[R.Prototype], limit: int = 5,
    ) -> list[RankedPrototype]:
        scored = [RankedPrototype(p, _cosine(concept_vec, np.asarray(p.metadata_vec, float)))
                  for p in candidates if p.metadata_vec]
        scored.sort(key=lambda x: x.similarity, reverse=True)
        return scored[:limit]

    # 3차: 대표 WSI(medoid)
    def representative_wsi(self, prototype_id: str) -> R.WsiMetadata | None:
        return (self.db.query(R.WsiMetadata)
                .filter(R.WsiMetadata.prototype_id == prototype_id,
                        R.WsiMetadata.is_representative.is_(True))
                .first())

    def member_concept_matrix(self) -> np.ndarray:
        """전체 멤버 concept 벡터 (OOD 분포 적합용). (n_members, DIM)."""
        rows = self.db.query(R.PrototypeMember.concept_vec).all()
        return np.asarray([r[0] for r in rows], dtype=float) if rows else np.empty((0, R.VEC_DIM))
