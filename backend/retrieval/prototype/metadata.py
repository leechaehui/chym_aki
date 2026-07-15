"""Prototype Metadata 인코더 — 프로토타입의 concept-space 메타 벡터 생성.

핵심(사용자 정정안): 2차 유사도 대상은 WSI 임베딩이 아니라 **Prototype Metadata 벡터**.
프로토타입 메타 벡터 = 그 프로토타입에 속한 KPMP 환자들의 ConceptVector 평균.
→ "왜 이 프로토타입이 선택됐는가"를 임상 concept 으로 설명 가능.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..concept.schema import DIM, ETIOLOGY_ORDER, ConceptVector


@dataclass(frozen=True)
class PrototypeMetadata:
    """프로토타입 1개의 집계 메타. metadata_vec 는 pgvector 에 저장될 벡터."""

    metadata_vec: np.ndarray          # (DIM,) 멤버 concept 평균(가중 적용)
    kdigo_band: str                   # 1차 규칙 필터용(다수결)
    etiology_hint: str                # 1차 규칙 필터용(다수결)
    egfr_mean: float | None
    n_members: int


def _mode_or_unknown(values: list[str]) -> str:
    vals = [v for v in values if v and v != "unknown"]
    if not vals:
        return "unknown"
    return max(set(vals), key=vals.count)


def build_prototype_metadata(members: list[ConceptVector]) -> PrototypeMetadata:
    """멤버 ConceptVector 들 → PrototypeMetadata.

    metadata_vec: 가중 concept 벡터의 평균(코사인 검색 대상).
    kdigo_band·etiology_hint: 1차 규칙 prefilter 용 다수결.
    """
    if not members:
        raise ValueError("prototype 에 멤버가 없습니다")

    mat = np.stack([m.to_vector(weighted=True) for m in members])   # (n, DIM)
    meta_vec = mat.mean(axis=0).astype(np.float32)
    assert meta_vec.shape == (DIM,)

    band = _mode_or_unknown([m.kdigo_band() for m in members])
    etio = _mode_or_unknown([m.etiology_hint for m in members])
    egfrs = [m.egfr for m in members if m.egfr is not None]
    egfr_mean = round(float(np.mean(egfrs)), 1) if egfrs else None

    return PrototypeMetadata(
        metadata_vec=meta_vec,
        kdigo_band=band,
        etiology_hint=etio,
        egfr_mean=egfr_mean,
        n_members=len(members),
    )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """concept 벡터 코사인 유사도(검색 순위용). pgvector <=> 와 동치 검증용."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
