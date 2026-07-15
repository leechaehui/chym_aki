"""Retrieval CDSS ORM — Prototype Atlas · WSI 메타 · 멤버 · 검색로그 · concept 분포.

Frozen Architecture 의 데이터 기반. PostgreSQL 전용(ARRAY/JSON) — pgvector 설치 시
metadata_vec/mean_vec 를 vector(16) 로 ALTER + HNSW 인덱스 추가만 하면 그대로 가속된다
(컬럼은 double precision[] → vector 캐스팅 호환). 현재는 Python 코사인으로 검색(수십 prototype 규모).

주의: PG 전용 타입이라 models/__init__ 전역 등록에서 제외(메인 init_db 의 sqlite 폴백 보호).
생성은 scripts/migrate_retrieval.py 가 PostgreSQL 대상으로 수행한다.
"""
from __future__ import annotations

from sqlalchemy import (
    ARRAY, JSON, BigInteger, Boolean, Float, ForeignKey, Index, Integer, String,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database import Base
from models.base import TimestampMixin, new_id

VEC_DIM = 16  # ConceptVector.DIM 와 일치


class Prototype(Base, TimestampMixin):
    """병리 reference unit. metadata_vec = 멤버 KPMP 환자 ConceptVector 평균(검색 대상)."""

    __tablename__ = "prototypes"

    id: Mapped[str] = mapped_column(String(40), primary_key=True,
                                    default=lambda: new_id("proto"))
    label: Mapped[str] = mapped_column(String(80), nullable=False)
    cluster_id: Mapped[int] = mapped_column(Integer, nullable=False)
    n_members: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # concept-space 메타 벡터(2차 cosine 대상). pgvector 설치 시 vector(16) 로 ALTER.
    metadata_vec: Mapped[list[float]] = mapped_column(ARRAY(Float), nullable=False)

    # 1차 규칙 prefilter 컬럼.
    kdigo_band: Mapped[str | None] = mapped_column(String(8), nullable=True)
    etiology_hint: Mapped[str | None] = mapped_column(String(20), nullable=True)
    egfr_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    chronicity: Mapped[float | None] = mapped_column(Float, nullable=True)

    is_rare: Mapped[bool] = mapped_column(Boolean, default=False)
    batch_adjusted: Mapped[bool] = mapped_column(Boolean, default=True)
    validated_by: Mapped[str | None] = mapped_column(String(60), nullable=True)

    members: Mapped[list["PrototypeMember"]] = relationship(
        back_populates="prototype", cascade="all, delete-orphan")
    slides: Mapped[list["WsiMetadata"]] = relationship(back_populates="prototype")

    __table_args__ = (Index("ix_proto_filter", "kdigo_band", "etiology_hint"),)


class WsiMetadata(Base):
    """대표/멤버 WSI 메타 — 프로토타입의 representative WSI 표시용."""

    __tablename__ = "wsi_metadata"

    slide_id: Mapped[str] = mapped_column(String(100), primary_key=True)  # KPMP case_code
    prototype_id: Mapped[str | None] = mapped_column(
        ForeignKey("prototypes.id", ondelete="SET NULL"), nullable=True)
    is_representative: Mapped[bool] = mapped_column(Boolean, default=False)
    stain: Mapped[str | None] = mapped_column(String(20), nullable=True)
    magnification: Mapped[str | None] = mapped_column(String(10), nullable=True)
    thumb_path: Mapped[str | None] = mapped_column(String(300), nullable=True)
    dzi_available: Mapped[bool] = mapped_column(Boolean, default=False)
    source: Mapped[str] = mapped_column(String(20), default="KPMP")

    prototype: Mapped["Prototype | None"] = relationship(back_populates="slides")
    __table_args__ = (Index("ix_wsi_proto", "prototype_id"),)


class PrototypeMember(Base):
    """프로토타입 ↔ KPMP 멤버 환자(설명가능성·medoid 추적)."""

    __tablename__ = "prototype_members"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prototype_id: Mapped[str] = mapped_column(
        ForeignKey("prototypes.id", ondelete="CASCADE"), nullable=False)
    patient_id: Mapped[str] = mapped_column(String(40), nullable=False)
    concept_vec: Mapped[list[float]] = mapped_column(ARRAY(Float), nullable=False)
    is_medoid: Mapped[bool] = mapped_column(Boolean, default=False)

    prototype: Mapped["Prototype"] = relationship(back_populates="members")
    __table_args__ = (Index("ix_member_proto", "prototype_id"),)


class RetrievalLog(Base, TimestampMixin):
    """검색 1건 audit + Evaluation/Calibration 학습 소스."""

    __tablename__ = "retrieval_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    subject_id: Mapped[int | None] = mapped_column(Integer, nullable=True)  # MIMIC, 비-FK
    concept_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    ood_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_ood: Mapped[bool] = mapped_column(Boolean, default=False)
    hits_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    raw_conf: Mapped[float | None] = mapped_column(Float, nullable=True)
    calibrated_conf: Mapped[float | None] = mapped_column(Float, nullable=True)


class ConceptDistribution(Base):
    """KPMP concept 분포(OOD Mahalanobis 기준). 단일 행(id=1)."""

    __tablename__ = "concept_distribution"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    mean_vec: Mapped[list[float]] = mapped_column(ARRAY(Float), nullable=False)
    cov_inv: Mapped[dict] = mapped_column(JSON, nullable=False)   # 16x16 역공분산
    ood_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    n_ref: Mapped[int | None] = mapped_column(Integer, nullable=True)
