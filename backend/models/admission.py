"""입원 엔티티 (병상 트랜잭션의 핵심 기록).

책임: 환자-병상 점유의 시작/종료 이력을 보관.
- status = active 인 행은 환자당 최대 1건(중복 입원 방지 규칙의 근거).
- bed 상태 변경과 admission 생성/종료는 동일 트랜잭션에서 수행된다(service).
"""
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from core.database import Base
from models.base import utcnow


class Admission(Base):
    __tablename__ = "admissions"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    patient_id: Mapped[str] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"), nullable=False
    )
    bed_id: Mapped[str] = mapped_column(
        ForeignKey("beds.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="active"
    )  # active | discharged
    admitted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    discharged_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # 중복 입원 체크(환자+status)와 병상별 활성 입원 조회 최적화.
    __table_args__ = (
        Index("ix_admissions_patient_status", "patient_id", "status"),
        Index("ix_admissions_bed_status", "bed_id", "status"),
    )
