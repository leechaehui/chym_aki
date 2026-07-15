"""협진 엔티티 (신장내과 ↔ 병리과, 응급/ICU → 신장내과).

책임: 협진 요청/상태전이/회신과 진행 타임라인 보관.
- timeline 은 1:N 자식(selectinload).
- reply 는 회신 1건 → JSON 컬럼으로 보관(단순 값 객체).
"""
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database import Base
from models.base import utcnow


class Consultation(Base):
    __tablename__ = "consultations"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    kind: Mapped[str] = mapped_column(String(20), nullable=False)  # pathology|nephrology
    patient_mrn: Mapped[str] = mapped_column(String(40), nullable=False)
    patient_name: Mapped[str] = mapped_column(String(60), nullable=False)
    diagnosis: Mapped[str] = mapped_column(String(200), nullable=False)
    key_labs: Mapped[str] = mapped_column(String(300), nullable=False, default="")
    reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    urgency: Mapped[str] = mapped_column(String(20), nullable=False)  # routine|urgent|emergency
    # requested | in_progress | read | replied
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="requested")
    requested_by: Mapped[str] = mapped_column(String(60), nullable=False)
    # 요청자 계정 id — 회신 완료 알림을 이 계정에만 정밀 타겟팅하기 위함(NULL 이면 부서 전체).
    requested_by_user_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    requested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    bed_label: Mapped[str | None] = mapped_column(String(20), nullable=True)
    # 회신(findings/diagnosis/recommendation/author/repliedAt) JSON.
    reply_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    timeline: Mapped[list["ConsultEvent"]] = relationship(
        back_populates="consultation",
        cascade="all, delete-orphan",
        order_by="ConsultEvent.at",
    )

    # 부서 인박스(종류+상태) 조회 최적화.
    __table_args__ = (Index("ix_consultations_kind_status", "kind", "status"),)


class ConsultEvent(Base):
    """협진 진행 단계 한 항목(요청→접수→분석→판독→회신)."""

    __tablename__ = "consult_events"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    consultation_id: Mapped[str] = mapped_column(
        ForeignKey("consultations.id", ondelete="CASCADE"), nullable=False
    )
    stage: Mapped[str] = mapped_column(String(20), nullable=False)
    label: Mapped[str] = mapped_column(String(80), nullable=False)
    at: Mapped[str] = mapped_column(String(40), nullable=False)
    actor: Mapped[str] = mapped_column(String(80), nullable=False)

    consultation: Mapped["Consultation"] = relationship(back_populates="timeline")

    __table_args__ = (Index("ix_consult_events_consultation", "consultation_id"),)
