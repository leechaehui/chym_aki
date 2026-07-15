"""환자 타임라인 이벤트 엔티티 (EMR 공통 이벤트 레이어).

핵심 구조: TIMELINE 은 독립 CRUD 도메인이 아니라 EMR 공통 시스템이다.
모든 도메인 이벤트가 이 테이블로 흘러 들어온다.
  BED_CHANGE / LAB_RESULT / CONSULTATION / AI_ALERT / DIAGNOSIS_UPDATE
- EMERGENCY: 생성(쓰기) + 조회.
- NEPHROLOGY: 읽기 전용(쓰기 금지) — 서비스 레이어에서 강제.
정렬은 event_time DESC, 필터는 severity 기준.
"""
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from core.database import Base
from models.base import TimestampMixin, utcnow


class TimelineEvent(Base, TimestampMixin):
    __tablename__ = "timeline_events"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    patient_id: Mapped[str] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"), nullable=False
    )
    # BED_CHANGE | LAB_RESULT | CONSULTATION | AI_ALERT | DIAGNOSIS_UPDATE
    event_type: Mapped[str] = mapped_column(String(30), nullable=False)
    # INFO | WARNING | ACTION_REQUIRED | CRITICAL
    severity: Mapped[str] = mapped_column(String(20), nullable=False, default="INFO")
    title: Mapped[str] = mapped_column(String(120), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 이벤트 출처 도메인(BED_SYSTEM / LAB_SYSTEM / CONSULTATION / AI_SYSTEM / DIAGNOSIS)
    source: Mapped[str] = mapped_column(String(30), nullable=False)
    actor: Mapped[str | None] = mapped_column(String(80), nullable=True)
    event_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    payload_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # 환자 타임라인 조회: patient_id + event_time DESC 가 지배적 패턴.
    # severity 필터를 함께 태워 인덱스 커버리지 확보.
    __table_args__ = (
        Index("ix_timeline_patient_time", "patient_id", "event_time"),
        Index("ix_timeline_patient_severity", "patient_id", "severity"),
    )
