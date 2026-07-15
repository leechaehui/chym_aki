"""EventLog 엔티티 (지시서 §8 event_log — full trace).

책임: 버스를 지나는 모든 이벤트(LAB/AKI/ALERT/AUDIT)의 불변 기록.
- append-only. 수정/삭제 없음(지시서 §12 Immutable logs).
- payload 는 JSON 문자열(Text)로 저장 — SQLite/PG 동일 동작(기존 audit.payload 패턴).
"""
from sqlalchemy import Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from core.database import Base
from models.base import TimestampMixin


class EventLog(Base, TimestampMixin):
    __tablename__ = "event_log"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    event_type: Mapped[str] = mapped_column(String(20), nullable=False)
    patient_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    payload: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_eventlog_patient", "patient_id"),
        Index("ix_eventlog_type", "event_type"),
    )
