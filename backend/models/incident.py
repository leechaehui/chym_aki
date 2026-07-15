"""운영 장애(Incident) 엔티티.

책임: 자동 감지된 장애와 관리자 조치 내역, 전자서명 및 감사 정보를 영속화.
- 동일 원인의 장애는 중복 생성되지 않고 occurrence_count 만 증가한다.
"""
from datetime import datetime

from sqlalchemy import DateTime, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from core.database import Base
from models.base import TimestampMixin, func


class Incident(Base, TimestampMixin):
    __tablename__ = "incidents"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    incident_no: Mapped[str] = mapped_column(String(40), unique=True, nullable=False, index=True)
    
    # OPEN, INVESTIGATING, RESOLVED, SIGNED, LOCKED
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="OPEN", index=True)
    # CRITICAL, HIGH, MEDIUM, LOW
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    
    affected_service: Mapped[str | None] = mapped_column(String(100), nullable=True)
    module_name: Mapped[str] = mapped_column(String(100), nullable=False)
    endpoint: Mapped[str | None] = mapped_column(String(255), nullable=True)
    
    error_message: Mapped[str] = mapped_column(Text, nullable=False)
    stack_trace: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    occurrence_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    
    first_occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=func.now()
    )
    last_occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=func.now()
    )
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # 조치 내용
    root_cause: Mapped[str | None] = mapped_column(Text, nullable=True)
    action_taken: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # 배정 담당자 (현재 진행 담당자)
    assigned_to_user_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    assigned_to_name: Mapped[str | None] = mapped_column(String(60), nullable=True)
    
    # 조치 완료자 (RESOLVED 처리자)
    resolved_by_user_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    resolved_by_name: Mapped[str | None] = mapped_column(String(60), nullable=True)
    
    # 서명 및 승인
    signature_path: Mapped[str | None] = mapped_column(String(255), nullable=True)
    signed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # 수정 시각 트래킹용
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_incidents_dedupe", "severity", "module_name", "endpoint"),
    )
