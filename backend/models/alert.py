"""Alert 엔티티 (지시서 §8 alerts 테이블).

책임: AKI 파이프라인이 생성한 임상 알림의 영속 + 추적성.
- dedup_key 로 중복 알림을 억제(Alert 서비스가 사용).
- source_event_id 로 어떤 AKI_EVENT 에서 비롯됐는지 추적(traceability).
- status 는 audit 액션(VIEWED/DISMISSED/ACKNOWLEDGED/ESCALATED)으로 갱신된다.
"""
from sqlalchemy import Float, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from core.database import Base
from models.base import TimestampMixin


class Alert(Base, TimestampMixin):
    __tablename__ = "alerts"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    patient_id: Mapped[str] = mapped_column(String(40), nullable=False)
    # AKI_CONFIRMED | AKI_SUSPECTED | PRE_AKI | SYSTEM
    type: Mapped[str] = mapped_column(String(40), nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)  # info|warning|critical
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    aki_stage: Mapped[str | None] = mapped_column(String(40), nullable=True)
    aki_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    subject_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dedup_key: Mapped[str] = mapped_column(String(120), nullable=False)
    title: Mapped[str] = mapped_column(String(160), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    source_event_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    department: Mapped[str] = mapped_column(String(20), nullable=False)
    # active | viewed | dismissed | acknowledged | escalated
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active")

    __table_args__ = (
        # 활성 dedup 조회 + 부서/상태 필터 + 우선순위 정렬 최적화.
        Index("ix_alert_dedup", "dedup_key", "status"),
        Index("ix_alert_dept_status", "department", "status"),
    )
