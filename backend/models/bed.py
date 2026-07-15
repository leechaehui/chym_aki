"""병상 엔티티 (응급의학과 도메인).

책임: 병상의 현재 상태/점유 환자만 표현한다.
입원(admission) 생성·종료는 별도 엔티티/서비스가 담당한다(관심사 분리).
"""
from sqlalchemy import ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from core.database import Base


class Bed(Base):
    __tablename__ = "beds"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    zone: Mapped[str] = mapped_column(String(20), nullable=False)  # er|icu|ward|isolation
    label: Mapped[str] = mapped_column(String(20), nullable=False)  # 예: ER-01
    # available | occupied | cleaning | reserved
    state: Mapped[str] = mapped_column(String(20), nullable=False, default="available")
    patient_id: Mapped[str | None] = mapped_column(
        ForeignKey("patients.id", ondelete="SET NULL"), nullable=True
    )
    # 보드 표시용 비정규화 이름(빠른 목록 렌더 — 조인 회피).
    patient_name: Mapped[str | None] = mapped_column(String(60), nullable=True)

    # 구역별 보드 조회(가장 빈번)와 상태 필터 최적화.
    __table_args__ = (Index("ix_beds_zone_state", "zone", "state"),)
