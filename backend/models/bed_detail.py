"""병상 입실 상세 / 예약 엔티티 (응급의학과 도메인).

책임: 병상 클릭 시 표시되는 입실 환자 상세(처방·검사·AKI 처치 권고)와
예약 병상의 예약 정보 보관. 처방/검사/항목은 JSON(Text)로 보관한다.
명시 상세가 없는 사용중 병상은 서비스가 기본 상세로 보강한다(프론트 동작과 동일).
"""
from sqlalchemy import Boolean, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from core.database import Base


class BedDetail(Base):
    __tablename__ = "bed_details"

    bed_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    diagnosis: Mapped[str] = mapped_column(String(200), nullable=False)
    attending: Mapped[str] = mapped_column(String(60), nullable=False)
    admitted_at: Mapped[str] = mapped_column(String(40), nullable=False)
    aki_risk: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    aki_stage: Mapped[str | None] = mapped_column(String(60), nullable=True)
    medications_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    labs_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    recent_inputs_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    treatment_report: Mapped[str | None] = mapped_column(Text, nullable=True)
    treatment_items_json: Mapped[str | None] = mapped_column(Text, nullable=True)


