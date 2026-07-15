"""병리 분석 결과 엔티티 (병리과 도메인).

책임: 협진 건에 연결된 WSI 정량 분석 결과/보고서 보관.
- layers/metrics 는 가변 길이 구조 → JSON(Text)로 보관(단순 값 객체, 별도 정규화 이득 없음).
- report 는 1:1 값 객체 → 평면 컬럼.
"""
from sqlalchemy import Boolean, String, Text  # noqa: F401  (Boolean reserved for future)
from sqlalchemy.orm import Mapped, mapped_column

from core.database import Base


class PathologyResult(Base):
    __tablename__ = "pathology_results"

    # 협진 1건당 결과 1건 → consult_id 를 PK 로.
    consult_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    stain: Mapped[str] = mapped_column(String(20), nullable=False)  # PAS|Silver|Masson|HE
    image_url: Mapped[str | None] = mapped_column(String(300), nullable=True)
    layers_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    metrics_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    report_findings: Mapped[str] = mapped_column(Text, nullable=False, default="")
    report_diagnosis: Mapped[str] = mapped_column(Text, nullable=False, default="")
    report_status: Mapped[str] = mapped_column(String(10), nullable=False, default="draft")
    report_updated_at: Mapped[str | None] = mapped_column(String(40), nullable=True)
