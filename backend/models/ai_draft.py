"""AI 진료 초안 노트 엔티티 (신장내과 음성 기반 분석).

책임: 음성/대화 전사(transcript) → NLP 증상추출 → SOAP 초안 생성 결과의 영속화.
- symptoms_json: NLP 가 추출한 구조화 증상.
- draft_text  : SOAP 4섹션 JSON 직렬화 결과.
생성은 트랜잭션으로 수행되고 audit log 가 남는다(service).
"""
from sqlalchemy import ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from core.database import Base
from models.base import TimestampMixin


class AiDraftNote(Base, TimestampMixin):
    __tablename__ = "ai_draft_notes"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    patient_id: Mapped[str] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"), nullable=False
    )
    transcript: Mapped[str] = mapped_column(Text, nullable=False)
    symptoms_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    draft_text: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="draft"
    )  # draft | finalized

    # 환자별 초안 이력(최신순) 조회 최적화.
    __table_args__ = (Index("ix_ai_draft_patient", "patient_id"),)
