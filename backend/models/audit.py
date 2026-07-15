"""감사 로그 엔티티.

책임: 모든 주요 상태변경 이벤트의 불변 기록.
- 누가(user_id) 무엇을(action) 어떤 대상에(target_type/target_id) 했는지 + payload.
- 쓰기 전용 성격(수정/삭제 없음). 서비스 트랜잭션 내부에서 기록된다.
"""
from sqlalchemy import Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from core.database import Base
from models.base import TimestampMixin


class AuditLog(Base, TimestampMixin):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    user_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    action: Mapped[str] = mapped_column(String(60), nullable=False)
    target_type: Mapped[str] = mapped_column(String(40), nullable=False)
    target_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    payload: Mapped[str | None] = mapped_column(Text, nullable=True)

    # 대상별 감사 추적과 사용자별 활동 조회 최적화.
    __table_args__ = (
        Index("ix_audit_target", "target_type", "target_id"),
        Index("ix_audit_user", "user_id"),
    )
