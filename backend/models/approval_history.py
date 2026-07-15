"""가입 승인/거부 이력 엔티티.

책임: 관리자(또는 시스템)의 사용자 승인 상태 변경 이력을 영구 보관한다.
Append-Only 원칙을 따른다 (UPDATE, DELETE 금지).
"""
from datetime import datetime

from sqlalchemy import Integer, DateTime, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from core.database import Base


class ApprovalHistory(Base):
    __tablename__ = "approval_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(40), nullable=False)
    admin_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    actor_type: Mapped[str] = mapped_column(String(20), nullable=False)
    old_status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    new_status: Mapped[str] = mapped_column(String(20), nullable=False)
    reason: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    __table_args__ = (
        Index("idx_approval_history_user_id", "user_id"),
        Index("idx_approval_history_created_at", "created_at"),
    )
