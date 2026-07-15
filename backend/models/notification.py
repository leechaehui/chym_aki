"""알림 엔티티 (부서별 영속 알림 히스토리).

책임: 드로어 히스토리로 표시되는 부서 알림의 영속 + 읽음 상태.
- 실시간 표현 채널(Toast/Banner/Drawer/Modal) 라우팅은 프론트 책임(severity 기반).
- INFO(휘발성 토스트)는 보통 영속하지 않는다(시드에서 제외).
"""
from sqlalchemy import Boolean, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from core.database import Base
from models.base import TimestampMixin


class Notification(Base, TimestampMixin):
    __tablename__ = "notifications"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    # 대상 부서(= role): admin|emergency|nephrology|pathology
    department: Mapped[str] = mapped_column(String(20), nullable=False)
    # 특정 사용자 타겟(있으면 그 계정에만 보임/푸시). NULL 이면 부서 전체(기존 동작).
    target_user_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    # INFO|WARNING|ACTION_REQUIRED|CRITICAL
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    title: Mapped[str] = mapped_column(String(120), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    link: Mapped[str | None] = mapped_column(String(200), nullable=True)
    tone: Mapped[str | None] = mapped_column(String(20), nullable=True)
    read: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
