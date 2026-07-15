"""사용자/계정 엔티티.

책임: 인증 주체의 영속 표현. 비밀번호는 해시만 저장한다(평문 금지).
RBAC 의 role, 가입 승인 흐름의 approval 을 포함한다.
"""
from datetime import datetime

from sqlalchemy import DateTime, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from core.database import Base
from models.base import TimestampMixin


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    # 로그인 식별자 — 유니크 + 인덱스(로그인 조회가 가장 빈번한 경로).
    username: Mapped[str] = mapped_column(String(60), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(120), nullable=False)
    name: Mapped[str] = mapped_column(String(60), nullable=False)
    # admin | emergency | nephrology | pathology
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    department: Mapped[str] = mapped_column(String(60), nullable=False)
    # PACS 연동용 병원 사번 (hospital_employees.employee_id)
    employee_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    # pending | approved | rejected
    approval: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    last_login_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    signature_path: Mapped[str | None] = mapped_column(String(255), nullable=True)
    
    # 가입 거부 사유 관리 추가 필드
    rejection_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    approved_by: Mapped[str | None] = mapped_column(String(40), nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    rejected_by: Mapped[str | None] = mapped_column(String(40), nullable=True)
    rejected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # 승인 상태별 목록(관리자 화면)과 role 필터 최적화를 위한 복합 인덱스.
    __table_args__ = (Index("ix_users_approval_role", "approval", "role"),)
