"""모델 공통 믹스인.

책임: 모든 엔티티가 공유하는 컬럼 패턴(타임스탬프)과 id 생성 헬퍼.
도메인 의미는 각 모델 파일이 갖는다.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, func
from sqlalchemy.orm import Mapped, mapped_column


def new_id(prefix: str) -> str:
    """접두사 기반 식별자 생성(예: gen_id('t') -> 't-3f9a...').

    프론트엔드가 사용하는 prefix-id 컨벤션과 호환되도록 문자열 PK 를 쓴다.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TimestampMixin:
    """생성 시각 컬럼. 감사/정렬 기준으로 폭넓게 사용된다."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
