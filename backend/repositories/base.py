"""레포지토리 베이스 (Repository Pattern).

책임: DB 접근의 공통 기반. 세션을 보유하고 단순 by-id 조회를 제공한다.
- 비즈니스 규칙 없음(서비스의 몫).
- 트랜잭션 commit 없음 — 트랜잭션 경계는 서비스가 관리한다.
"""
from typing import Generic, TypeVar

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.database import Base

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    model: type[ModelT]

    def __init__(self, db: Session):
        self.db = db

    def get(self, entity_id: str | int) -> ModelT | None:
        """PK 단건 조회(인덱스 기반)."""
        return self.db.get(self.model, entity_id)

    def add(self, entity: ModelT) -> ModelT:
        """세션에 추가(flush 로 PK/제약 즉시 확정 — commit 은 서비스)."""
        self.db.add(entity)
        self.db.flush()
        return entity

    def _all(self, stmt) -> list[ModelT]:
        return list(self.db.execute(stmt).scalars().all())

    def list_all(self) -> list[ModelT]:
        return self._all(select(self.model))
