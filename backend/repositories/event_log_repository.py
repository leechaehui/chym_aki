"""EventLog 레포지토리 — append-only 트레이스 조회."""
from __future__ import annotations

from sqlalchemy import select

from core.query_optimizer import Page
from models.event_log import EventLog
from repositories.base import BaseRepository


class EventLogRepository(BaseRepository[EventLog]):
    model = EventLog

    def list_recent(self, page: Page, *, patient_id: str | None = None) -> list[EventLog]:
        stmt = select(EventLog)
        if patient_id:
            stmt = stmt.where(EventLog.patient_id == patient_id)
        stmt = stmt.order_by(EventLog.created_at.desc()).limit(page.limit).offset(page.offset)
        return self._all(stmt)
