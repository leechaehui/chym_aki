"""Alert 레포지토리 — dedup 조회 + 부서/상태 필터 + 우선순위 정렬."""
from __future__ import annotations

from sqlalchemy import select

from core.query_optimizer import Page
from models.alert import Alert
from repositories.base import BaseRepository


class AlertRepository(BaseRepository[Alert]):
    model = Alert

    def find_active_by_dedup(self, dedup_key: str) -> Alert | None:
        """동일 dedup_key 의 미해소(active/viewed) 알림 — 중복 억제 판단용."""
        stmt = (
            select(Alert)
            .where(Alert.dedup_key == dedup_key, Alert.status.in_(("active", "viewed")))
            .limit(1)
        )
        return self.db.execute(stmt).scalars().first()

    def list_filtered(
        self,
        page: Page,
        *,
        department: str | None = None,
        status: str | None = None,
    ) -> list[Alert]:
        """부서/상태 필터 + 우선순위 desc, 최신순 — 큐 표시용."""
        stmt = select(Alert)
        if department:
            stmt = stmt.where(Alert.department == department)
        if status:
            stmt = stmt.where(Alert.status == status)
        stmt = (
            stmt.order_by(Alert.priority.desc(), Alert.created_at.desc())
            .limit(page.limit)
            .offset(page.offset)
        )
        return self._all(stmt)
