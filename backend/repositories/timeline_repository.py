"""타임라인 레포지토리 — event_time DESC 정렬 + severity 필터(인덱스 기반)."""
from sqlalchemy import select

from core.query_optimizer import Page, paginate
from models.timeline import TimelineEvent
from repositories.base import BaseRepository


class TimelineRepository(BaseRepository[TimelineEvent]):
    model = TimelineEvent

    def list_for_patient(
        self,
        patient_id: str,
        page: Page,
        severity: str | None = None,
        event_type: str | None = None,
    ) -> list[TimelineEvent]:
        """환자 타임라인 — patient_id + event_time DESC, severity/type 옵션 필터.

        (ix_timeline_patient_time / ix_timeline_patient_severity 활용)
        """
        stmt = select(TimelineEvent).where(TimelineEvent.patient_id == patient_id)
        if severity:
            stmt = stmt.where(TimelineEvent.severity == severity)
        if event_type:
            stmt = stmt.where(TimelineEvent.event_type == event_type)
        stmt = paginate(stmt.order_by(TimelineEvent.event_time.desc()), page)
        return list(self.db.execute(stmt).scalars().all())
