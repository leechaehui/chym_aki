"""감사 로그 레포지토리 — append-only 조회/적재."""
from sqlalchemy import select

from core.query_optimizer import Page, paginate
from models.audit import AuditLog
from repositories.base import BaseRepository


class AuditRepository(BaseRepository[AuditLog]):
    model = AuditLog

    def list_recent(
        self, page: Page, target_type: str | None = None
    ) -> list[AuditLog]:
        """최근 감사 로그 — 생성 역순 + 페이지네이션, target_type 옵션 필터."""
        stmt = select(AuditLog)
        if target_type:
            stmt = stmt.where(AuditLog.target_type == target_type)
        stmt = paginate(stmt.order_by(AuditLog.created_at.desc()), page)
        return list(self.db.execute(stmt).scalars().all())
