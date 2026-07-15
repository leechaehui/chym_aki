"""알림 레포지토리."""
from sqlalchemy import or_, select

from core.query_optimizer import Page, paginate
from models.notification import Notification
from repositories.base import BaseRepository


class NotificationRepository(BaseRepository[Notification]):
    model = Notification

    def list_for_department(
        self, department: str, page: Page, user_id: str | None = None
    ) -> list[Notification]:
        """부서별 알림 — 최신순 + 페이지네이션(드로어 히스토리).

        target_user_id 가 지정된 알림은 그 계정에만 보인다(정밀 타겟팅). NULL 이면
        부서 전체 공개(기존 동작). user_id 를 주면 타겟 필터를 적용한다."""
        stmt = select(Notification).where(Notification.department == department)
        if user_id is not None:
            stmt = stmt.where(
                or_(
                    Notification.target_user_id.is_(None),
                    Notification.target_user_id == user_id,
                )
            )
        stmt = paginate(stmt.order_by(Notification.created_at.desc()), page)
        return list(self.db.execute(stmt).scalars().all())
