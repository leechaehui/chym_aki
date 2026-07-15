"""알림 서비스 — 부서별 알림 히스토리 조회 + 읽음 처리 + 생성.

실시간 채널 라우팅(Toast/Banner/Drawer/Modal)은 프론트(severity 기반)가 담당하고,
여기서는 영속 히스토리와 읽음 상태만 관리한다.
"""
from sqlalchemy.orm import Session

from core.exceptions import NotFoundError
from core.query_optimizer import Page
from models.base import new_id
from models.notification import Notification
from repositories.notification_repository import NotificationRepository
from schemas.notification import NotificationOut


def _to_out(n: Notification) -> NotificationOut:
    return NotificationOut(
        id=n.id,
        title=n.title,
        message=n.message,
        severity=n.severity,
        department=n.department,
        created_at=n.created_at.isoformat() if n.created_at else "",
        read=n.read,
        link=n.link,
        tone=n.tone,
    )


class NotificationService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = NotificationRepository(db)

    def list_for_department(
        self, department: str, page: Page, user_id: str | None = None
    ) -> list[NotificationOut]:
        return [_to_out(n) for n in self.repo.list_for_department(department, page, user_id)]

    def mark_read(self, notification_id: str) -> NotificationOut:
        n = self.repo.get(notification_id)
        if not n:
            raise NotFoundError("알림을 찾을 수 없습니다.")
        n.read = True
        self.db.commit()
        return _to_out(n)

    def create(self, data: dict) -> NotificationOut:
        n = Notification(
            id=new_id("noti"),
            department=data["department"],
            severity=data["severity"],
            title=data["title"],
            message=data["message"],
            link=data.get("link"),
            tone=data.get("tone"),
            read=data.get("read", False),
        )
        self.repo.add(n)
        self.db.commit()
        return _to_out(n)
