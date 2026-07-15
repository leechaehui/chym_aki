"""알림 DTO — 프론트 AppNotification/NotificationInput 대응."""
from schemas.base import CamelModel


class NotificationOut(CamelModel):
    id: str
    title: str
    message: str
    severity: str
    department: str
    created_at: str
    read: bool
    link: str | None = None
    tone: str | None = None


class NotificationCreate(CamelModel):
    title: str
    message: str
    severity: str
    department: str
    read: bool = False
    link: str | None = None
    tone: str | None = None
