"""감사 로그 DTO."""
from schemas.base import CamelModel


class AuditLogOut(CamelModel):
    id: str
    user_id: str | None = None
    action: str
    target_type: str
    target_id: str | None = None
    payload: dict | None = None
    created_at: str | None = None
