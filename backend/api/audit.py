"""감사 로그 API (관리자 전용 조회)."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.serializers import audit_to_out
from core.deps import get_db, require_roles
from core.query_optimizer import Page
from models.user import User
from schemas.audit import AuditLogOut
from services.audit_service import AuditService

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("", response_model=list[AuditLogOut])
def list_audit_logs(
    target_type: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    _: User = Depends(require_roles("admin")),
):
    logs = AuditService(db).list_recent(Page.of(limit, offset), target_type)
    return [audit_to_out(log) for log in logs]
