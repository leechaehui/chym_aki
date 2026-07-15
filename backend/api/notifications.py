"""알림 API — 부서별 히스토리 조회 + 읽음 처리 + 생성."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from core.deps import get_current_user, get_db
from core.query_optimizer import Page
from models.user import User
from schemas.notification import NotificationCreate, NotificationOut
from services.notification_service import NotificationService

router = APIRouter(prefix="/notifications", tags=["notification"])


@router.get("", response_model=list[NotificationOut])
def list_notifications(
    department: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """알림 히스토리. department 미지정 시 현재 사용자 부서(role) 기준.

    현재 사용자에게 정밀 타겟팅된 알림(target_user_id)만 필터해서 반환한다."""
    dept = department or current_user.role
    return NotificationService(db).list_for_department(
        dept, Page.of(limit, offset), user_id=current_user.id
    )


@router.post("/{notification_id}/read", response_model=NotificationOut)
def mark_read(
    notification_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    return NotificationService(db).mark_read(notification_id)


@router.post("", response_model=NotificationOut, status_code=201)
def create_notification(
    body: NotificationCreate,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    return NotificationService(db).create(body.model_dump())
