"""타임라인 API (EMR 공통 이벤트 레이어).

- 조회: 모든 인증 사용자(응급/신장/병리/관리자).
- 생성(쓰기): 관리자만.
  신장내과는 읽기 전용이므로 쓰기 권한에서 제외된다(RBAC).
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.serializers import timeline_to_out
from core.deps import get_current_user, get_db, require_roles
from core.query_optimizer import Page
from models.user import User
from schemas.timeline import TimelineEventCreate, TimelineEventOut
from services.timeline_service import TimelineService

router = APIRouter(prefix="/timeline", tags=["timeline"])


@router.get("/patient/{patient_id}", response_model=list[TimelineEventOut])
def get_patient_timeline(
    patient_id: str,
    severity: str | None = None,
    event_type: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """환자 타임라인 조회 — event_time DESC, severity/type 필터."""
    events = TimelineService(db).list_for_patient(
        patient_id, Page.of(limit, offset), severity, event_type
    )
    return [timeline_to_out(e) for e in events]


@router.post("/event", response_model=TimelineEventOut, status_code=201)
def create_event(
    body: TimelineEventCreate,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("admin")),
):
    """수동 이벤트 생성(관리자). 트랜잭션 커밋은 여기서 수행."""
    service = TimelineService(db)
    event = service.add_event(
        patient_id=body.patient_id,
        event_type=body.event_type,
        title=body.title,
        severity=body.severity,
        description=body.description,
        source=body.source,
        actor=body.actor or user.name,
        payload=body.payload,
    )
    db.commit()
    db.refresh(event)
    return timeline_to_out(event)
