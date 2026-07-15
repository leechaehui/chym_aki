from datetime import datetime
from pydantic import BaseModel, Field


class IncidentBase(BaseModel):
    incident_no: str
    status: str
    severity: str
    affected_service: str | None
    module_name: str
    endpoint: str | None
    error_message: str
    occurrence_count: int
    first_occurred_at: datetime
    last_occurred_at: datetime
    resolved_at: datetime | None
    root_cause: str | None
    action_taken: str | None
    assigned_to_user_id: str | None
    assigned_to_name: str | None
    resolved_by_user_id: str | None
    resolved_by_name: str | None
    signature_path: str | None
    signed_at: datetime | None
    locked_at: datetime | None
    created_at: datetime
    updated_at: datetime


class IncidentOut(IncidentBase):
    id: str
    stack_trace: str | None


class IncidentStatusUpdate(BaseModel):
    status: str = Field(..., description="OPEN, INVESTIGATING, RESOLVED, SIGNED, LOCKED 중 하나")


class IncidentResolveUpdate(BaseModel):
    root_cause: str
    action_taken: str


class IncidentSignUpdate(BaseModel):
    # 이제 Base64를 보내지 않아도 됩니다 (프로필에 저장된 서명을 사용하기 때문).
    # 하지만 클라이언트 하위 호환성을 위해 유지하거나 빈 필드로 둘 수 있습니다.
    pass


class IncidentAssignUpdate(BaseModel):
    assigned_to_user_id: str | None = None
    assigned_to_name: str | None = None

class IncidentAnalysisOut(BaseModel):
    root_cause: str
    action_taken: str
