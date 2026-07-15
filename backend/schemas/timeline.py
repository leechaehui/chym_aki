"""타임라인 DTO — EMR 공통 이벤트.

EMERGENCY 가 이벤트를 생성(쓰기)하고, 모든 도메인이 조회한다.
"""
from schemas.base import CamelModel


class TimelineEventOut(CamelModel):
    id: str
    patient_id: str
    event_type: str
    severity: str
    title: str
    description: str | None = None
    source: str
    actor: str | None = None
    event_time: str
    payload: dict | None = None


class TimelineEventCreate(CamelModel):
    """이벤트 생성 입력 — 보통 다른 서비스가 내부 호출하지만 API 로도 노출(응급)."""

    patient_id: str
    event_type: str  # BED_CHANGE|LAB_RESULT|CONSULTATION|AI_ALERT|DIAGNOSIS_UPDATE
    title: str
    severity: str = "INFO"
    description: str | None = None
    source: str = "MANUAL"
    actor: str | None = None
    payload: dict | None = None
