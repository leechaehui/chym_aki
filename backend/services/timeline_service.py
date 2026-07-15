"""타임라인 서비스 (EMR 공통 이벤트 레이어).

핵심 규칙
- 모든 EMR 이벤트(BED/LAB/CONSULT/AI/DIAGNOSIS)는 이 서비스를 통해 타임라인에 적재된다.
- add_event() 는 commit 하지 않는다 → 이벤트를 발생시킨 도메인 트랜잭션에 포함된다
  (예: 병상 배정 성공과 BED_CHANGE 이벤트는 원자적으로 함께 커밋).
- 조회는 event_time DESC, severity 필터 지원.

읽기 전용 소비자(신장내과)는 API RBAC 에서 쓰기 권한을 차단한다(여기선 능력만 제공).
"""
import json

from sqlalchemy.orm import Session

from core.query_optimizer import Page
from models.base import new_id, utcnow
from models.timeline import TimelineEvent
from repositories.timeline_repository import TimelineRepository

# 출처 도메인 → 기본 event_type 매핑(이벤트 소스 통합 규약).
SOURCE_EVENT_TYPE = {
    "BED_SYSTEM": "BED_CHANGE",
    "LAB_SYSTEM": "LAB_RESULT",
    "CONSULTATION": "CONSULTATION",
    "AI_SYSTEM": "AI_ALERT",
    "DIAGNOSIS": "DIAGNOSIS_UPDATE",
}


class TimelineService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = TimelineRepository(db)

    def add_event(
        self,
        *,
        patient_id: str,
        event_type: str,
        title: str,
        severity: str = "INFO",
        description: str | None = None,
        source: str = "MANUAL",
        actor: str | None = None,
        payload: dict | None = None,
    ) -> TimelineEvent:
        """타임라인 이벤트 적재(flush only). 도메인 트랜잭션 안에서 호출된다."""
        event = TimelineEvent(
            id=new_id("tl"),
            patient_id=patient_id,
            event_type=event_type,
            severity=severity,
            title=title,
            description=description,
            source=source,
            actor=actor,
            event_time=utcnow(),
            payload_json=json.dumps(payload, ensure_ascii=False) if payload else None,
        )
        return self.repo.add(event)

    def list_for_patient(
        self,
        patient_id: str,
        page: Page,
        severity: str | None = None,
        event_type: str | None = None,
    ) -> list[TimelineEvent]:
        """환자 타임라인 조회(읽기) — event_time DESC."""
        return self.repo.list_for_patient(patient_id, page, severity, event_type)
