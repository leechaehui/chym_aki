"""ORM → DTO 변환 헬퍼.

대부분의 DTO 는 from_attributes 로 자동 변환되지만,
JSON 문자열 컬럼(payload/reply)·datetime → ISO 문자열 변환이 필요한 경우만 여기서 처리한다.
API 레이어를 얇게 유지하기 위한 순수 변환 함수 모음.
"""
import json

from models.audit import AuditLog
from models.consultation import Consultation
from models.timeline import TimelineEvent
from schemas.audit import AuditLogOut
from schemas.consultation import ConsultEventOut, ConsultOut, ConsultReplyOut
from schemas.timeline import TimelineEventOut


def consult_to_out(consult: Consultation, *, patient_name: str | None = None) -> ConsultOut:
    reply = None
    if consult.reply_json:
        raw = json.loads(consult.reply_json)
        reply = ConsultReplyOut(
            findings=raw.get("findings", ""),
            diagnosis=raw.get("diagnosis", ""),
            recommendation=raw.get("recommendation", ""),
            author=raw.get("author", ""),
            replied_at=raw.get("repliedAt", ""),
            signature_path=raw.get("signature_path") or raw.get("signaturePath"),
            analysis=raw.get("analysis"),
        )
    return ConsultOut(
        id=consult.id,
        kind=consult.kind,
        patient_mrn=consult.patient_mrn,
        # 명단에 있는 환자면 현재 표시명으로 해소(대시보드/모니터링과 일치), 없으면 저장값.
        patient_name=patient_name or consult.patient_name,
        diagnosis=consult.diagnosis,
        key_labs=consult.key_labs,
        reason=consult.reason,
        urgency=consult.urgency,
        status=consult.status,
        requested_by=consult.requested_by,
        requested_at=consult.requested_at.isoformat() if consult.requested_at else "",
        bed_label=consult.bed_label,
        timeline=[
            ConsultEventOut(
                id=e.id, stage=e.stage, label=e.label, at=e.at, actor=e.actor
            )
            for e in consult.timeline
        ],
        reply=reply,
    )


def timeline_to_out(event: TimelineEvent) -> TimelineEventOut:
    return TimelineEventOut(
        id=event.id,
        patient_id=event.patient_id,
        event_type=event.event_type,
        severity=event.severity,
        title=event.title,
        description=event.description,
        source=event.source,
        actor=event.actor,
        event_time=event.event_time.isoformat() if event.event_time else "",
        payload=json.loads(event.payload_json) if event.payload_json else None,
    )


def audit_to_out(log: AuditLog) -> AuditLogOut:
    return AuditLogOut(
        id=log.id,
        user_id=log.user_id,
        action=log.action,
        target_type=log.target_type,
        target_id=log.target_id,
        payload=json.loads(log.payload) if log.payload else None,
        created_at=log.created_at.isoformat() if log.created_at else None,
    )
