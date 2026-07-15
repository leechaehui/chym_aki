"""부가 소비자 — EventLog writer + Audit consumer (지시서 §6, §8).

- EventLog writer: 모든 토픽(LAB/AKI/ALERT/AUDIT) 구독 → event_log 적재(full trace, immutable).
- Audit consumer: AUDIT_EVENT 구독 → 기존 audit_logs 재사용(target_type='alert').
각 핸들러는 자체 SessionLocal 단명 세션으로 commit 한다.
"""
from __future__ import annotations

import json

from core.database import SessionLocal
from core.event_bus import AUDIT_EVENT
from core.logging import get_logger
from models.base import new_id
from models.event_log import EventLog
from services.audit_service import AuditService

log = get_logger("chym.event_consumers")


def write_event_log(event: dict) -> None:
    """버스를 지나는 모든 이벤트를 event_log 에 append(불변 트레이스)."""
    db = SessionLocal()
    try:
        row = EventLog(
            id=new_id("evt"),
            event_type=event.get("eventType", "UNKNOWN"),
            patient_id=event.get("patientId") or event.get("patient_id"),
            payload=json.dumps(event, ensure_ascii=False, default=str),
        )
        db.add(row)
        db.commit()
    except Exception:
        db.rollback()
        log.exception("event_log write failed")
    finally:
        db.close()


def process_audit_event(event: dict) -> None:
    """AUDIT_EVENT → audit_logs 기록(VIEWED/DISMISSED/ACKNOWLEDGED/ESCALATED)."""
    db = SessionLocal()
    try:
        AuditService(db).record(
            user_id=event.get("userId"),
            action=f"alert_{(event.get('action') or '').lower()}",
            target_type="alert",
            target_id=event.get("alertId"),
            payload={
                "role": event.get("role"),
                "patientId": event.get("patientId"),
                "action": event.get("action"),
            },
        )
        db.commit()
    except Exception:
        db.rollback()
        log.exception("audit event write failed")
    finally:
        db.close()
