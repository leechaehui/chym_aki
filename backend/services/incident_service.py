"""운영 장애(Incident) 서비스.

책임: 장애 감지(report_exception), 조회, 상태 전이, 전자서명 파일 처리 및 감사 로깅.
"""
import base64
import os
import traceback
from datetime import datetime
from collections.abc import Sequence

from fastapi import HTTPException
from sqlalchemy.orm import Session

from core.event_bus import event_bus, ALERT_EVENT
from core.exceptions import DomainError
from models.base import new_id, utcnow
from models.incident import Incident
from repositories.incident_repository import IncidentRepository
from services.audit_service import AuditService


class IncidentService:
    def __init__(self, db: Session, audit_svc: AuditService):
        self.db = db
        self.repo = IncidentRepository(db)
        self.audit_svc = audit_svc
        
        self.signatures_dir = os.path.join(os.getcwd(), "uploads", "signatures")
        os.makedirs(self.signatures_dir, exist_ok=True)

    def _determine_severity(self, exc: Exception) -> str:
        """예외 타입이나 메시지에 따라 Severity 판단."""
        msg = str(exc).lower()
        if "connection" in msg or "timeout" in msg or "database" in msg or "psycopg" in msg:
            return "CRITICAL"
        if "model" in msg or "inference" in msg or "predict" in msg or "emr" in msg:
            return "HIGH"
        return "MEDIUM"

    def report_exception(
        self, 
        exc: Exception, 
        module_name: str, 
        endpoint: str | None = None
    ) -> None:
        """Unhandled Exception 발생 시 자동 보고 (미들웨어 등에서 비동기적 혹은 분리 트랜잭션으로 호출 권장).
        - 400~422 에러 및 Validation 에러는 무시.
        """
        if isinstance(exc, HTTPException) and 400 <= exc.status_code < 500:
            return
        if isinstance(exc, DomainError) and exc.status_code and 400 <= exc.status_code < 500:
            return
        # Validation error (pydantic)
        if exc.__class__.__name__ == "ValidationError":
            return

        severity = self._determine_severity(exc)
        error_message = f"{exc.__class__.__name__}: {str(exc)}"
        stack_trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

        now = utcnow()
        
        # 중복 감지
        existing = self.repo.find_open_incident(
            severity=severity,
            module_name=module_name,
            endpoint=endpoint,
            error_message=error_message
        )
        
        if existing:
            self.repo.increment_occurrence(existing.id, now)
            incident = existing
        else:
            prefix = now.strftime("%y%m%d")
            incident_id = new_id("inc")
            incident = Incident(
                id=incident_id,
                incident_no=f"INC-{prefix}-{incident_id[-6:]}",
                status="OPEN",
                severity=severity,
                module_name=module_name,
                endpoint=endpoint,
                error_message=error_message,
                stack_trace=stack_trace,
                occurrence_count=1,
                first_occurred_at=now,
                last_occurred_at=now,
            )
            self.repo.add(incident)
            
            # 관리자(admin) 부서로 실시간 알림 발송
            event_bus.publish(
                ALERT_EVENT,
                {
                    "department": "admin",
                    "type": "INCIDENT_CREATED",
                    "severity": severity,
                    "module": module_name,
                    "timestamp": now.isoformat(),
                    "title": "운영 장애 발생",
                    "message": error_message,
                    "incident_id": incident.id,
                }
            )

        self.db.commit()

    def list_incidents(self, status: str | None = None, sort_by: str = "latest") -> Sequence[Incident]:
        return self.repo.list_incidents(status=status, sort_by=sort_by)
        
    def get_incident(self, incident_id: str) -> Incident:
        inc = self.repo.get(incident_id)
        if not inc:
            raise DomainError("해당 Incident 를 찾을 수 없습니다.", status_code=404)
        return inc

    def update_status(self, incident_id: str, new_status: str, user_id: str, user_name: str, ip: str, ua: str) -> Incident:
        inc = self.get_incident(incident_id)
        if inc.status == "LOCKED":
            raise DomainError("LOCKED 상태의 Incident 는 수정할 수 없습니다.", status_code=403)
            
        old_status = inc.status
        inc.status = new_status
        if new_status == "LOCKED":
            inc.locked_at = utcnow()
            
        self.audit_svc.record(
            user_id=user_id,
            action=f"STATUS_CHANGED_{old_status}_TO_{new_status}",
            target_type="incident",
            target_id=incident_id,
            payload={"old": old_status, "new": new_status, "user_name": user_name},
            ip_address=ip,
            user_agent=ua,
        )
        self.db.commit()
        return inc

    def resolve_incident(
        self, incident_id: str, root_cause: str, action_taken: str, user_id: str, user_name: str, ip: str, ua: str
    ) -> Incident:
        inc = self.get_incident(incident_id)
        if inc.status == "LOCKED":
            raise DomainError("LOCKED 상태에서는 조치 내용을 수정할 수 없습니다.", status_code=403)
            
        inc.root_cause = root_cause
        inc.action_taken = action_taken
        inc.resolved_by_user_id = user_id
        inc.resolved_by_name = user_name
        inc.resolved_at = utcnow()
        inc.status = "RESOLVED"
        
        self.audit_svc.record(
            user_id=user_id,
            action="RESOLVED",
            target_type="incident",
            target_id=incident_id,
            payload={"root_cause": root_cause, "action_taken": action_taken, "user_name": user_name},
            ip_address=ip,
            user_agent=ua,
        )
        self.db.commit()
        return inc

    def update_draft(self, incident_id: str, root_cause: str, action_taken: str) -> Incident:
        inc = self.get_incident(incident_id)
        if inc.status == "LOCKED":
            raise DomainError("LOCKED 상태에서는 조치 내용을 수정할 수 없습니다.", status_code=403)
        inc.root_cause = root_cause
        inc.action_taken = action_taken
        self.db.commit()
        return inc

    def assign_incident(
        self, incident_id: str, assign_to_user_id: str | None, assign_to_name: str | None, 
        user_id: str, user_name: str, ip: str, ua: str
    ) -> Incident:
        inc = self.get_incident(incident_id)
        if inc.status == "LOCKED":
            raise DomainError("LOCKED 상태의 Incident 는 담당자를 변경할 수 없습니다.", status_code=403)
            
        old_assigned = inc.assigned_to_user_id
        inc.assigned_to_user_id = assign_to_user_id
        inc.assigned_to_name = assign_to_name
        
        self.audit_svc.record(
            user_id=user_id,
            action="ASSIGN_CHANGED",
            target_type="incident",
            target_id=incident_id,
            payload={"old_assigned": old_assigned, "new_assigned": assign_to_user_id, "assignee_name": assign_to_name},
            ip_address=ip,
            user_agent=ua,
        )
        self.db.commit()
        return inc

    def sign_incident(
        self, incident_id: str, user_id: str, user_name: str, user_signature_path: str | None, ip: str, ua: str
    ) -> Incident:
        inc = self.get_incident(incident_id)
        if inc.status == "LOCKED":
            raise DomainError("이미 LOCKED 되었습니다.", status_code=403)
        if inc.status != "RESOLVED":
            raise DomainError("RESOLVED 상태에서만 서명이 가능합니다.", status_code=400)
            
        if not user_signature_path:
            raise DomainError("사용자 프로필에 등록된 서명이 없습니다. 먼저 프로필에서 서명을 등록하세요.", status_code=400)
            
        inc.signature_path = user_signature_path
        inc.signed_at = utcnow()
        inc.status = "SIGNED"
        
        self.audit_svc.record(
            user_id=user_id,
            action="ELECTRONIC_SIGNATURE",
            target_type="incident",
            target_id=incident_id,
            payload={"user_name": user_name, "signature_path": inc.signature_path},
            ip_address=ip,
            user_agent=ua,
        )
        self.db.commit()
        return inc
