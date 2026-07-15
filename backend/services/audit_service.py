"""감사 서비스.

책임: 주요 상태변경 이벤트의 감사 로그 기록 + 조회.
- record() 는 commit 하지 않는다 → 호출한 서비스의 트랜잭션에 함께 포함된다
  (상태변경과 감사기록의 원자성 보장).
"""
import json

from sqlalchemy.orm import Session

from core.query_optimizer import Page
from models.audit import AuditLog
from models.base import new_id
from repositories.audit_repository import AuditRepository


class AuditService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = AuditRepository(db)

    def record(
        self,
        *,
        user_id: str | None,
        action: str,
        target_type: str,
        target_id: str | None = None,
        payload: dict | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> AuditLog:
        """감사 로그 적재(flush only). 트랜잭션 commit 은 상위 서비스 책임."""
        
        # payload 내부에 부가 정보 병합
        actual_payload = payload.copy() if payload else {}
        if ip_address:
            actual_payload["ip_address"] = ip_address
        if user_agent:
            actual_payload["user_agent"] = user_agent
            
        log = AuditLog(
            id=new_id("audit"),
            user_id=user_id,
            action=action,
            target_type=target_type,
            target_id=target_id,
            payload=json.dumps(actual_payload, ensure_ascii=False, default=str) if actual_payload else None,
        )
        return self.repo.add(log)

    def list_recent(self, page: Page, target_type: str | None = None) -> list[AuditLog]:
        return self.repo.list_recent(page, target_type)
