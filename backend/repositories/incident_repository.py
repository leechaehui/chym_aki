"""장애 관리 저장소.

책임: Incident 엔티티에 대한 영속성 처리.
- 발생 빈도가 높을 수 있으므로 upsert 성격의 중복 감지가 중요하다.
"""
from datetime import datetime
from collections.abc import Sequence

from sqlalchemy import select, update, desc, asc
from sqlalchemy.orm import Session

from models.incident import Incident


class IncidentRepository:
    def __init__(self, db: Session):
        self.db = db

    def get(self, incident_id: str) -> Incident | None:
        return self.db.execute(
            select(Incident).where(Incident.id == incident_id)
        ).scalar_one_or_none()
        
    def find_open_incident(
        self, severity: str, module_name: str, endpoint: str | None, error_message: str
    ) -> Incident | None:
        """동일한 장애가 OPEN 인 상태로 존재하는지 확인."""
        query = select(Incident).where(
            Incident.severity == severity,
            Incident.module_name == module_name,
            Incident.error_message == error_message,
            Incident.status == "OPEN",
        )
        if endpoint:
            query = query.where(Incident.endpoint == endpoint)
        else:
            query = query.where(Incident.endpoint.is_(None))
            
        return self.db.execute(query).scalar_one_or_none()

    def add(self, incident: Incident) -> Incident:
        self.db.add(incident)
        self.db.flush()
        return incident
        
    def increment_occurrence(self, incident_id: str, occurred_at: datetime) -> None:
        """기존 장애 발생 횟수 증가 및 최근 발생시각 갱신."""
        self.db.execute(
            update(Incident)
            .where(Incident.id == incident_id)
            .values(
                occurrence_count=Incident.occurrence_count + 1,
                last_occurred_at=occurred_at,
                updated_at=occurred_at,
            )
        )
        self.db.flush()

    def list_incidents(
        self, 
        status: str | None = None,
        sort_by: str = "latest"  # latest, severity, frequency
    ) -> Sequence[Incident]:
        query = select(Incident)
        
        if status:
            query = query.where(Incident.status == status)
            
        if sort_by == "latest":
            query = query.order_by(desc(Incident.last_occurred_at))
        elif sort_by == "severity":
            from sqlalchemy import case
            severity_order = case(
                (Incident.severity == "CRITICAL", 1),
                (Incident.severity == "HIGH", 2),
                (Incident.severity == "MEDIUM", 3),
                (Incident.severity == "LOW", 4),
                else_=5
            )
            query = query.order_by(severity_order, desc(Incident.last_occurred_at))
        elif sort_by == "frequency":
            query = query.order_by(desc(Incident.occurrence_count), desc(Incident.last_occurred_at))
        else:
            query = query.order_by(desc(Incident.created_at))
            
        return self.db.execute(query).scalars().all()
