"""환자 서비스.

책임: 환자 목록/상세 조회. (생성은 응급 입실 트랜잭션에서 BedService 가 수행)
"""
from sqlalchemy.orm import Session

from core.exceptions import NotFoundError
from core.query_optimizer import Page
from models.patient import Patient
from repositories.patient_repository import PatientRepository


class PatientService:
    def __init__(self, db: Session):
        self.db = db
        self.patients = PatientRepository(db)

    def list_summary(self, page: Page) -> list[Patient]:
        """목록(요약) — 검사 배열 미적재."""
        return self.patients.list_summary(page)

    def list_detailed(self, page: Page) -> list[Patient]:
        """목록(검사 포함) — labs/trend/urine eager-load. 신장내과 위험판정 화면용."""
        return self.patients.list_detailed(page)

    def get_detail(self, patient_id: str) -> Patient:
        """상세 — labs/trend/urine 포함. 없으면 404."""
        patient = self.patients.get_with_details(patient_id)
        if not patient:
            raise NotFoundError("환자를 찾을 수 없습니다.")
        return patient
