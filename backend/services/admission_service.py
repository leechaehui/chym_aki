"""입원 서비스.

책임: 입원 레코드의 생성/종료 단위 연산만 담당한다.
- 이 서비스는 bed 상태를 변경하지 않는다(BedService 의 책임) → 관심사 분리.
- commit 하지 않는다 → BedService 트랜잭션의 일부로 호출된다.

bed_service 와 admission_service 는 분리됨:
  BedService 가 트랜잭션을 열고 잠금/상태변경을 조율하며,
  AdmissionService 는 입원 행 생성/종료라는 단일 책임만 수행한다.
"""
from sqlalchemy.orm import Session

from core.exceptions import ConflictError
from models.admission import Admission
from models.base import new_id, utcnow
from repositories.admission_repository import AdmissionRepository


class AdmissionService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = AdmissionRepository(db)

    def open(self, patient_id: str, bed_id: str) -> Admission:
        """입원 생성. 환자에게 활성 입원이 있으면 중복 입원으로 차단."""
        if self.repo.get_active_by_patient(patient_id):
            raise ConflictError("이미 입원(활성) 상태인 환자입니다.")
        admission = Admission(
            id=new_id("adm"),
            patient_id=patient_id,
            bed_id=bed_id,
            status="active",
            admitted_at=utcnow(),
        )
        return self.repo.add(admission)

    def close_by_bed(self, bed_id: str) -> Admission | None:
        """병상의 활성 입원을 종료(discharged). 없으면 None."""
        admission = self.repo.get_active_by_bed(bed_id)
        if not admission:
            return None
        admission.status = "discharged"
        admission.discharged_at = utcnow()
        self.db.flush()
        return admission
