"""입원 레포지토리 — 활성 입원 조회(중복 입원/해제 트랜잭션 근거)."""
from sqlalchemy import select

from models.admission import Admission
from repositories.base import BaseRepository


class AdmissionRepository(BaseRepository[Admission]):
    model = Admission

    def get_active_by_patient(self, patient_id: str) -> Admission | None:
        """환자의 활성 입원 — 중복 입원 방지 체크에 사용."""
        stmt = (
            select(Admission)
            .where(Admission.patient_id == patient_id, Admission.status == "active")
            .limit(1)
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def get_active_by_bed(self, bed_id: str) -> Admission | None:
        """병상의 활성 입원 — 해제 트랜잭션에서 종료 대상 식별."""
        stmt = (
            select(Admission)
            .where(Admission.bed_id == bed_id, Admission.status == "active")
            .with_for_update()
            .limit(1)
        )
        return self.db.execute(stmt).scalar_one_or_none()
