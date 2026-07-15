"""병상 입실 상세 / 예약 조회 서비스 (읽기 전용 enrichment).

bed_service(트랜잭션) 와 분리됨: 이 서비스는 병상 클릭 시 보여줄 상세/예약 정보의
조회만 담당한다. 명시 데이터가 없는 사용중/예약 병상은 기본값으로 보강한다
(프론트 bedDetailFor / reservationFor 동작과 동일).
"""
import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from models.bed import Bed
from models.bed_detail import BedDetail
from schemas.bed_detail import (
    BedLabOut,
    BedPatientDetailOut,
    MedicationOut,
    VitalEntryOut,
)

# 명시 상세가 없는 사용중 병상의 기본 상세(프론트 기본값과 동일).
_DEFAULT_MEDS = [
    MedicationOut(name="생리식염수 (0.9% NaCl)", dose="유지 60 mL/hr", route="IV", status="투여중"),
    MedicationOut(name="Acetaminophen", dose="650 mg prn", route="PO", status="필요시"),
]
_DEFAULT_LABS = [
    BedLabOut(label="Creatinine", value="0.9", unit="mg/dL", flag="normal"),
    BedLabOut(label="WBC", value="7.4", unit="10³/µL", flag="normal"),
    BedLabOut(label="Hb", value="13.1", unit="g/dL", flag="normal"),
]


def _default_detail() -> BedPatientDetailOut:
    return BedPatientDetailOut(
        diagnosis="입원 경과 관찰 중",
        attending="담당의 미지정",
        admitted_at="2026-06-14T00:00:00",
        aki_risk=False,
        medications=list(_DEFAULT_MEDS),
        labs=list(_DEFAULT_LABS),
    )


def _row_to_detail(row: BedDetail) -> BedPatientDetailOut:
    recent = json.loads(row.recent_inputs_json) if row.recent_inputs_json else None
    items = json.loads(row.treatment_items_json) if row.treatment_items_json else None
    return BedPatientDetailOut(
        diagnosis=row.diagnosis,
        attending=row.attending,
        admitted_at=row.admitted_at,
        aki_risk=row.aki_risk,
        aki_stage=row.aki_stage,
        medications=[MedicationOut(**m) for m in json.loads(row.medications_json or "[]")],
        labs=[BedLabOut(**lab) for lab in json.loads(row.labs_json or "[]")],
        recent_inputs=[VitalEntryOut(**v) for v in recent] if recent else None,
        treatment_report=row.treatment_report,
        treatment_items=items,
    )




class BedDetailService:
    def __init__(self, db: Session):
        self.db = db

    def list_details(self) -> dict[str, BedPatientDetailOut]:
        """모든 사용중 병상의 상세(명시값 우선, 없으면 기본값). bedId → 상세."""
        occupied = (
            self.db.execute(select(Bed).where(Bed.state == "occupied")).scalars().all()
        )
        rows = {r.bed_id: r for r in self.db.execute(select(BedDetail)).scalars().all()}
        result: dict[str, BedPatientDetailOut] = {}
        for bed in occupied:
            row = rows.get(bed.id)
            result[bed.id] = _row_to_detail(row) if row else _default_detail()
        return result

