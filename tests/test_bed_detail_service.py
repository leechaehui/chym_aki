"""Service — 병상 입실 상세/예약 조회(enrichment) (작업지시서 7.1)."""
import json

from models.bed_detail import BedDetail
from services.bed_detail_service import BedDetailService
from tests.factories import make_bed


def test_occupied_bed_without_row_uses_default_detail(db):
    bed = make_bed(db, state="occupied")
    details = BedDetailService(db).list_details()
    assert bed.id in details
    assert details[bed.id].diagnosis == "입원 경과 관찰 중"  # 기본값
    assert details[bed.id].medications  # 기본 처방 존재


def test_occupied_bed_with_row_uses_explicit_detail(db):
    bed = make_bed(db, state="occupied")
    db.add(
        BedDetail(
            bed_id=bed.id,
            diagnosis="급성 신손상",
            attending="홍신장",
            admitted_at="2026-06-15T09:00:00",
            aki_risk=True,
            aki_stage="AKI Stage 2",
            medications_json=json.dumps(
                [{"name": "Furosemide", "dose": "40mg", "route": "IV", "status": "투여중"}]
            ),
            labs_json=json.dumps(
                [{"label": "Creatinine", "value": "3.1", "unit": "mg/dL", "flag": "high"}]
            ),
        )
    )
    db.commit()
    detail = BedDetailService(db).list_details()[bed.id]
    assert detail.diagnosis == "급성 신손상"
    assert detail.aki_risk is True
    assert detail.medications[0].name == "Furosemide"


