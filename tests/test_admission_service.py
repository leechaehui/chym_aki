"""Service — 입원 생성/종료 단일 책임 (작업지시서 7.1, SRP)."""
import pytest

from core.exceptions import ConflictError
from services.admission_service import AdmissionService
from tests.factories import make_bed, make_patient


def test_open_creates_active_admission(db):
    patient = make_patient(db)
    bed = make_bed(db)
    adm = AdmissionService(db).open(patient.id, bed.id)
    db.commit()
    assert adm.status == "active"
    assert adm.patient_id == patient.id and adm.bed_id == bed.id


def test_open_duplicate_active_raises_conflict(db):
    patient = make_patient(db)
    bed = make_bed(db)
    svc = AdmissionService(db)
    svc.open(patient.id, bed.id)
    db.commit()
    with pytest.raises(ConflictError):
        svc.open(patient.id, bed.id)


def test_close_by_bed_discharges_active(db):
    patient = make_patient(db)
    bed = make_bed(db)
    svc = AdmissionService(db)
    svc.open(patient.id, bed.id)
    db.commit()

    closed = svc.close_by_bed(bed.id)
    db.commit()
    assert closed is not None
    assert closed.status == "discharged"
    assert closed.discharged_at is not None


def test_close_by_bed_without_active_returns_none(db):
    bed = make_bed(db)
    assert AdmissionService(db).close_by_bed(bed.id) is None
