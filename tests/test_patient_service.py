"""Service — 환자 목록/상세 조회 (작업지시서 7.1)."""
import pytest

from core.exceptions import NotFoundError
from core.query_optimizer import Page
from models.patient import PatientLab
from services.patient_service import PatientService
from tests.factories import make_patient


def test_list_summary_returns_patients(db):
    make_patient(db)
    rows = PatientService(db).list_summary(Page.of(20, 0))
    assert len(rows) >= 1


def test_list_summary_respects_page_size(db):
    for _ in range(3):
        make_patient(db)
    rows = PatientService(db).list_summary(Page.of(2, 0))
    assert len(rows) <= 2


def test_get_detail_returns_patient_with_labs(db):
    p = make_patient(db)
    db.add(PatientLab(patient_id=p.id, seq=0, key="cr", label="Cr",
                      value=1.2, unit="mg/dL", flag="normal"))
    db.commit()
    detail = PatientService(db).get_detail(p.id)
    assert detail.id == p.id
    assert any(lab.key == "cr" for lab in detail.labs)


def test_get_detail_unknown_raises_not_found(db):
    with pytest.raises(NotFoundError):
        PatientService(db).get_detail("p-does-not-exist")
