"""Service — 병상 배정/해제 트랜잭션 (작업지시서 7.1, 핵심 트랜잭션)."""
import pytest

from core.exceptions import ConflictError, NotFoundError
from models.timeline import TimelineEvent
from repositories.admission_repository import AdmissionRepository
from services.bed_service import BedService
from tests.factories import make_bed, make_patient


def test_assign_available_bed_to_existing_patient(db):
    bed = make_bed(db, state="available")
    patient = make_patient(db)

    out_bed, admission = BedService(db).assign_bed(
        bed_id=bed.id, actor_id="u-admin", actor_name="관리자", patient_id=patient.id
    )

    assert out_bed.state == "occupied"
    assert out_bed.patient_id == patient.id
    assert admission.status == "active"
    # BED_CHANGE 타임라인 이벤트가 같은 트랜잭션에 기록됨.
    events = (
        db.query(TimelineEvent)
        .filter(TimelineEvent.patient_id == patient.id, TimelineEvent.event_type == "BED_CHANGE")
        .all()
    )
    assert any("배정" in e.title for e in events)


def test_assign_occupied_bed_raises_conflict(db):
    bed = make_bed(db, state="occupied")
    patient = make_patient(db)
    with pytest.raises(ConflictError):
        BedService(db).assign_bed(
            bed_id=bed.id, actor_id="u", actor_name="x", patient_id=patient.id
        )


def test_assign_unknown_bed_raises_not_found(db):
    patient = make_patient(db)
    with pytest.raises(NotFoundError):
        BedService(db).assign_bed(
            bed_id="bed-does-not-exist", actor_id="u", actor_name="x", patient_id=patient.id
        )


def test_assign_walk_in_creates_new_patient(db):
    bed = make_bed(db, state="available")
    out_bed, admission = BedService(db).assign_bed(
        bed_id=bed.id, actor_id="u", actor_name="응급의", patient_name="익명응급환자", sex="F", age=33
    )
    assert out_bed.state == "occupied"
    assert out_bed.patient_name == "익명응급환자"
    assert admission.patient_id == out_bed.patient_id


def test_assign_without_patient_info_raises_conflict(db):
    bed = make_bed(db, state="available")
    with pytest.raises(ConflictError):
        BedService(db).assign_bed(bed_id=bed.id, actor_id="u", actor_name="x")


def test_release_occupied_bed_closes_admission(db):
    bed = make_bed(db, state="available")
    patient = make_patient(db)
    svc = BedService(db)
    svc.assign_bed(bed_id=bed.id, actor_id="u", actor_name="x", patient_id=patient.id)

    out_bed, admission = svc.release_bed(bed_id=bed.id, actor_id="u", actor_name="x")
    assert out_bed.state == "cleaning"
    assert out_bed.patient_id is None
    assert admission.status == "discharged"
    # 활성 입원이 더 이상 없어야 한다.
    assert AdmissionRepository(db).get_active_by_bed(bed.id) is None


def test_release_non_occupied_raises_conflict(db):
    bed = make_bed(db, state="available")
    with pytest.raises(ConflictError):
        BedService(db).release_bed(bed_id=bed.id, actor_id="u", actor_name="x")


def test_duplicate_admission_blocked(db):
    """같은 환자를 두 병상에 동시 입원시키면 차단(활성 입원 1개 규칙)."""
    bed1 = make_bed(db, state="available")
    bed2 = make_bed(db, state="available")
    patient = make_patient(db)
    svc = BedService(db)
    svc.assign_bed(bed_id=bed1.id, actor_id="u", actor_name="x", patient_id=patient.id)
    with pytest.raises(ConflictError):
        svc.assign_bed(bed_id=bed2.id, actor_id="u", actor_name="x", patient_id=patient.id)


def test_summarize_counts_by_zone(db):
    beds = [
        make_bed(db, zone="ztest", state="occupied"),
        make_bed(db, zone="ztest", state="available"),
        make_bed(db, zone="ztest", state="cleaning"),
    ]
    summary = {s["zone"]: s for s in BedService(db).summarize(beds)}
    z = summary["ztest"]
    assert z["total"] == 3 and z["occupied"] == 1 and z["available"] == 1
