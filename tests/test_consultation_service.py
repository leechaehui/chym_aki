"""Service — 협진 생성/상태전이(접수·회신) (작업지시서 7.1)."""
import json

import pytest

from core.exceptions import NotFoundError
from services.consultation_service import ConsultationService
from tests.factories import make_patient


def _request_data(mrn="MRN-TEST", kind="pathology"):
    return {
        "kind": kind,
        "patient_mrn": mrn,
        "patient_name": "협진환자",
        "diagnosis": "IgA 신병증 의증",
        "key_labs": "Cr 2.1",
        "reason": "조직검사 판독",
        "urgency": "urgent",
        "requested_by": "neph_hong",
    }


def test_request_creates_requested_consult_with_timeline(db):
    svc = ConsultationService(db)
    consult = svc.request(actor_id="u", data=_request_data())
    assert consult.status == "requested"
    assert len(consult.timeline) >= 1
    assert consult.timeline[0].stage == "requested"


def test_request_mirrors_to_patient_timeline_when_mrn_matches(db):
    patient = make_patient(db)
    svc = ConsultationService(db)
    svc.request(actor_id="u", data=_request_data(mrn=patient.mrn))
    from models.timeline import TimelineEvent

    events = (
        db.query(TimelineEvent)
        .filter(TimelineEvent.patient_id == patient.id, TimelineEvent.event_type == "CONSULTATION")
        .all()
    )
    assert events  # 환자 타임라인으로 미러링됨


def test_accept_transitions_to_in_progress(db):
    svc = ConsultationService(db)
    c = svc.request(actor_id="u", data=_request_data())
    accepted = svc.accept(c.id, actor_id="u", actor="path_lee")
    assert accepted.status == "in_progress"
    assert any(e.stage == "received" for e in accepted.timeline)


def test_accept_is_idempotent_when_not_requested(db):
    svc = ConsultationService(db)
    c = svc.request(actor_id="u", data=_request_data())
    svc.accept(c.id, actor_id="u", actor="path_lee")
    again = svc.accept(c.id, actor_id="u", actor="path_lee")  # 이미 in_progress
    assert again.status == "in_progress"


def test_reply_stores_payload_and_marks_replied(db):
    svc = ConsultationService(db)
    c = svc.request(actor_id="u", data=_request_data())
    svc.accept(c.id, actor_id="u", actor="path_lee")
    replied = svc.reply(
        c.id, actor_id="u", reply={"author": "path_lee", "diagnosis": "IgAN 확진"}
    )
    assert replied.status == "replied"
    payload = json.loads(replied.reply_json)
    assert payload["diagnosis"] == "IgAN 확진"
    assert "repliedAt" in payload


def test_get_unknown_raises_not_found(db):
    with pytest.raises(NotFoundError):
        ConsultationService(db).get("c-does-not-exist")
