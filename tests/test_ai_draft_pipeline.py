"""Integration — AI Draft 전체 파이프라인 (작업지시서 10·11).

Audio → STT → SOAP → Problem List → CDSS Risk → Timeline Event 까지 검증.
"""
from models.patient import PatientLab, PatientTrendPoint, PatientUrinePoint
from models.timeline import TimelineEvent
from services.ai_draft_service import AiDraftService
from tests.factories import make_patient


def _aki_patient(db, *, severe=True):
    p = make_patient(db, diagnosis="당뇨병성 신증")
    if severe:
        rows = [
            PatientTrendPoint(patient_id=p.id, date="2026-06-01", creatinine=1.0, egfr=80, bun=20),
            PatientTrendPoint(patient_id=p.id, date="2026-06-03", creatinine=4.2, egfr=14, bun=62),
            PatientUrinePoint(patient_id=p.id, date="2026-06-03", value=0.2),
            PatientLab(patient_id=p.id, seq=0, key="cr", label="Cr", value=4.2, unit="mg/dL", flag="high"),
            PatientLab(patient_id=p.id, seq=1, key="k", label="K", value=6.3, unit="mmol/L", flag="high"),
        ]
    else:
        rows = [
            PatientTrendPoint(patient_id=p.id, date="2026-06-01", creatinine=0.9, egfr=95, bun=12),
            PatientUrinePoint(patient_id=p.id, date="2026-06-01", value=1.4),
            PatientLab(patient_id=p.id, seq=0, key="cr", label="Cr", value=0.9, unit="mg/dL", flag="normal"),
        ]
    db.add_all(rows)
    db.commit()
    return p


def test_transcribe_passthrough_without_audio(db):
    out = AiDraftService(db).transcribe(None, "환자 호소 텍스트")
    assert out["transcript"] == "환자 호소 텍스트"
    assert out["engine"] in {"passthrough", "whisper"}


def test_full_pipeline_persists_and_emits_timeline(db):
    p = _aki_patient(db, severe=True)
    svc = AiDraftService(db)
    note = svc.generate_draft(
        patient_id=p.id,
        transcript="다리가 붓고 소변이 줄었어요. 메스꺼워요.",
        actor_id="u-neph",
    )
    out = svc.to_out(note)

    # SOAP 구조 + 근거.
    assert set(out["soap"]) == {"S", "O", "A", "P"}
    assert out["soap"]["A"]["evidence"]
    # Problem list (A 기반).
    assert out["problem_list"]
    assert all(pr["source"] == "A" for pr in out["problem_list"])
    # Risk breakdown + 검증 통과.
    assert out["risk"]["breakdown"]
    assert out["validation"]["passed"]

    # 타임라인: ai_draft_note 이벤트 적재.
    drafts = (
        db.query(TimelineEvent)
        .filter(TimelineEvent.patient_id == p.id, TimelineEvent.event_type == "ai_draft_note")
        .all()
    )
    assert drafts
    # 고위험 → AI_ALERT + nephrology trigger.
    if out["risk"]["nephrology_trigger"]:
        alerts = (
            db.query(TimelineEvent)
            .filter(TimelineEvent.patient_id == p.id, TimelineEvent.event_type == "AI_ALERT")
            .all()
        )
        assert alerts and alerts[0].severity == "CRITICAL"


def test_pipeline_reproducible_risk(db):
    p = _aki_patient(db, severe=True)
    svc = AiDraftService(db)
    r1 = svc.run_pipeline(p, "소변이 줄었어요")["risk"]["risk_score"]
    r2 = svc.run_pipeline(p, "소변이 줄었어요")["risk"]["risk_score"]
    assert r1 == r2


def test_low_risk_no_nephrology_trigger(db):
    p = _aki_patient(db, severe=False)
    result = AiDraftService(db).run_pipeline(p, "별다른 증상 없어요")
    assert result["risk"]["tier"] in {"LOW", "MEDIUM"}
    if result["risk"]["tier"] == "LOW":
        assert result["risk"]["nephrology_trigger"] is False


def test_api_draft_endpoint_returns_structured(client, neph_headers, db):
    p = _aki_patient(db, severe=True)
    r = client.post(
        "/api/voice/draft",
        headers=neph_headers,
        json={"patientId": p.id, "transcript": "다리가 붓고 소변이 줄었어요"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert "soap" in body and "problemList" in body and "risk" in body
    assert body["validation"]["passed"] is True
    assert set(body["soap"]) == {"S", "O", "A", "P"}
