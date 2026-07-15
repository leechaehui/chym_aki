"""Integration — 도메인 API 엔드포인트 계약 (작업지시서 3.3 / 7.1).

beds / consultation / timeline / notification / audit / voice 의 핵심 계약을
인증·RBAC 포함해 검증한다(서비스+레포+직렬화 통합 경로).
"""
import uuid

from models.pathology import PathologyResult
from tests.factories import make_bed, make_patient


# ---------------- beds ----------------
def test_list_beds(client, neph_headers):
    r = client.get("/api/beds", headers=neph_headers)
    assert r.status_code == 200 and isinstance(r.json(), list)


def test_bed_summary(client, neph_headers):
    r = client.get("/api/beds/summary", headers=neph_headers)
    assert r.status_code == 200


def test_bed_details(client, neph_headers):
    assert client.get("/api/beds/details", headers=neph_headers).status_code == 200



def test_assign_and_release_bed_via_api(client, admin_headers, db):
    bed = make_bed(db, state="available")
    patient = make_patient(db)
    assign = client.post(
        f"/api/beds/{bed.id}/assign", headers=admin_headers,
        json={"patientId": patient.id},
    )
    assert assign.status_code == 200, assign.text
    assert assign.json()["bed"]["state"] == "occupied"

    release = client.post(f"/api/beds/{bed.id}/release", headers=admin_headers)
    assert release.status_code == 200
    assert release.json()["bed"]["state"] == "cleaning"


def test_assign_requires_nephrology_or_admin(client, path_headers, db):
    bed = make_bed(db, state="available")
    patient = make_patient(db)
    r = client.post(
        f"/api/beds/{bed.id}/assign", headers=path_headers, json={"patientId": patient.id}
    )
    assert r.status_code == 403
    assert r.json()["error_code"] == "FORBIDDEN"


# ---------------- consultation ----------------
def test_consultation_lifecycle_via_api(client, neph_headers):
    req = client.post(
        "/api/consultations", headers=neph_headers,
        json={"kind": "pathology", "patientMrn": "MRN-API", "patientName": "API환자",
              "diagnosis": "IgAN 의증", "requestedBy": "neph_hong"},
    )
    assert req.status_code == 201, req.text
    cid = req.json()["id"]

    assert client.get("/api/consultations", headers=neph_headers).status_code == 200
    assert client.get(f"/api/consultations/{cid}", headers=neph_headers).status_code == 200

    acc = client.post(f"/api/consultations/{cid}/accept", headers=neph_headers,
                      json={"actor": "path_lee"})
    assert acc.status_code == 200 and acc.json()["status"] == "in_progress"

    rep = client.post(f"/api/consultations/{cid}/reply", headers=neph_headers,
                      json={"findings": "f", "diagnosis": "IgAN", "recommendation": "r", "author": "path_lee"})
    assert rep.status_code == 200 and rep.json()["status"] == "replied"


# ---------------- timeline ----------------
def test_timeline_create_and_read(client, admin_headers, db):
    patient = make_patient(db)
    create = client.post(
        "/api/timeline/event", headers=admin_headers,
        json={"patientId": patient.id, "eventType": "LAB_RESULT", "title": "Cr 상승",
              "severity": "WARNING", "source": "LAB_SYSTEM"},
    )
    assert create.status_code == 201, create.text
    read = client.get(f"/api/timeline/patient/{patient.id}", headers=admin_headers)
    assert read.status_code == 200
    assert any(e["title"] == "Cr 상승" for e in read.json())


def test_timeline_create_requires_nephrology_or_admin(client, path_headers, db):
    patient = make_patient(db)
    r = client.post(
        "/api/timeline/event", headers=path_headers,
        json={"patientId": patient.id, "eventType": "LAB_RESULT", "title": "x", "source": "MANUAL"},
    )
    assert r.status_code == 403


# ---------------- notification ----------------
def test_notification_create_list_read(client, neph_headers):
    create = client.post(
        "/api/notifications", headers=neph_headers,
        json={"title": "AKI 경보", "message": "고위험", "severity": "CRITICAL", "department": "nephrology"},
    )
    assert create.status_code == 201, create.text
    nid = create.json()["id"]
    assert client.get("/api/notifications?department=nephrology", headers=neph_headers).status_code == 200
    read = client.post(f"/api/notifications/{nid}/read", headers=neph_headers)
    assert read.status_code == 200 and read.json()["read"] is True


# ---------------- audit (admin only) ----------------
def test_audit_requires_admin(client, neph_headers, admin_headers):
    assert client.get("/api/audit", headers=neph_headers).status_code == 403
    assert client.get("/api/audit", headers=admin_headers).status_code == 200


# ---------------- auth (signup/승인 플로우) ----------------
def test_signup_then_admin_approval_flow(client, admin_headers):
    import uuid
    uname = f"api-{uuid.uuid4().hex[:8]}"

    signup = client.post(
        "/api/auth/signup",
        json={"username": uname, "password": "Secret123!", "name": "신규",
              "role": "nephrology", "department": "신장내과", "signatureBase64": "ZHVtbXk="},
    )
    assert signup.status_code == 201, signup.text

    # pending 이라 로그인 차단.
    pending_login = client.post("/api/auth/login", json={"username": uname, "password": "Secret123!"})
    assert pending_login.status_code == 403

    # admin 이 계정 목록에서 찾아 승인.
    accounts = client.get("/api/auth/accounts", headers=admin_headers)
    assert accounts.status_code == 200
    uid = next(a["id"] for a in accounts.json() if a["username"] == uname)
    approve = client.patch(
        f"/api/auth/accounts/{uid}/approval", headers=admin_headers,
        json={"approval": "approved"},
    )
    assert approve.status_code == 200 and approve.json()["approval"] == "approved"

    # 승인 후 로그인 성공.
    ok = client.post("/api/auth/login", json={"username": uname, "password": "Secret123!"})
    assert ok.status_code == 200 and "accessToken" in ok.json()


def test_accounts_list_requires_admin(client, neph_headers):
    assert client.get("/api/auth/accounts", headers=neph_headers).status_code == 403


# ---------------- pathology (병리과 API) ----------------
def _make_pathology(db, consult_id):
    row = PathologyResult(
        consult_id=consult_id, stain="PAS", image_url="http://x/wsi.svs",
        layers_json="[]", metrics_json="[]",
        report_findings="사구체 경화", report_diagnosis="IgA 신병증",
        report_status="final", report_updated_at=None,
    )
    db.add(row)
    db.commit()
    return row


def test_pathology_list_and_get_by_consult(client, neph_headers, db):
    cid = f"c-{uuid.uuid4().hex[:8]}"
    _make_pathology(db, cid)

    lst = client.get("/api/pathology", headers=neph_headers)
    assert lst.status_code == 200 and isinstance(lst.json(), list)

    one = client.get(f"/api/pathology/by-consult/{cid}", headers=neph_headers)
    assert one.status_code == 200, one.text
    assert one.json()["report"]["diagnosis"] == "IgA 신병증"


def test_pathology_unknown_consult_returns_404(client, neph_headers):
    r = client.get("/api/pathology/by-consult/c-none", headers=neph_headers)
    assert r.status_code == 404
    assert r.json()["error_code"] == "NOT_FOUND"


# ---------------- nephrology (AKI API) ----------------
def test_nephrology_analyze_patient_api(client, neph_headers, db):
    patient = make_patient(db)
    r = client.post(f"/api/nephrology/aki/analyze/{patient.id}", headers=neph_headers)
    assert r.status_code == 200, r.text
    assert 0 <= r.json()["riskScore"] <= 100


def test_nephrology_predict_vector_api(client, neph_headers):
    r = client.post(
        "/api/nephrology/aki/predict-vector", headers=neph_headers,
        json={"creatinine_max": 3.0, "creatinine_min": 1.0, "urine_output_6h": 50},
    )
    assert r.status_code == 200, r.text
    assert r.json()["source"] in {"model", "rule-based", "hybrid"}


def test_nephrology_analyze_patient_requires_role(client, db):
    """인증 없으면 401(RBAC)."""
    patient = make_patient(db)
    r = client.post(f"/api/nephrology/aki/analyze/{patient.id}")
    assert r.status_code == 401


# ---------------- voice drafts 이력 ----------------
def test_voice_list_drafts_api(client, neph_headers, db):
    patient = make_patient(db)
    # 초안 1건 생성 후 이력 조회.
    client.post("/api/voice/draft", headers=neph_headers,
                json={"patientId": patient.id, "transcript": "소변이 줄었어요"})
    r = client.get(f"/api/voice/drafts/{patient.id}", headers=neph_headers)
    assert r.status_code == 200
    drafts = r.json()
    assert isinstance(drafts, list) and len(drafts) >= 1
    assert "soap" in drafts[0]


# ---------------- voice transcribe ----------------
def test_voice_transcribe_passthrough(client, neph_headers):
    r = client.post("/api/voice/transcribe", headers=neph_headers, data={"text": "환자 호소"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["transcript"] == "환자 호소"
    assert body["engine"] in {"passthrough", "whisper"}
