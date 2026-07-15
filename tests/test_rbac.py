"""RBAC / 인증 게이트 검증 (작업지시서 3.3, DB 권한 설계).

User.role(admin|emergency|nephrology|pathology) + approval 기반 접근제어가
엔드포인트별로 올바르게 강제되는지 역할 매트릭스로 검증한다.
표준 error_code(FORBIDDEN/UNAUTHENTICATED)도 함께 확인한다.
"""
import pytest

from core.security import create_access_token
from tests.factories import make_bed, make_patient, make_user


# ============================================================
# 1) 쓰기/관리 엔드포인트 — 허용 역할만 통과, 나머지는 403 FORBIDDEN
# ============================================================
def _assert_forbidden(resp):
    assert resp.status_code == 403, resp.text
    assert resp.json()["error_code"] == "FORBIDDEN"



@pytest.mark.parametrize("role_headers", ["neph_headers", "path_headers"])
def test_bed_assign_forbidden_for_non_admin(client, db, request, role_headers):
    headers = request.getfixturevalue(role_headers)
    bed = make_bed(db, state="available")
    patient = make_patient(db)
    _assert_forbidden(
        client.post(f"/api/beds/{bed.id}/assign", headers=headers, json={"patientId": patient.id})
    )


def test_timeline_write_allowed_for_admin_forbidden_for_others(
    client, admin_headers, neph_headers, path_headers, db
):
    patient = make_patient(db)
    body = {"patientId": patient.id, "eventType": "LAB_RESULT", "title": "t", "source": "LAB_SYSTEM"}
    assert client.post("/api/timeline/event", headers=admin_headers, json=body).status_code == 201
    _assert_forbidden(client.post("/api/timeline/event", headers=neph_headers, json=body))
    _assert_forbidden(client.post("/api/timeline/event", headers=path_headers, json=body))


@pytest.mark.parametrize("role_headers", ["neph_headers", "path_headers"])
def test_audit_admin_only(client, request, role_headers):
    _assert_forbidden(client.get("/api/audit", headers=request.getfixturevalue(role_headers)))


def test_audit_allowed_for_admin(client, admin_headers):
    assert client.get("/api/audit", headers=admin_headers).status_code == 200


@pytest.mark.parametrize("role_headers", ["neph_headers", "path_headers"])
def test_accounts_admin_only(client, request, role_headers):
    _assert_forbidden(client.get("/api/auth/accounts", headers=request.getfixturevalue(role_headers)))


def test_approval_admin_only(client, neph_headers, admin_headers, db):
    target = make_user(db, approval="pending")
    # 비관리자 → 403
    _assert_forbidden(
        client.patch(f"/api/auth/accounts/{target.id}/approval",
                     headers=neph_headers, json={"approval": "approved"})
    )
    # 관리자 → 200
    ok = client.patch(f"/api/auth/accounts/{target.id}/approval",
                      headers=admin_headers, json={"approval": "approved"})
    assert ok.status_code == 200 and ok.json()["approval"] == "approved"


def test_aki_analyze_nephrology_or_admin_only(client, neph_headers, path_headers, db):
    patient = make_patient(db)
    url = f"/api/nephrology/aki/analyze/{patient.id}"
    assert client.post(url, headers=neph_headers).status_code == 200  # 허용
    _assert_forbidden(client.post(url, headers=path_headers))         # 병리 ✗


def test_voice_draft_nephrology_or_admin_only(client, neph_headers, path_headers, db):
    patient = make_patient(db)
    body = {"patientId": patient.id, "transcript": "소변이 줄었어요"}
    assert client.post("/api/voice/draft", headers=neph_headers, json=body).status_code == 201
    _assert_forbidden(client.post("/api/voice/draft", headers=path_headers, json=body))


# ============================================================
# 1-b) 관리자(admin) — 모든 역할제한 엔드포인트 통과(슈퍼유저 속성)
# ============================================================
def test_admin_can_assign_bed(client, admin_headers, db):
    bed = make_bed(db, state="available")
    patient = make_patient(db)
    r = client.post(f"/api/beds/{bed.id}/assign", headers=admin_headers,
                    json={"patientId": patient.id})
    assert r.status_code == 200, r.text


def test_admin_can_write_timeline(client, admin_headers, db):
    patient = make_patient(db)
    r = client.post("/api/timeline/event", headers=admin_headers,
                    json={"patientId": patient.id, "eventType": "DIAGNOSIS_UPDATE",
                          "title": "관리자 기록", "source": "MANUAL"})
    assert r.status_code == 201, r.text


def test_admin_can_run_aki_analyze(client, admin_headers, db):
    patient = make_patient(db)
    r = client.post(f"/api/nephrology/aki/analyze/{patient.id}", headers=admin_headers)
    assert r.status_code == 200, r.text


def test_admin_can_generate_voice_draft(client, admin_headers, db):
    patient = make_patient(db)
    r = client.post("/api/voice/draft", headers=admin_headers,
                    json={"patientId": patient.id, "transcript": "소변이 줄었어요"})
    assert r.status_code == 201, r.text


def test_admin_can_list_audit_and_accounts(client, admin_headers):
    assert client.get("/api/audit", headers=admin_headers).status_code == 200
    assert client.get("/api/auth/accounts", headers=admin_headers).status_code == 200


def test_admin_can_set_approval(client, admin_headers, db):
    target = make_user(db, approval="pending")
    r = client.patch(f"/api/auth/accounts/{target.id}/approval",
                     headers=admin_headers, json={"approval": "rejected", "rejectionReason": "just because"})
    assert r.status_code == 200 and r.json()["approval"] == "rejected"


# ============================================================
# 2) 인증 게이트(get_current_user) — 토큰/승인 상태 검증
# ============================================================
def test_missing_token_is_401(client):
    r = client.get("/api/auth/me")
    assert r.status_code == 401
    assert r.json()["error_code"] == "UNAUTHENTICATED"


def test_malformed_token_is_401(client):
    r = client.get("/api/auth/me", headers={"Authorization": "Bearer not-a-real-jwt"})
    assert r.status_code == 401


def test_non_bearer_scheme_is_401(client):
    r = client.get("/api/auth/me", headers={"Authorization": "Basic abc123"})
    assert r.status_code == 401


def test_unapproved_account_token_is_rejected(client, db):
    """승인 안 된 계정의 (유효) 토큰으로도 접근 차단 — approval 게이트."""
    pending = make_user(db, approval="pending")
    token = create_access_token(subject=pending.id, role=pending.role)
    r = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 401


def test_token_for_unknown_user_is_rejected(client):
    token = create_access_token(subject="u-does-not-exist", role="admin")
    r = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 401


# ============================================================
# 3) 읽기 엔드포인트 — 인증만 되면 모든 역할 허용
# ============================================================
@pytest.mark.parametrize("role_headers", ["admin_headers", "neph_headers", "path_headers"])
def test_read_endpoints_open_to_all_authenticated(client, request, role_headers):
    headers = request.getfixturevalue(role_headers)
    assert client.get("/api/patients", headers=headers).status_code == 200
    assert client.get("/api/beds", headers=headers).status_code == 200
