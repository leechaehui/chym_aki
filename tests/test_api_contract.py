"""Integration — API contract (작업지시서 3.3 / 7.1).

검증: 인증, request_id 트레이싱 헤더, 표준 error_code, 핵심 도메인 엔드포인트 계약,
AKI 분석 응답 스키마, 신장내과 타임라인 읽기.
"""


def test_health_has_tracing_headers(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert "X-Request-ID" in r.headers
    assert "X-Response-Time-ms" in r.headers


def test_request_id_is_propagated(client):
    r = client.get("/health", headers={"X-Request-ID": "trace-xyz"})
    assert r.headers["X-Request-ID"] == "trace-xyz"


def test_unauthenticated_returns_standard_error_code(client):
    r = client.get("/api/patients")
    assert r.status_code == 401
    body = r.json()
    assert body["error_code"] == "UNAUTHENTICATED"
    assert "request_id" in body


def test_login_and_me(client, neph_headers):
    r = client.get("/api/auth/me", headers=neph_headers)
    assert r.status_code == 200
    assert r.json()["role"] in {"nephrology", "admin"}


def test_list_patients_contract(client, neph_headers):
    r = client.get("/api/patients", headers=neph_headers)
    assert r.status_code == 200
    rows = r.json()
    assert isinstance(rows, list) and rows
    assert {"id", "name"} <= set(rows[0])


def test_analyze_features_response_schema(client, neph_headers):
    r = client.post(
        "/api/nephrology/aki/analyze",
        headers=neph_headers,
        json={"creatinine_max": 3.0, "baseline_creatinine": 1.0, "urine_ml_kg_hr": 0.2},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # API 는 camelCase 별칭으로 직렬화.
    assert {"risk", "riskScore", "stage", "rationale"} <= set(body)
    assert 0 <= body["riskScore"] <= 100


def test_not_found_patient_returns_404_code(client, neph_headers):
    r = client.post(
        "/api/nephrology/aki/analyze/does-not-exist", headers=neph_headers
    )
    assert r.status_code == 404
    assert r.json()["error_code"] == "NOT_FOUND"


def test_nephrology_timeline_read_only(client, neph_headers):
    # 먼저 분석을 돌려 환자 컨텍스트 확보(고위험이면 AI_ALERT 적재).
    pid = client.get("/api/patients", headers=neph_headers).json()[0]["id"]
    r = client.get(f"/api/nephrology/timeline/{pid}", headers=neph_headers)
    assert r.status_code == 200
    assert isinstance(r.json(), list)
