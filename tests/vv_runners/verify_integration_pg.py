"""PostgreSQL 백엔드 통합 검증 — 로그인/회원가입 + 4개 역할 데이터 연결.

backend/.env 의 DATABASE_URL(PostgreSQL)로 앱을 부팅(init_db + 멱등 시드)하고,
TestClient 로 역할별(admin/emergency/nephrology/pathology) 주요 화면 데이터 API 를
호출해 연결을 검증한다.

실행(리포 루트):
  backend\\.venv\\Scripts\\python.exe tests\\vv_runners\\verify_integration_pg.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent.parent / "backend"
sys.path.insert(0, str(BACKEND))

from core.config import settings  # noqa: E402

ACCOUNTS = {
    "admin": ("admin", "Admin2026!"),
    "emergency": ("er_kim", "Emer2026!"),
    "nephrology": ("neph_hong", "Neph2026!"),
    "pathology": ("path_lee", "Path2026!"),
}

OK, FAIL = "OK", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, cond: bool, detail: str = ""):
    results.append((OK if cond else FAIL, name, detail))
    mark = "[OK]" if cond else "[!!]"
    print(f"  {mark} {name}  {detail}")


def main() -> int:
    print(f"DATABASE_URL: {settings.database_url}")
    if not settings.database_url.startswith("postgresql"):
        print("[ERROR] PostgreSQL 이 아닙니다. backend/.env 확인.")
        return 2

    from fastapi.testclient import TestClient

    from main import app

    with TestClient(app) as c:  # lifespan: init_db + 멱등 시드
        # ---- 헬스 ----
        check("health", c.get("/health").json().get("status") == "ok")

        tokens = {}
        # ---- 로그인(4개 역할) ----
        print("\n[로그인]")
        for role, (u, p) in ACCOUNTS.items():
            r = c.post("/api/auth/login", json={"username": u, "password": p})
            ok = r.status_code == 200 and r.json().get("user", {}).get("role") == role
            tokens[role] = r.json().get("accessToken") if r.status_code == 200 else None
            check(f"login {role} ({u})", ok, f"status={r.status_code}")

        def H(role):
            return {"Authorization": f"Bearer {tokens[role]}"}

        # ---- 회원가입 → 관리자 승인 → 로그인 ----
        print("\n[회원가입/승인 플로우]")
        import uuid
        uname = f"e2e_{uuid.uuid4().hex[:6]}"
        su = c.post("/api/auth/signup", json={"username": uname, "password": "Pw2026!!",
                    "name": "통합테스트", "role": "nephrology", "department": "신장내과"})
        check("signup(201)", su.status_code == 201)
        pend = c.post("/api/auth/login", json={"username": uname, "password": "Pw2026!!"})
        check("pending 로그인 차단(401)", pend.status_code == 401)
        accts = c.get("/api/auth/accounts", headers=H("admin"))
        uid = next((a["id"] for a in accts.json() if a["username"] == uname), None)
        appr = c.patch(f"/api/auth/accounts/{uid}/approval", headers=H("admin"),
                       json={"approval": "approved"})
        check("관리자 승인(200)", appr.status_code == 200)
        ok_login = c.post("/api/auth/login", json={"username": uname, "password": "Pw2026!!"})
        check("승인 후 로그인(200)", ok_login.status_code == 200)

        # ---- 관리자 페이지 데이터 ----
        print("\n[관리자(admin) 데이터]")
        check("계정 목록", len(accts.json()) >= 4, f"{len(accts.json())}명")
        au = c.get("/api/audit", headers=H("admin"))
        check("감사 로그", au.status_code == 200, f"{len(au.json())}건")
        no = c.get("/api/notifications", headers=H("admin"))
        check("알림", no.status_code == 200)

        # ---- 응급의학과 페이지 데이터 ----
        print("\n[응급의학과(emergency) 데이터]")
        beds = c.get("/api/beds", headers=H("emergency"))
        check("병상 보드", beds.status_code == 200 and len(beds.json()) > 0, f"{len(beds.json())}개")
        summ = c.get("/api/beds/summary", headers=H("emergency"))
        check("병상 집계", summ.status_code == 200)
        ep = c.get("/api/beds/emergency-patients", headers=H("emergency"))
        check("응급 환자 목록", ep.status_code == 200, f"{len(ep.json())}명")
        det = c.get("/api/beds/details", headers=H("emergency"))
        check("입실 상세", det.status_code == 200)

        # ---- 신장내과 페이지 데이터 ----
        print("\n[신장내과(nephrology) 데이터]")
        pts = c.get("/api/patients", headers=H("nephrology"))
        check("환자 목록", pts.status_code == 200 and len(pts.json()) > 0, f"{len(pts.json())}명")
        pid = pts.json()[0]["id"]
        aki = c.post(f"/api/nephrology/aki/analyze/{pid}", headers=H("nephrology"))
        check("AKI 분석", aki.status_code == 200, f"risk={aki.json().get('riskScore')}")
        tl = c.get(f"/api/nephrology/timeline/{pid}", headers=H("nephrology"))
        check("환자 타임라인", tl.status_code == 200, f"{len(tl.json())}건")
        draft = c.post("/api/voice/draft", headers=H("nephrology"),
                       json={"patientId": pid, "transcript": "소변이 줄고 다리가 부어요"})
        check("AI 음성초안", draft.status_code == 201,
              f"위험 {draft.json().get('risk',{}).get('tier')}" if draft.status_code == 201 else "")

        # ---- 병리과 페이지 데이터 ----
        print("\n[병리과(pathology) 데이터]")
        path = c.get("/api/pathology", headers=H("pathology"))
        check("병리 결과 목록", path.status_code == 200, f"{len(path.json())}건")
        cons = c.get("/api/consultations", headers=H("pathology"))
        check("협진 목록", cons.status_code == 200, f"{len(cons.json())}건")

    # ---- 요약 ----
    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    print("\n" + "=" * 50)
    print(f"통합 검증: {len(results) - n_fail}/{len(results)} OK, 실패 {n_fail}")
    print("=" * 50)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
