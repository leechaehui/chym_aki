"""pytest 공통 픽스처 (V&V 테스트 부트스트랩).

- 임시 SQLite DB 로 격리(실데이터/운영 DB 미오염).
- 앱 임포트 이전에 환경변수를 설정해 settings 싱글턴이 테스트 설정으로 고정되게 한다.
- 시드 데이터 + 인증 토큰 헬퍼 제공.
"""
import os
import sys
import tempfile
import uuid
from pathlib import Path

import pytest

# tests/ 는 리포 루트에 있고, 앱 코드는 backend/ 에 있다.
BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND))

# 앱(core.config) 임포트 전에 환경 고정 — lru_cache 된 settings 가 이 값을 읽는다.
_DB = Path(tempfile.gettempdir()) / f"chym_test_{uuid.uuid4().hex[:8]}.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_DB.as_posix()}"
os.environ["SEED_ON_STARTUP"] = "false"
os.environ["APP_ENV"] = "test"
# 테스트는 STT 모델(whisper) 로드 없이 빠르고 격리되게 — passthrough 강제.
os.environ["STT_STRATEGY"] = "passthrough"

# 데모 계정(README 기준) — 역할별 RBAC 테스트용.
ADMIN = ("admin", "Admin2026!")          # role: admin
NEPH = ("neph_hong", "Neph2026!")        # role: nephrology
PATH = ("path_lee", "Path2026!")         # role: pathology


@pytest.fixture(scope="session")
def client():
    """시드된 임시 DB 위에서 동작하는 TestClient(세션 스코프)."""
    from fastapi.testclient import TestClient

    from core.database import SessionLocal, init_db
    from db.seed import seed

    init_db()
    with SessionLocal() as db:
        from sqlalchemy import select

        from models.patient import Patient

        if not db.execute(select(Patient).limit(1)).first():
            seed()

    from main import app

    with TestClient(app) as c:
        yield c

    if _DB.exists():
        try:
            _DB.unlink()
        except OSError:
            pass


def _token(client, username, password) -> str:
    r = client.post("/api/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    # API 는 camelCase 별칭으로 직렬화(accessToken).
    return r.json()["accessToken"]


@pytest.fixture(scope="session")
def admin_headers(client):
    return {"Authorization": f"Bearer {_token(client, *ADMIN)}"}


@pytest.fixture(scope="session")
def neph_headers(client):
    return {"Authorization": f"Bearer {_token(client, *NEPH)}"}


@pytest.fixture(scope="session")
def path_headers(client):
    return {"Authorization": f"Bearer {_token(client, *PATH)}"}


@pytest.fixture()
def db(client):
    """함수 스코프 DB 세션(서비스 단위 테스트용).

    client 픽스처에 의존해 테이블/시드가 준비된 상태를 보장한다.
    서비스는 내부에서 commit 하므로, 각 테스트는 factories 로 고유 엔티티를
    새로 만들어 상호 격리한다(공유 시드 행을 변형하지 않는다).
    """
    from core.database import SessionLocal

    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
