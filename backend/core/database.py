"""데이터베이스 엔진/세션 구성.

책임: SQLAlchemy 엔진·세션 팩토리·Declarative Base 제공 + 세션 의존성.
- 비즈니스 로직은 여기 두지 않는다(연결 관리 전용).
- SQLite/PostgreSQL 양쪽을 동일 인터페이스로 지원한다.
"""
from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from core.config import settings


class Base(DeclarativeBase):
    """모든 ORM 모델의 공통 베이스."""


# 연결 인자:
# - SQLite: 동일 스레드 제약 완화(FastAPI 스레드풀).
# - PostgreSQL(mimic4): libpq options 로 search_path 를 app_schema(chym) **단독**으로 고정.
#   앱 테이블과 MIMIC 파생 임상 테이블(cohort/cr_timeseries 등)이 모두 chym 에 통합됐고
#   public 은 비어 있어 더 이상 검색경로에 두지 않는다(pg_catalog 는 항상 암묵 검색).
#   (트랜잭션 롤백에도 유지되도록 SET 문이 아닌 연결 옵션으로 설정.)
if settings.is_sqlite:
    _connect_args: dict = {"check_same_thread": False}
else:
    _connect_args = {"options": f"-c search_path={settings.app_schema}"}

engine = create_engine(
    settings.database_url,
    connect_args=_connect_args,
    # 운영(PG)에서는 커넥션 풀 사전 점검으로 끊긴 커넥션 방지.
    pool_pre_ping=not settings.is_sqlite,
    echo=False,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db() -> Generator[Session, None, None]:
    """요청 단위 DB 세션 의존성.

    트랜잭션 commit/rollback 책임은 Service 레이어에 있다.
    이 의존성은 세션 수명(요청 종료 시 close)만 보장한다.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """테이블 생성. 모든 모델을 import 한 뒤 호출되어야 한다(메타데이터 등록).

    PostgreSQL 에서는 앱 전용 스키마(app_schema)를 먼저 만들고, search_path 가
    그 스키마를 우선하므로 create_all 이 앱 테이블을 거기에 생성한다.
    MIMIC 파생 테이블(cohort/cr_timeseries 등, 같은 app_schema 에 적재)은 ORM 모델이
    아니므로 create_all 이 건드리지 않는다.
    """
    import models  # noqa: F401  (모델 등록 트리거)

    if not settings.is_sqlite:
        with engine.begin() as conn:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {settings.app_schema}"))

    Base.metadata.create_all(bind=engine)
