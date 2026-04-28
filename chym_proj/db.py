"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
chym_proj/db.py  —  DB 연결 & 공통 Pydantic 모델
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

▶ 역할
  config.py의 DB 연결 설정을 읽어 SQLAlchemy 엔진과 세션을 구성한다.
  모든 SCR 백엔드(scr03~07)와 xgb_model/에서 이 파일만 import한다.

▶ config.py 의존
  chym_proj/config.py 에 DB 접속 정보가 있어야 한다.
  지원하는 config.py 패턴 (아래 3가지 중 하나면 자동 탐지):

    패턴 A — DATABASE_URL 문자열:
      DATABASE_URL = "postgresql+psycopg2://user:pw@host:5432/dbname"

    패턴 B — 개별 변수:
      DB_HOST = "localhost"
      DB_PORT = 5432
      DB_NAME = "mimiciv"
      DB_USER = "postgres"
      DB_PASSWORD = "password"

    패턴 C — get_db_url() 함수:
      def get_db_url() -> str:
          return "postgresql+psycopg2://..."

▶ 파일 위치 (chym_aki 프로젝트 기준)
  chym_proj/db.py         ← 이 파일
  chym_proj/config.py     ← DB 설정 (기존 파일, 수정 불필요)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import importlib
import logging
from contextlib import contextmanager
from typing import Optional, Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel

logger = logging.getLogger("aki_cdss.db")


# ─────────────────────────────────────────────────────────────────────────────
# config.py 에서 DATABASE_URL 자동 탐지
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_database_url_from_config() -> str:
    """chym_proj/config.py 에서 DB 연결 문자열을 읽어온다.

    패턴 A·B·C 순서로 탐지하며, 모두 실패하면 환경변수 DATABASE_URL을 사용한다.
    환경변수도 없으면 기본값(localhost)을 반환하고 경고를 출력한다.

    Returns:
        SQLAlchemy 연결 문자열 (postgresql+psycopg2://...)
    """
    # ── 패턴 A: config.DATABASE_URL ────────────────────────────────────────
    try:
        config = importlib.import_module("config")

        if hasattr(config, "DATABASE_URL") and config.DATABASE_URL:
            logger.info("[DB] config.DATABASE_URL 사용")
            return config.DATABASE_URL

        # 패턴 A-2: config.DB_URL (sqlalchemy engine과 함께 정의하는 방식)
        if hasattr(config, "DB_URL") and config.DB_URL:
            logger.info("[DB] config.DB_URL 사용")
            return config.DB_URL

        # ── 패턴 C: config.get_db_url() ────────────────────────────────────
        if hasattr(config, "get_db_url") and callable(config.get_db_url):
            url = config.get_db_url()
            if url:
                logger.info("[DB] config.get_db_url() 사용")
                return url

        # ── 패턴 B: config.DB_HOST / DB_PORT / ... ─────────────────────────
        required = ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"]
        if all(hasattr(config, attr) for attr in required):
            host = getattr(config, "DB_HOST", "localhost")
            port = getattr(config, "DB_PORT", 5432)
            name = getattr(config, "DB_NAME", "mimiciv")
            user = getattr(config, "DB_USER", "postgres")
            pw   = getattr(config, "DB_PASSWORD", "")
            url  = f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{name}"
            logger.info(f"[DB] config.DB_HOST 등 개별 변수 사용 → {host}:{port}/{name}")
            return url

    except ModuleNotFoundError:
        logger.warning("[DB] config.py 를 찾을 수 없음 — 환경변수로 fallback")

    # ── 환경변수 fallback ─────────────────────────────────────────────────
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        logger.info("[DB] 환경변수 DATABASE_URL 사용")
        return env_url

    # ── 최종 기본값 (로컬 개발용) ─────────────────────────────────────────
    default = "postgresql+psycopg2://postgres:postgres@localhost:5432/mimiciv"
    logger.warning(f"[DB] DB 설정을 찾을 수 없음. 기본값 사용: {default}")
    return default


# ─────────────────────────────────────────────────────────────────────────────
# SQLAlchemy 엔진 & 세션 초기화
# ─────────────────────────────────────────────────────────────────────────────

DATABASE_URL = _resolve_database_url_from_config()

engine = create_engine(
    DATABASE_URL,
    pool_size=10,          # 항상 유지할 최소 연결 수
    max_overflow=20,       # pool_size 초과 시 추가로 열 수 있는 연결 수
    pool_pre_ping=True,    # 사용 전 ping으로 죽은 연결 감지 후 교체
    pool_recycle=3600,     # 1시간마다 연결 재생성 (DB 세션 타임아웃 대비)
    echo=False,            # SQL 로깅 필요 시 True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """DB 세션 컨텍스트 매니저.

    with get_db_session() as session: 형태로 사용하거나
    FastAPI의 Depends() 패턴에서 사용한다.

    예외 발생 시 자동 rollback, 정상 종료 시 commit 후 close.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def execute_query(sql: str, params: dict = None) -> list[dict]:
    """SQL 실행 후 결과를 딕셔너리 리스트로 반환하는 공통 헬퍼.

    모든 SCR 백엔드(scr03~07)에서 조회 함수의 마지막 단계로 사용한다.
    결과가 없으면 빈 리스트([])를 반환한다.

    Args:
        sql:    실행할 SQL 문자열 (:param 방식 바인딩)
        params: 바인딩 파라미터 딕셔너리

    Returns:
        [{"컬럼명": 값, ...}, ...] 형태의 딕셔너리 리스트
    """
    with get_db_session() as session:
        result  = session.execute(text(sql), params or {})
        columns = result.keys()
        rows    = result.fetchall()
    return [dict(zip(columns, row)) for row in rows]


# ─────────────────────────────────────────────────────────────────────────────
# 공통 Pydantic 모델
# ─────────────────────────────────────────────────────────────────────────────

class PatientBase(BaseModel):
    """모든 화면(SCR-03~07)에서 공통으로 사용하는 환자 기본 정보."""
    stay_id:        int
    subject_id:     int
    age:            Optional[int]   = None
    gender:         Optional[str]   = None
    first_careunit: Optional[str]   = None
    icu_los_hours:  Optional[float] = None
    aki_label:      Optional[int]   = None
    aki_stage:      Optional[int]   = None

    # 사망 관련 (PatientBase 확장)
    hospital_expire_flag: Optional[int]   = None
    hours_to_death:       Optional[float] = None
    competed_with_death:  Optional[int]   = None

    class Config:
        from_attributes = True


class RiskLevel(BaseModel):
    """위험도 수준 공통 모델 (SCR-03~07 전체 색상 결정에 사용)."""
    level:        str   # "high" | "warning" | "normal"
    label_kr:     str   # "높음" | "주의" | "정상"
    color:        str   # "#ef4444" | "#f97316" | "#6b7280"
    border_color: str


class APIResponse(BaseModel):
    """표준 API 응답 래퍼."""
    success: bool
    data:    Optional[dict] = None
    message: Optional[str]  = None
    error:   Optional[str]  = None


# ─────────────────────────────────────────────────────────────────────────────
# 위험도 분류 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def classify_value_as_risk_level(
    value:             float,
    high_threshold:    Optional[float] = None,
    warning_threshold: Optional[float] = None,
    low_threshold:     Optional[float] = None,
    low_warning:       Optional[float] = None,
) -> RiskLevel:
    """수치값을 임계값과 비교해 위험도(높음/주의/정상)를 반환한다.

    SCR-04 색상 결정 등 모든 화면에서 공통 사용.
    """
    is_high = is_warning = False

    if high_threshold    is not None and value >  high_threshold:    is_high    = True
    elif warning_threshold is not None and value > warning_threshold: is_warning = True
    if low_threshold     is not None and value <  low_threshold:     is_high    = True
    elif low_warning     is not None and value <  low_warning:       is_warning = True

    if is_high:
        return RiskLevel(level="high",    label_kr="높음", color="#ef4444", border_color="#ef4444")
    elif is_warning:
        return RiskLevel(level="warning", label_kr="주의", color="#f97316", border_color="#f97316")
    else:
        return RiskLevel(level="normal",  label_kr="정상", color="#6b7280", border_color="#d1d5db")


# ─────────────────────────────────────────────────────────────────────────────
# 연결 상태 확인 (서버 기동 시 헬스체크용)
# ─────────────────────────────────────────────────────────────────────────────

def check_db_connection() -> bool:
    """DB 연결 상태를 확인한다. main.py lifespan에서 호출."""
    try:
        with get_db_session() as session:
            session.execute(text("SELECT 1"))
        logger.info(f"[DB] 연결 성공: {DATABASE_URL.split('@')[-1]}")  # 비밀번호 숨김
        return True
    except Exception as e:
        logger.error(f"[DB] 연결 실패: {e}")
        return False