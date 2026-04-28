"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
kidney_nlp/import_nlp_csv.py  —  NLP CSV → PostgreSQL 임포트 스크립트
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

▶ 역할
  NLP CSV 파일 두 개를 PostgreSQL 스테이징 테이블에 적재한다.
  psql \\COPY보다 유연하며, chym_proj/config.py 연결 설정을 자동 사용한다.

▶ 파일 위치 (chym_aki 프로젝트 기준)
  kidney_nlp/import_nlp_csv.py            ← 이 파일
  kidney_nlp/nlp_keyword_features.csv     ← 업로드된 NLP 키워드 CSV
  kidney_nlp/radiology_nlp_text.csv       ← 업로드된 방사선 원문 CSV

▶ 실행 방법
  cd chym_aki
  python kidney_nlp/import_nlp_csv.py

▶ 실행 후 다음 스크립트 실행
  psql -f Scripts/03_features_nlp_(트랙D).sql
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import sys
import time
import logging

import pandas as pd
from sqlalchemy import create_engine, text

# chym_proj를 sys.path에 추가 (config.py import를 위해)
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHYM_PROJ = os.path.join(PROJ_ROOT, "chym_proj")
if CHYM_PROJ not in sys.path:
    sys.path.insert(0, CHYM_PROJ)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("nlp_import")


# ─────────────────────────────────────────────────────────────────────────────
# DB 연결 (config.py 자동 탐지)
# ─────────────────────────────────────────────────────────────────────────────

def get_database_url() -> str:
    """config.py 또는 환경변수에서 DB URL을 가져온다."""
    try:
        import importlib
        config = importlib.import_module("config")
        if hasattr(config, "DATABASE_URL"):
            return config.DATABASE_URL
        if hasattr(config, "DB_HOST"):
            h = config.DB_HOST
            p = getattr(config, "DB_PORT", 5432)
            n = getattr(config, "DB_NAME", "mimiciv")
            u = getattr(config, "DB_USER", "postgres")
            pw= getattr(config, "DB_PASSWORD", "")
            return f"postgresql+psycopg2://{u}:{pw}@{h}:{p}/{n}"
    except ModuleNotFoundError:
        pass
    return os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/mimiciv")


# ─────────────────────────────────────────────────────────────────────────────
# 임포트 함수
# ─────────────────────────────────────────────────────────────────────────────

def import_nlp_keyword_features(engine, csv_path: str, chunksize: int = 5000) -> None:
    """nlp_keyword_features.csv → aki_project.stg_nlp_keyword_features 적재.

    stg_ 테이블은 SQL(STEP 0-A)에서 이미 생성되어 있어야 한다.
    기존 데이터를 TRUNCATE하고 재적재한다 (멱등성 보장).

    Args:
        engine:    SQLAlchemy 엔진
        csv_path:  nlp_keyword_features.csv 경로
        chunksize: 한 번에 삽입할 행 수 (기본 5,000)
    """
    if not os.path.exists(csv_path):
        logger.error(f"파일 없음: {csv_path}")
        return

    logger.info(f"[NLP 키워드] 로드 중: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"  shape: {df.shape}  컬럼: {list(df.columns)}")

    # NaN 처리: kw_ 컬럼은 0으로, 텍스트는 빈 문자열로
    kw_cols = [c for c in df.columns if c.startswith("kw_")]
    df[kw_cols] = df[kw_cols].fillna(0).astype(int)
    if "nlp_text_combined" in df.columns:
        df["nlp_text_combined"] = df["nlp_text_combined"].fillna("")

    logger.info("  기존 데이터 삭제 중 (TRUNCATE)...")
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE aki_project.stg_nlp_keyword_features"))
        conn.commit()

    logger.info(f"  삽입 중 ({len(df):,}행, chunksize={chunksize}) ...")
    t0 = time.time()
    df.to_sql(
        name="stg_nlp_keyword_features",
        con=engine,
        schema="aki_project",
        if_exists="append",   # TRUNCATE 후 append
        index=False,
        chunksize=chunksize,
        method="multi",       # 배치 INSERT로 성능 향상
    )
    logger.info(f"  ✅ 완료: {len(df):,}행, {time.time()-t0:.1f}초")


def import_radiology_nlp_text(engine, csv_path: str, chunksize: int = 2000) -> None:
    """radiology_nlp_text.csv → aki_project.stg_radiology_nlp_text 적재.

    파일이 약 35 MB이므로 chunksize를 작게 설정해 메모리를 절약한다.
    charttime 파싱 오류를 처리하기 위해 parse_dates 옵션을 사용한다.

    Args:
        engine:    SQLAlchemy 엔진
        csv_path:  radiology_nlp_text.csv 경로
        chunksize: 한 번에 삽입할 행 수 (기본 2,000)
    """
    if not os.path.exists(csv_path):
        logger.error(f"파일 없음: {csv_path}")
        return

    logger.info(f"[방사선 원문] 로드 중: {csv_path}")

    logger.info("  기존 데이터 삭제 중 (TRUNCATE)...")
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE aki_project.stg_radiology_nlp_text"))
        conn.commit()

    # 청크 단위로 읽어서 메모리 절약 (전체 35 MB)
    total_rows = 0
    t0 = time.time()
    for chunk_df in pd.read_csv(csv_path, chunksize=chunksize,
                                 parse_dates=["charttime"]):
        # 컬럼 정리
        chunk_df["findings"]   = chunk_df["findings"].fillna("")
        chunk_df["impression"] = chunk_df["impression"].fillna("")

        # stay_id가 없는 행 처리 (ICU 연결 전 note)
        chunk_df["stay_id"]   = pd.to_numeric(chunk_df["stay_id"],   errors="coerce")
        chunk_df["hadm_id"]   = pd.to_numeric(chunk_df["hadm_id"],   errors="coerce")
        chunk_df["subject_id"]= pd.to_numeric(chunk_df["subject_id"],errors="coerce")

        chunk_df.to_sql(
            name="stg_radiology_nlp_text",
            con=engine,
            schema="aki_project",
            if_exists="append",
            index=False,
            chunksize=500,
            method="multi",
        )
        total_rows += len(chunk_df)
        logger.info(f"  진행: {total_rows:,}행 적재됨 ...")

    logger.info(f"  ✅ 완료: {total_rows:,}행, {time.time()-t0:.1f}초")


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """NLP CSV 임포트 전체 실행."""
    db_url = get_database_url()
    logger.info(f"[DB] 연결: {db_url.split('@')[-1]}")  # 비밀번호 숨김

    engine = create_engine(db_url, pool_pre_ping=True)

    # 현재 파일 위치 기준으로 CSV 경로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))

    import_nlp_keyword_features(
        engine,
        csv_path=os.path.join(base_dir, "nlp_keyword_features.csv"),
    )

    import_radiology_nlp_text(
        engine,
        csv_path=os.path.join(base_dir, "radiology_nlp_text.csv"),
    )

    logger.info("\n[완료] 다음 단계:")
    logger.info("  psql -f Scripts/03_features_nlp_(트랙D).sql")


if __name__ == "__main__":
    main()