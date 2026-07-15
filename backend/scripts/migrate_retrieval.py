"""Retrieval CDSS pgvector 마이그레이션 (idempotent).

동작:
  1) CREATE EXTENSION vector 시도 → 가용하면 pgvector 모드, 아니면 fallback(double precision[]).
  2) chym 스키마에 5개 테이블 생성(create_all, checkfirst).
  3) pgvector 모드면 metadata_vec/mean_vec/concept_vec → vector(16) ALTER + HNSW 인덱스.
  4) 결과 요약 출력.

사용: (team venv) python scripts/migrate_retrieval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text  # noqa: E402

from core.config import settings  # noqa: E402
from core.database import Base, engine  # noqa: E402
import models.retrieval as R  # noqa: E402  (테이블 등록)

VECTOR_COLS = [  # (table, column) — pgvector 업그레이드 대상
    ("prototypes", "metadata_vec"),
    ("prototype_members", "concept_vec"),
    ("concept_distribution", "mean_vec"),
]
TABLES = [
    R.Prototype.__table__, R.WsiMetadata.__table__, R.PrototypeMember.__table__,
    R.RetrievalLog.__table__, R.ConceptDistribution.__table__,
]


def _pgvector_available() -> bool:
    if settings.is_sqlite:
        return False
    try:
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as c:
            c.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[migrate] pgvector 미가용 → fallback(double precision[]). 사유: "
              f"{str(exc).splitlines()[0][:120]}")
        return False


def main() -> None:
    sch = settings.app_schema
    print(f"[migrate] target={'sqlite' if settings.is_sqlite else 'postgres:'+sch}")

    # 스키마 보장(PG)
    if not settings.is_sqlite:
        with engine.connect() as c:
            c.execute(text(f"CREATE SCHEMA IF NOT EXISTS {sch}"))
            c.commit()

    has_vec = _pgvector_available()

    # 테이블 생성(idempotent)
    Base.metadata.create_all(bind=engine, tables=TABLES, checkfirst=True)
    print(f"[migrate] tables ensured: {[t.name for t in TABLES]}")

    if has_vec:
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as c:
            for tbl, col in VECTOR_COLS:
                # double precision[] → vector(16) (데이터 있어도 캐스팅 호환)
                c.execute(text(
                    f"ALTER TABLE {sch}.{tbl} ALTER COLUMN {col} "
                    f"TYPE vector({R.VEC_DIM}) USING {col}::vector({R.VEC_DIM})"))
            # HNSW cosine 인덱스(검색 가속)
            c.execute(text(
                f"CREATE INDEX IF NOT EXISTS ix_proto_meta_hnsw ON {sch}.prototypes "
                f"USING hnsw (metadata_vec vector_cosine_ops)"))
        print("[migrate] pgvector 모드: vector(16) + HNSW 인덱스 적용")
    else:
        print("[migrate] fallback 모드: metadata_vec=double precision[], Python 코사인 검색")
        print("          (pgvector 설치 후 본 스크립트 재실행 시 자동으로 vector(16)+HNSW 전환)")

    # 검증
    with engine.connect() as c:
        if settings.is_sqlite:
            names = [r[0] for r in c.execute(text(
                "SELECT name FROM sqlite_master WHERE type='table'")).all()]
        else:
            names = [r[0] for r in c.execute(text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema=:s"), {"s": sch}).all()]
    present = [t.name for t in TABLES if t.name in names]
    print(f"[migrate] verified present: {present}")
    print("[migrate] done.")


if __name__ == "__main__":
    main()
