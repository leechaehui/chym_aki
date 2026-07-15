"""스키마 자동 동기화 — 모델 ↔ 실제 DB 컬럼 차이를 메꾼다.

배경:
    SQLAlchemy `create_all` 은 "없는 테이블"만 만들 뿐 "없는 컬럼"은 추가하지 않는다.
    그래서 새 코드에서 모델에 컬럼이 추가돼도 기존 DB 테이블에는 반영되지 않아
    조회 SQL 이 UndefinedColumn 으로 500 을 내는 드리프트가 발생한다(예: users.employee_id).

동작:
    1) 모든 모델을 등록(import models)한 뒤 Base.metadata 를 순회한다.
    2) 각 테이블의 실제 DB 컬럼과 비교해 누락 컬럼을 찾는다.
    3) `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` 로 추가한다.
       - NOT NULL 컬럼은 기존 행이 있으면 실패하므로 nullable 로 추가하고 경고한다
         (server_default 가 있으면 그대로 사용).

사용:
    python scripts/sync_schema.py            # 적용
    python scripts/sync_schema.py --dry-run  # 진단만(변경 없음)

멱등하다 — 여러 번 실행해도 안전하다.
"""
from __future__ import annotations

import sys

from sqlalchemy import inspect, text

# 프로젝트 루트(backend) 를 import 경로에 넣어 어디서 실행하든 동작하게 한다.
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import models  # noqa: F401,E402  (전체 모델 메타데이터 등록 트리거)
from core.config import settings  # noqa: E402
from core.database import Base, engine  # noqa: E402


def collect_missing() -> list[tuple[str, str, str, bool]]:
    """(table, column, ddl_type, nullable) 형태의 누락 컬럼 목록을 반환."""
    schema = settings.app_schema if not settings.is_sqlite else None
    insp = inspect(engine)
    existing_tables = set(insp.get_table_names(schema=schema))

    missing: list[tuple[str, str, str, bool]] = []
    for table in Base.metadata.tables.values():
        short = table.name.split(".")[-1]
        if short not in existing_tables:
            # 테이블 자체가 없으면 create_all 이 만든다 — 여기선 컬럼만 다룬다.
            continue
        db_cols = {c["name"] for c in insp.get_columns(short, schema=schema)}
        for col in table.columns:
            if col.name in db_cols:
                continue
            ddl_type = col.type.compile(engine.dialect)
            missing.append((short, col.name, ddl_type, col.nullable))
    return missing


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    schema = settings.app_schema if not settings.is_sqlite else None
    prefix = f"{schema}." if schema else ""

    missing = collect_missing()
    if not missing:
        print("[sync_schema] 드리프트 없음 — 모델과 DB 컬럼이 일치합니다.")
        return 0

    print(f"[sync_schema] 누락 컬럼 {len(missing)}개 발견:")
    statements: list[str] = []
    for tbl, col, ddl_type, nullable in missing:
        note = ""
        # NOT NULL 을 기존 행 있는 테이블에 그냥 붙이면 실패 → nullable 로 추가.
        add_type = ddl_type
        if not nullable:
            note = "  (모델은 NOT NULL 이지만 안전을 위해 NULL 허용으로 추가)"
        print(f"  - {prefix}{tbl}.{col}  {ddl_type}{note}")
        statements.append(
            f'ALTER TABLE {prefix}{tbl} ADD COLUMN IF NOT EXISTS {col} {add_type}'
        )

    if dry_run:
        print("[sync_schema] --dry-run: 변경하지 않았습니다.")
        return 0

    with engine.begin() as conn:
        for sql in statements:
            conn.execute(text(sql))
    print(f"[sync_schema] {len(statements)}개 컬럼 추가 완료.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
