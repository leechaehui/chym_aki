"""
final_features_48h CSV → chym.final_features_48h 적재 스크립트.

Phase 1: CSV 사전 검증 (DB 적재 전)
Phase 2: DB 적재 (CREATE TABLE + INSERT)
Phase 3: 인덱스 생성
Phase 4: 적재 후 검증 (Row Count · 중복 · NULL · CSV↔DB 비교)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text, create_engine

# ── 설정 ─────────────────────────────────────────
CSV_PATH = Path(__file__).resolve().parent.parent.parent / "AKI" / "feature" / "final_features_48h.csv"


def _resolve_db_url() -> str:
    """접속정보는 코드에 하드코딩하지 않는다 — 환경변수 DATABASE_URL → backend/.env 순으로 읽는다."""
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    env = Path(__file__).resolve().parent.parent / ".env"   # backend/.env
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("DATABASE_URL="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("DATABASE_URL 미설정 — 환경변수 또는 backend/.env 에 설정하세요.")


DB_URL = _resolve_db_url()
SCHEMA = "chym"
TABLE = "final_features_48h"
FULL_TABLE = f"{SCHEMA}.{TABLE}"

# 주요 검사 대상 컬럼 (raw 범위 확인용)
KEY_COLS = ["bun_max", "sodium_min", "potassium_max", "bicarbonate_min"]
# Raw 범위 기대값 (의료 단위) — Scaled 값(-2~+2 수준)과 구분하기 위한 최소 검증
RAW_RANGES = {
    "bun_max":          (1, 300),      # mg/dL
    "sodium_min":       (100, 180),    # mmol/L
    "potassium_max":    (1, 15),       # mmol/L
    "bicarbonate_min":  (1, 60),       # mmol/L
}

# 컬럼별 PostgreSQL 타입
COL_TYPES = {
    "stay_id":              "BIGINT",
    "subject_id":           "BIGINT",
    "hadm_id":              "BIGINT",
    "age":                  "INTEGER",
    "gender":               "VARCHAR(10)",
    "aki_label":            "INTEGER",
    "aki_stage":            "INTEGER",
    "aki_onset_time":       "TIMESTAMP",
    "prediction_cutoff":    "TIMESTAMP",
    "index_time":           "TIMESTAMP",
    "map_mean":             "DOUBLE PRECISION",
    "map_min":              "DOUBLE PRECISION",
    "map_below65_hours":    "DOUBLE PRECISION",
    "sbp_min":              "DOUBLE PRECISION",
    "sbp_mean":             "DOUBLE PRECISION",
    "shock_index_mean":     "DOUBLE PRECISION",
    "hr_max":               "DOUBLE PRECISION",
    "hr_mean":              "DOUBLE PRECISION",
    "rr_max":               "DOUBLE PRECISION",
    "rr_mean":              "DOUBLE PRECISION",
    "temp_max":             "DOUBLE PRECISION",
    "temp_mean":            "DOUBLE PRECISION",
    "urine_output_sum":     "DOUBLE PRECISION",
    "urine_output_6h":      "DOUBLE PRECISION",
    "urine_ml_kg_hr":       "DOUBLE PRECISION",
    "oliguria_flag":        "INTEGER",
    "creatinine_min":       "DOUBLE PRECISION",
    "creatinine_max":       "DOUBLE PRECISION",
    "creatinine_delta":     "DOUBLE PRECISION",
    "bun_max":              "DOUBLE PRECISION",
    "bun_cr_ratio":         "DOUBLE PRECISION",
    "lactate_max":          "DOUBLE PRECISION",
    "lactate_mean":         "DOUBLE PRECISION",
    "vasopressor_flag":     "INTEGER",
    "vasopressor_hours":    "DOUBLE PRECISION",
    "norepi_dose_max":      "DOUBLE PRECISION",
    "potassium_max":        "DOUBLE PRECISION",
    "potassium_mean":       "DOUBLE PRECISION",
    "bicarbonate_min":      "DOUBLE PRECISION",
    "bicarbonate_mean":     "DOUBLE PRECISION",
    "sodium_min":           "DOUBLE PRECISION",
    "sodium_max":           "DOUBLE PRECISION",
    "hemoglobin_min":       "DOUBLE PRECISION",
    "hemoglobin_mean":      "DOUBLE PRECISION",
    "spo2_min":             "DOUBLE PRECISION",
    "spo2_mean":            "DOUBLE PRECISION",
}

SAMPLE_N = 10  # CSV vs DB 비교 샘플 수


def _hr():
    print("─" * 70)


def phase1_validate_csv(df: pd.DataFrame) -> bool:
    """Phase 1: CSV 사전 검증."""
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  Phase 1: CSV 사전 검증                                ║")
    print("╚══════════════════════════════════════════════════════════╝")

    ok = True

    # 기본 통계
    _hr()
    print(f"  파일     : {CSV_PATH}")
    print(f"  행 수    : {len(df):,}")
    print(f"  컬럼 수  : {len(df.columns)}")
    print(f"  컬럼 목록: {list(df.columns)}")

    # stay_id 중복
    _hr()
    dup = df[df.duplicated(subset=["stay_id"], keep=False)]
    if len(dup):
        print(f"  ⚠ stay_id 중복 발견: {len(dup)}건")
        print(dup[["stay_id"]].value_counts().head(10))
        ok = False
    else:
        print(f"  ✓ stay_id 중복 없음 (유니크 {df['stay_id'].nunique():,}개)")

    # 주요 컬럼 NULL 비율
    _hr()
    print("  주요 컬럼 NULL 비율:")
    for col in KEY_COLS:
        if col in df.columns:
            null_n = df[col].isna().sum()
            null_pct = null_n / len(df) * 100
            mark = "✓" if null_pct < 30 else "⚠"
            print(f"    {mark} {col:25s} : {null_n:6,} / {len(df):,}  ({null_pct:.1f}%)")
        else:
            print(f"    ✗ {col} — 컬럼 없음!")
            ok = False

    # Raw 값 범위 확인 (Scaled 혼입 방지)
    _hr()
    print("  값 범위 확인 (Scaled 혼입 방지):")
    for col, (lo, hi) in RAW_RANGES.items():
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if s.empty:
            print(f"    — {col}: 전부 NULL (검증 불가)")
            continue
        vmin, vmax = s.min(), s.max()
        in_range = lo <= vmin and vmax <= hi
        mark = "✓" if in_range else "⚠"
        print(f"    {mark} {col:25s} : min={vmin:.2f}  max={vmax:.2f}  (기대: {lo}–{hi})")
        if not in_range:
            print(f"      → 범위 초과 값이 있으나, MIMIC 특이 케이스일 수 있음. 데이터 자체는 Raw.")

    _hr()
    if ok:
        print("  ★ CSV 사전 검증 통과 — 적재 진행 가능")
    else:
        print("  ★ 경고 항목 있음 — 내용 확인 후 적재 판단 필요")
    return ok


def phase2_load_to_db(df: pd.DataFrame, engine):
    """Phase 2: DB 적재."""
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  Phase 2: DB 적재                                      ║")
    print("╚══════════════════════════════════════════════════════════╝")

    with engine.begin() as conn:
        # 스키마 확인
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))

        # 기존 테이블 DROP
        conn.execute(text(f"DROP TABLE IF EXISTS {FULL_TABLE} CASCADE"))
        print(f"  ✓ DROP TABLE IF EXISTS {FULL_TABLE}")

        # CREATE TABLE
        col_defs = ",\n    ".join(f"{c} {t}" for c, t in COL_TYPES.items())
        ddl = f"CREATE TABLE {FULL_TABLE} (\n    {col_defs}\n)"
        conn.execute(text(ddl))
        print(f"  ✓ CREATE TABLE {FULL_TABLE} (46 columns)")

    # pandas to_sql (append 모드 — 테이블은 이미 생성됨)
    # timestamp 컬럼 변환
    for ts_col in ["aki_onset_time", "prediction_cutoff", "index_time"]:
        if ts_col in df.columns:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

    # NaN → None 변환 (integer 컬럼)
    int_cols = ["aki_label", "aki_stage", "oliguria_flag", "vasopressor_flag"]
    for ic in int_cols:
        if ic in df.columns:
            df[ic] = df[ic].astype("Int64")  # nullable integer

    rows = df.to_sql(
        TABLE,
        engine,
        schema=SCHEMA,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000,
    )
    print(f"  ✓ 데이터 적재 완료: {len(df):,}행")


def phase3_create_index(engine):
    """Phase 3: 인덱스 생성."""
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  Phase 3: 인덱스 생성                                  ║")
    print("╚══════════════════════════════════════════════════════════╝")

    idx_name = "idx_final_features_48h_stay_id"
    with engine.begin() as conn:
        conn.execute(text(f"DROP INDEX IF EXISTS {SCHEMA}.{idx_name}"))
        conn.execute(text(
            f"CREATE INDEX {idx_name} ON {FULL_TABLE}(stay_id)"
        ))
    print(f"  ✓ {idx_name} 생성 완료")


def phase4_verify(df_csv: pd.DataFrame, engine):
    """Phase 4: 적재 후 검증."""
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  Phase 4: 적재 후 검증                                 ║")
    print("╚══════════════════════════════════════════════════════════╝")

    with engine.connect() as conn:
        # Row Count
        _hr()
        db_count = conn.execute(text(f"SELECT COUNT(*) FROM {FULL_TABLE}")).scalar()
        csv_count = len(df_csv)
        match = "✓" if db_count == csv_count else "✗"
        print(f"  {match} Row Count — CSV: {csv_count:,}  |  DB: {db_count:,}")

        # stay_id 중복
        _hr()
        dup_rows = conn.execute(text(
            f"SELECT stay_id, COUNT(*) AS cnt FROM {FULL_TABLE} "
            f"GROUP BY stay_id HAVING COUNT(*) > 1"
        )).fetchall()
        if dup_rows:
            print(f"  ⚠ DB 중복 stay_id: {len(dup_rows)}개")
            for r in dup_rows[:5]:
                print(f"    stay_id={r[0]}, count={r[1]}")
        else:
            print(f"  ✓ DB 중복 stay_id 없음")

        # NULL 비율 (DB)
        _hr()
        print("  DB 주요 컬럼 NULL 비율:")
        for col in KEY_COLS:
            null_n = conn.execute(text(
                f"SELECT COUNT(*) FROM {FULL_TABLE} WHERE {col} IS NULL"
            )).scalar()
            null_pct = null_n / db_count * 100 if db_count else 0
            print(f"    {col:25s} : {null_n:6,} / {db_count:,}  ({null_pct:.1f}%)")

        # 샘플 10개 CSV vs DB 비교
        _hr()
        print(f"  샘플 {SAMPLE_N}개 stay_id CSV↔DB 비교:")
        sample_ids = df_csv["stay_id"].sample(n=min(SAMPLE_N, len(df_csv)), random_state=42).tolist()

        compare_cols = ["stay_id"] + KEY_COLS
        csv_sample = df_csv[df_csv["stay_id"].isin(sample_ids)][compare_cols].set_index("stay_id")

        db_sample_df = pd.read_sql(
            text(f"SELECT {', '.join(compare_cols)} FROM {FULL_TABLE} WHERE stay_id = ANY(:ids)"),
            conn,
            params={"ids": sample_ids},
        ).set_index("stay_id")

        all_match = True
        for sid in sample_ids:
            if sid not in db_sample_df.index:
                print(f"    ✗ stay_id={sid} — DB에 없음!")
                all_match = False
                continue
            csv_row = csv_sample.loc[sid]
            db_row = db_sample_df.loc[sid]
            row_ok = True
            for c in KEY_COLS:
                cv = csv_row[c] if pd.notna(csv_row[c]) else None
                dv = db_row[c] if pd.notna(db_row[c]) else None
                if cv is None and dv is None:
                    continue
                if cv is not None and dv is not None and abs(float(cv) - float(dv)) < 0.001:
                    continue
                row_ok = False
            mark = "✓" if row_ok else "✗"
            if not row_ok:
                all_match = False
            print(f"    {mark} stay_id={sid}")
            if not row_ok:
                for c in KEY_COLS:
                    print(f"        {c}: CSV={csv_row[c]}  DB={db_row[c]}")

        _hr()
        if all_match and db_count == csv_count:
            print("  ★ 적재 후 검증 모두 통과!")
        else:
            print("  ★ 일부 검증 실패 — 위 항목 확인 필요")


def main():
    print("=" * 70)
    print("  final_features_48h CSV → chym.final_features_48h 적재")
    print("=" * 70)

    # CSV 로드
    if not CSV_PATH.exists():
        print(f"✗ CSV 파일 없음: {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    print(f"\n  CSV 로드 완료: {len(df):,}행 × {len(df.columns)}컬럼\n")

    # Phase 1
    phase1_validate_csv(df)

    # DB 엔진
    engine = create_engine(DB_URL)

    # Phase 2
    phase2_load_to_db(df.copy(), engine)

    # Phase 3
    phase3_create_index(engine)

    # Phase 4
    phase4_verify(df, engine)

    print("\n" + "=" * 70)
    print("  완료")
    print("=" * 70)


if __name__ == "__main__":
    main()
