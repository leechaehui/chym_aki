"""ICU AKI 모니터 서비스 — mimic4 DB 의 ICU 코호트에 학습 모델 적용(DB 라이브).

모델의 실제 입력인 **스케일 완료 피처**(AKI 폴더 *_final.csv = mimic4 전처리 산출물, 42,210명)를
mimic4 DB 의 `chym.icu_features_scaled` 테이블로 1회 적재하고, 그 테이블을 라이브 쿼리한다.
careunit/재원시간 등 메타는 라이브 `chym.cohort`(앱 스키마) 와 조인한다 → 모든 조회가 mimic4 DB.

전처리 재현(transform_info.pkl)은 원본 파이프라인과 불일치(일부 컬럼 미스케일)해 모델 예측을
왜곡하므로, 모델이 학습·검증된 정확한 입력(*_final)을 그대로 쓴다(source=model).

- 모델: ml_models/stage1_LR_full.pkl, stage2_LGBM_v13_full_classweight.pkl
- 예측은 프로세스당 1회 배치 계산 후 캐시 → 요청은 정렬/필터/페이지네이션만.
"""
from __future__ import annotations

import functools
import pickle
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from core.config import settings
from core.database import engine
from core.fake_name import fake_korean_name
from queries.constants import STAGE_LABELS, RISK_TIERS, RISK_LABELS
from queries import monitor_queries as MQ

BACKEND = Path(__file__).resolve().parent.parent
MODELS = BACKEND / "ml_models"
FEATURE_CSV_DIR = BACKEND.parent / "AKI" / "전처리" / "최종 데이터셋_전처리 완료된"
_SCALED_TABLE = "icu_features_scaled"  # chym 스키마


@functools.lru_cache(maxsize=1)
def model_bundles() -> tuple[dict, dict]:
    """학습 모델 번들(Stage1 LR · Stage2 LGBM)을 1회 로드해 캐시.

    각 번들은 {'model', 'feature_cols', ...}. 예측·설명(SHAP)·검증이 동일 객체를 공유한다.
    """
    stage1 = pickle.load(open(MODELS / "stage1_LR_full.pkl", "rb"))
    stage2 = pickle.load(open(MODELS / "stage2_LGBM_v13_full_classweight.pkl", "rb"))
    return stage1, stage2


@functools.lru_cache(maxsize=1)
def _model_feature_cols() -> list[str]:
    stage1, _ = model_bundles()
    return list(stage1["feature_cols"])


def scaled_features_for_stay(stay_id: int) -> dict[str, float] | None:
    """해당 stay 의 모델 입력(스케일 완료) 35 피처값. 없으면 None.

    SHAP 기여도 계산은 모델이 실제로 본 스케일 피처에 대해 이뤄져야 하므로
    chym.icu_features_scaled 의 원본 행을 그대로 돌려준다.
    """
    feature_cols = _model_feature_cols()
    schema = settings.app_schema
    selected = ", ".join(f'"{c}"' for c in feature_cols)
    with engine.connect() as conn:
        row = conn.execute(
            MQ.sql_scaled_features_for_stay(schema, _SCALED_TABLE, selected),
            {"s": stay_id},
        ).mappings().first()
    if row is None:
        return None
    return {c: float(row[c]) for c in feature_cols}


def prediction_for_stay(stay_id: int) -> dict | None:
    """캐시된 코호트 예측표에서 해당 stay 한 행(예측 확률·정답)을 돌려준다. 없으면 None."""
    df = _predictions()
    match = df[df["stay_id"] == stay_id]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def _scaled_table_count() -> int:
    """chym.icu_features_scaled 행 수(없으면 -1)."""
    schema = settings.app_schema
    with engine.connect() as c:
        reg = c.execute(
            MQ.sql_scaled_table_regclass(schema, _SCALED_TABLE),
            {"t": f"{schema}.{_SCALED_TABLE}"},
        ).scalar()
        if reg is None:
            return -1
        return int(c.execute(MQ.sql_scaled_table_count(schema, _SCALED_TABLE)).scalar())


def ensure_scaled_table() -> None:
    """모델 입력(스케일 완료 피처)을 chym.icu_features_scaled 로 1회 적재(멱등).

    train/valid/test_final 3 분할(전체 42,210)을 합쳐 stay_id + 라벨 + 35 모델피처만 적재.
    """
    if _scaled_table_count() > 0:
        return
    feat = _model_feature_cols()
    keep = ["stay_id", "subject_id", "age", "gender", "aki_label", "aki_stage", *feat]
    frames = [pd.read_csv(FEATURE_CSV_DIR / f) for f in
              ("train_final.csv", "valid_final.csv", "test_final.csv")]
    df = pd.concat(frames, ignore_index=True)
    df = df[[c for c in keep if c in df.columns]]
    df.to_sql(_SCALED_TABLE, engine, schema=settings.app_schema,
              if_exists="replace", index=False)


def is_available() -> bool:
    """가용성: 모델 파일 + (적재된 테이블 또는 적재용 CSV) 존재."""
    if not (MODELS / "stage1_LR_full.pkl").exists():
        return False
    try:
        if _scaled_table_count() > 0:
            return True
    except Exception:
        return False
    return (FEATURE_CSV_DIR / "test_final.csv").exists()


_predictions_lock = threading.Lock()

@functools.lru_cache(maxsize=1)
def _predictions_cached() -> pd.DataFrame:
    """전체 ICU 코호트(mimic4 DB)에 학습 모델을 배치 적용한 예측 테이블(프로세스 캐시)."""
    ensure_scaled_table()
    s1, s2 = model_bundles()
    feat = s1["feature_cols"]
    lr, lgbm = s1["model"], s2["model"]
    schema = settings.app_schema

    # 스케일 피처 + 코호트 메타(둘 다 앱 스키마) 조인.
    # age/gender 는 cohort 의 raw 값을 쓴다(스케일 테이블 age 는 표준화돼 있어 부적합).
    df = pd.read_sql(MQ.sql_predictions_join(schema, _SCALED_TABLE), engine)

    # 모델 입력 피처(35개)가 통째로 0/NaN→0 인 환자(측정 결측 대치 아티팩트, 원본의 ~17%)는
    # 예측 확률·위험점수·SHAP 가 전부 0/무의미하게 나온다(입력이 0이라 coef×0=0). 코호트 목록·
    # 요약·검색·카운트 어디에도 노출되지 않도록 예측 단계에서 아예 제외한다.
    feat_present = [c for c in feat if c in df.columns]
    if feat_present:
        df = df[(df[feat_present].fillna(0) != 0).any(axis=1)].reset_index(drop=True)

    X = df[feat]
    # LR 은 numpy(학습 시 numpy), LGBM 은 피처명 DataFrame(학습 시 DataFrame) 으로 입력.
    p_aki = lr.predict_proba(X.to_numpy(dtype=float))[:, 1]
    p_sev = lgbm.predict_proba(X)[:, 1]
    p_non = 1.0 - p_aki
    p_s1 = p_aki * (1.0 - p_sev)
    p_s23 = p_aki * p_sev
    pred = np.argmax(np.column_stack([p_non, p_s1, p_s23]), axis=1)
    risk = np.minimum(100.0, p_s1 * 60.0 + p_s23 * 100.0)

    # AKI 발생 위험 예측 시각 = ICU 입실(icu_intime) + 48h 윈도우가 닫히는 시점(모델 입력 48h 기준).
    # 시:분만 표시(MIMIC 날짜는 비식별화돼 연도가 의미 없음). 결측이면 None.
    predict_at = (
        pd.to_datetime(df["icu_intime"], errors="coerce") + pd.Timedelta(hours=48)
    ).dt.strftime("%H:%M")

    out = pd.DataFrame({
        "stay_id": df["stay_id"],
        "subject_id": df.get("subject_id"),
        # 합성 표시명(프론트 fakeKoreanName 과 동일) — 이름 검색 매칭용.
        "display_name": df["subject_id"].map(fake_korean_name),
        "age": pd.to_numeric(df["cohort_age"], errors="coerce").astype("Int64"),
        "gender": df["cohort_gender"],
        "careunit": df["first_careunit"],
        "icu_los_hours": pd.to_numeric(df["icu_los_hours"], errors="coerce").round(1),
        "predict_at": predict_at,
        "p_aki": p_aki, "p_non_aki": p_non, "p_stage1": p_s1, "p_stage2_plus": p_s23,
        "pred": pred, "risk_score": np.rint(risk).astype(int),
        "actual_label": df["aki_label"].astype(int),
        "actual_stage": df["aki_stage"].astype(int),
    })
    return out

def _predictions() -> pd.DataFrame:
    """Thread-safe wrapper for the cached predictions to prevent concurrent cache misses."""
    with _predictions_lock:
        return _predictions_cached()

# 상수는 queries.constants 에서 import (RISK_LABELS, RISK_TIERS)


class IcuMonitorService:
    """ICU 코호트 AKI 위험 조회(읽기 전용). 영속 상태 없음."""

    def summary(self) -> dict:
        df = _predictions()
        n = len(df)
        rs = df["risk_score"]
        return {
            "total": int(n),
            # 모델 예측 분류(pred): 2=Stage2-3, 1=Stage1, 0=Non-AKI.
            "high": int((df["pred"] == 2).sum()),
            "moderate": int((df["pred"] == 1).sum()),
            "low": int((df["pred"] == 0).sum()),
            # AI 위험점수 밴드(전 화면 배지와 동일 기준): 고위험 ≥60 · 중등도 30–60 · 안정 <30.
            "risk_high": int((rs >= 60).sum()),
            "risk_moderate": int(((rs >= 30) & (rs < 60)).sum()),
            "risk_low": int((rs < 30).sum()),
            "n_careunits": int(df["careunit"].nunique()),
        }

    def list_patients(
        self, *, limit: int = 30, offset: int = 0, careunit: str | None = None,
        min_risk: int = 0,
    ) -> list[dict]:
        df = _predictions()
        if careunit:
            # careunit 값(예: "Medical Intensive Care Unit (MICU)")에는 괄호가 포함된다.
            # str.contains 는 정규식이라 "(MICU)" 를 캡처그룹으로 해석해 매칭이 깨진다 →
            # 칩이 보내는 값은 정확한 careunit 전체명이므로 정확 일치로 필터한다.
            df = df[df["careunit"] == careunit]
        if min_risk:
            df = df[df["risk_score"] >= min_risk]
        # 1순위: 모델 등급(pred, Stage2-3>Stage1>Non-AKI) 버킷, 2순위: 버킷 내 위험점수.
        # risk_score 단독 정렬이면 "Stage1인데 Stage2-3보다 점수가 높아 위에 옴" 같은
        # 등급-표시 역전이 생겨서(두 값이 서로 다른 공식이라 애매한 케이스에서 어긋남), 등급을 우선한다.
        df = df.sort_values(["pred", "risk_score"], ascending=[False, False]).iloc[offset : offset + limit]
        return [self._to_out(r) for _, r in df.iterrows()]

    def careunits(self) -> list[dict]:
        df = _predictions()
        g = df.groupby("careunit").agg(
            n=("stay_id", "size"), high=("pred", lambda s: int((s == 2).sum()))
        ).reset_index().sort_values("n", ascending=False)
        return [{"careunit": r.careunit, "n": int(r.n), "high": int(r.high)} for r in g.itertuples()]

    def _to_out(self, r) -> dict:
        pred = int(r["pred"])
        is_new_patient = _is_new_patient(int(r["stay_id"]))
        return {
            "stay_id": int(r["stay_id"]),
            "subject_id": int(r["subject_id"]),
            "age": int(r["age"]) if pd.notna(r["age"]) else None,
            "gender": r["gender"],
            "careunit": r["careunit"],
            "icu_los_hours": float(r["icu_los_hours"]) if pd.notna(r["icu_los_hours"]) else None,
            "predict_at": r["predict_at"] if pd.notna(r["predict_at"]) else None,
            "risk_score": int(r["risk_score"]),
            "risk": RISK_TIERS[pred],
            "risk_label": RISK_LABELS[pred],
            "stage": STAGE_LABELS[pred],
            "stage_num": pred,
            "p_non_aki": round(float(r["p_non_aki"]), 4),
            "p_stage1": round(float(r["p_stage1"]), 4),
            "p_stage2_plus": round(float(r["p_stage2_plus"]), 4),
            "is_new_patient": is_new_patient,
            "source": "model",
            "actual_label": int(r["actual_label"]),
            "actual_stage": int(r["actual_stage"]),
        }


def _viewed_new_patient_path() -> Path:
    return BACKEND / "data" / "viewed_new_patients.txt"


def _is_new_patient(stay_id: int) -> bool:
    """Return the recent-24h new marker state for the monitoring list."""
    if stay_id % 72 >= 24:
        return False
    path = _viewed_new_patient_path()
    if not path.exists():
        return True
    return str(stay_id) not in set(path.read_text().splitlines())


def mark_patient_viewed(stay_id: int) -> None:
    """Remove the new marker after a detail view."""
    path = _viewed_new_patient_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = set(path.read_text().splitlines()) if path.exists() else set()
    existing.add(str(stay_id))
    path.write_text("\n".join(sorted(existing)) + "\n")
