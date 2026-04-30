# xgb_model/diagnose.py
# 실행: python diagnose.py
# 목적: 누수 피처 확인 (학습 전에 반드시 실행)

import os
import warnings
import numpy as np
import pandas as pd
import sqlalchemy
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

DB_URI = os.getenv(
    "DATABASE_URL",
    "postgresql://bio4:bio4@localhost:5432/mimic4"
)

def main():
    # ── 1. 데이터 로드 ──────────────────────────────────────────────────
    print("[1] cdss_master_features 로드 중 ...")
    engine = sqlalchemy.create_engine(DB_URI)
    df = pd.read_sql("SELECT * FROM aki_project.cdss_master_features", engine)
    print(f"    {len(df):,}행 × {df.shape[1]}열 로드 완료\n")

    y = df["aki_label"].astype(int)

    # ── 2. 범주형 임시 인코딩 (AUROC 계산용) ────────────────────────────
    df_enc = df.copy()
    for col in ["gender", "first_careunit"]:
        if col in df_enc.columns:
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(
                df_enc[col].fillna("Unknown").astype(str)
            )

    # ── 3. 컬럼별 단독 AUROC 계산 ───────────────────────────────────────
    print("[2] 컬럼별 단독 AUROC 계산 중 ...")
    print("    (0.90 이상이면 누수 의심, 0.10 이하이면 반전 누수)\n")

    results = []
    skip_cols = {"aki_label", "aki_stage", "aki_onset_time",
                 "prediction_cutoff", "effective_cutoff",
                 "stay_id", "subject_id", "hadm_id",
                 "icu_intime", "icu_outtime"}

    for col in df_enc.columns:
        if col in skip_cols:
            continue
        series = df_enc[col]
        if series.dtype == object:
            continue
        filled = series.fillna(-999)
        if filled.nunique() < 2:
            continue
        try:
            auc = roc_auc_score(y, filled)
            results.append({"feature": col, "auroc": round(auc, 4)})
        except Exception:
            pass

    df_result = pd.DataFrame(results).sort_values(
        "auroc", key=lambda x: abs(x - 0.5), ascending=False
    )

    # ── 4. 결과 출력 ────────────────────────────────────────────────────
    print("=" * 55)
    print(f"{'feature':<35} {'auroc':>8}")
    print("=" * 55)

    leakage = []
    for _, row in df_result.iterrows():
        auc   = row["auroc"]
        feat  = row["feature"]
        flag  = ""
        if auc >= 0.90:
            flag = "  ← 누수 의심"
            leakage.append(feat)
        elif auc <= 0.10:
            flag = "  ← 반전 누수"
            leakage.append(feat)
        elif auc >= 0.80:
            flag = "  ← 주의"
        if auc >= 0.75 or auc <= 0.25:
            print(f"{feat:<35} {auc:>8.4f}{flag}")

    print("=" * 55)
    print(f"\n[결과 요약]")
    print(f"  누수 의심 피처 수: {len(leakage)}개")
    if leakage:
        print(f"  목록: {leakage}")

    # ── 5. icu_los_hours 집중 확인 ──────────────────────────────────────
    print("\n[3] icu_los_hours 집중 진단 ...")
    if "icu_los_hours" in df.columns:
        auc = roc_auc_score(y, df["icu_los_hours"].fillna(0))
        print(f"  icu_los_hours 단독 AUROC: {auc:.4f}")
        print(f"  AKI 환자 평균 재원시간:    "
              f"{df[y==1]['icu_los_hours'].mean():.1f}h")
        print(f"  비AKI 환자 평균 재원시간:  "
              f"{df[y==0]['icu_los_hours'].mean():.1f}h")
        if auc >= 0.80:
            print("  → 누수 확정. feature_config.py에서 제거 필요")
    else:
        print("  icu_los_hours 컬럼 없음")

    # ── 6. is_pseudo_cutoff 확인 (혹시 ALL_FEATURES에 있으면) ──────────
    print("\n[4] is_pseudo_cutoff 진단 ...")
    if "is_pseudo_cutoff" in df.columns:
        auc = roc_auc_score(y, df["is_pseudo_cutoff"].fillna(0))
        print(f"  is_pseudo_cutoff 단독 AUROC: {auc:.4f}")
        if auc >= 0.90:
            print("  → 누수 확정. EXCLUDE_COLS에 있는지 확인 필요")
        else:
            print("  → EXCLUDE_COLS에 의해 이미 차단됨 (정상)")
    else:
        print("  is_pseudo_cutoff 컬럼 없음")

    print("\n진단 완료.")

if __name__ == "__main__":
    main()