"""AKI 모델 Validation 러너 (작업지시서 4 + 9).

test_final.csv(홀드아웃) + 학습 .pkl 번들을 로드해 다음을 산출하고
docs/vv/ai_model_validation_report.{md,json} 로 저장한다:

  - Stage1(LR) 이진 AKI: AUROC/AUPRC/calibration/Brier/ECE
  - Stage2(LGBM) 중증(AKI 내 Stage2+3): 동일 지표
  - subgroup analysis (연령대, 성별)
  - missing-data sensitivity (*_missing 플래그 기반)
  - feature ablation (약물 정보 포함/미포함 성능 차이 — 4.2)
  - time-series validation (누수 검사 + onset 라벨 정합성 — 4.3)
  - clinical 단조성(배포 예측기 — Cr/eGFR/소변량 — 4.2)

실행:
  cd backend && set PYTHONPATH=. && .venv\\Scripts\\python.exe tests\\vv_runners\\run_aki_validation.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent        # <repo>/tests/vv_runners
REPO = HERE.parent.parent                      # 리포 루트
BACKEND = REPO / "backend"                      # <repo>/backend
sys.path.insert(0, str(BACKEND))

from validator.clinical import run_clinical_checks  # noqa: E402
from validator.metrics import binary_metrics  # noqa: E402
from validator.report import ValidationReport  # noqa: E402
from validator.subgroup import (  # noqa: E402
    feature_ablation,
    missing_sensitivity,
    quantile_bands,
    subgroup_metrics,
)
from validator.timeseries import check_label_consistency, check_no_leakage  # noqa: E402

DATA = REPO / "AKI" / "preprocessing" / "final_dataset" / "test_final.csv"
MODELS = BACKEND / "ml_models"
OUT_DIR = BACKEND / "docs" / "vv"

DRUG_COLS = ["vasopressor_flag", "vasopressor_hours", "norepi_dose_max"]
KIDNEY_COLS = ["creatinine_min", "creatinine_max", "creatinine_delta", "bun_max", "bun_cr_ratio"]
URINE_COLS = ["urine_output_sum", "urine_output_6h", "urine_ml_kg_hr", "oliguria_flag"]


def main() -> int:
    if not DATA.exists():
        print(f"[SKIP] 데이터셋 없음: {DATA}")
        return 1

    s1 = pickle.load(open(MODELS / "stage1_LR_full.pkl", "rb"))
    
    # scikit-learn 버전 호환성 패치 (multi_class 속성 유실 해결)
    lr = s1["model"]
    if not hasattr(lr, "multi_class"):
        lr.multi_class = "auto"
        
    s2 = pickle.load(open(MODELS / "stage2_LGBM_v13_full_classweight.pkl", "rb"))
    feat = s1["feature_cols"]
    fidx = {c: i for i, c in enumerate(feat)}

    df = pd.read_csv(DATA)
    X = df[feat].to_numpy(dtype=float)
    y_bin = df["aki_label"].to_numpy(dtype=int)
    y_stage = df["aki_stage"].to_numpy(dtype=int)

    lr, lgbm = s1["model"], s2["model"]
    p_aki = lr.predict_proba(X)[:, 1]
    # LGBM 은 학습 시 피처명을 기억 → 동일 컬럼명 DataFrame 으로 추론(경고 방지/정합).
    p_sev = lgbm.predict_proba(df[feat])[:, 1]

    rep = ValidationReport(title="AKI 2-Stage 모델 Validation 리포트")
    rep.dataset = {
        "source": str(DATA.relative_to(REPO)),
        "n_rows": int(len(df)),
        "features": len(feat),
        "aki_prevalence": f"{y_bin.mean():.1%}",
        "stage_distribution": {int(k): int(v) for k, v in pd.Series(y_stage).value_counts().sort_index().items()},
        "stage1_threshold": s1.get("best_threshold"),
        "stage2_threshold": s2.get("best_threshold"),
    }

    # --- 1) Stage1 이진 AKI ---
    m1 = binary_metrics(y_bin, p_aki)
    rep.add("Stage1 (LR) — Non-AKI vs AKI", m1.as_dict())

    # --- 2) Stage2 중증(AKI 내 Stage2+3) ---
    aki_mask = y_stage > 0
    y_sev = (y_stage[aki_mask] >= 2).astype(int)
    m2 = binary_metrics(y_sev, p_sev[aki_mask])
    rep.add("Stage2 (LGBM) — Stage1 vs Stage2+3 (AKI 내)", m2.as_dict())

    # --- 3) subgroup analysis ---
    # 주의: 데이터셋 age 는 표준화(z-score)되어 절대 연령대가 무의미 → 상대 분위(tercile)로 분리.
    bands = quantile_bands(df["age"].to_numpy())
    rep.add(
        "Subgroup — 연령 분위(표준화 age tercile, Stage1 AKI)",
        [r.as_dict() for r in subgroup_metrics(y_bin, p_aki, bands)],
    )
    if "gender" in df.columns:
        rep.add(
            "Subgroup — 성별 (Stage1 AKI)",
            [r.as_dict() for r in subgroup_metrics(y_bin, p_aki, df["gender"].to_numpy())],
        )

    # --- 4) missing-data sensitivity ---
    miss_cols = [c for c in df.columns if c.endswith("_missing")]
    miss_count = df[miss_cols].sum(axis=1).to_numpy() if miss_cols else np.zeros(len(df))
    rep.add(
        "Missing-data sensitivity (Stage1 AKI)",
        {
            "n_missing_flag_cols": len(miss_cols),
            **missing_sensitivity(y_bin, p_aki, miss_count).as_dict(),
        },
    )

    # --- 5) feature ablation (약물 포함/미포함 등) ---
    def predict_aki(Xm):
        return lr.predict_proba(Xm)[:, 1]

    abl = feature_ablation(
        predict_aki, X, y_bin, fidx,
        {"약물(vasopressor/norepi)": DRUG_COLS, "신장(Cr/BUN)": KIDNEY_COLS, "소변량": URINE_COLS},
    )
    rep.add("Feature ablation (Stage1 AKI AUROC drop)", [a.as_dict() for a in abl])

    # --- 6) time-series validation ---
    ts = {}
    if {"prediction_cutoff", "aki_onset_time"}.issubset(df.columns):
        ts["leakage_check"] = check_no_leakage(df).as_dict()
        ts["label_consistency"] = check_label_consistency(df).as_dict()
    rep.add("Time-series validation (4.3)", ts or {"note": "시간 컬럼 없음"})

    # --- 7) clinical 단조성(배포 예측기) ---
    from ai_draft.aki_model import get_aki_predictor

    checks = run_clinical_checks(get_aki_predictor())
    rep.add("Clinical validity — 단조성(배포 예측기, 4.2)", [c.as_dict() for c in checks])

    # --- known failures (수치 기반 자동 기록) ---
    if m2.auroc < 0.65:
        rep.known_failures.append(
            f"Stage2(LGBM) 중증 판별 AUROC {m2.auroc:.3f} — 우연(0.5) 대비 낮은 변별력. "
            f"중증도 분류는 보조 지표로만 사용, 단독 임상 판단 금지."
        )
    if m2.as_dict()["positives"] and m2.auprc < 0.5:
        rep.known_failures.append(
            f"Stage2 AUPRC {m2.auprc:.3f} — 양성(Stage2+3) 희소로 정밀도 한계(고recall/저precision)."
        )
    high_drop = [a for a in abl if a.auroc_drop < 0.005]
    for a in high_drop:
        rep.known_failures.append(
            f"'{a.feature_group}' 제거 시 AUROC 변화 {a.auroc_drop:+.4f} — 모델이 해당 신호를 거의 사용하지 않음."
        )
    for c in checks:
        if not c.passed:
            rep.known_failures.append(f"임상 단조성 위반: {c.name} → {c.risk_scores}")
    ms = missing_sensitivity(y_bin, p_aki, miss_count)
    if ms.delta > 0.03:
        rep.known_failures.append(
            f"결측 많은 환자 AUROC {ms.high_missing_auroc:.3f} (적은 환자 {ms.low_missing_auroc:.3f}) "
            f"— 결측 많을수록 성능 저하 Δ{ms.delta:.3f}."
        )
    if m1.ece > 0.1:
        rep.known_failures.append(
            f"Stage1 보정 불량(ECE {m1.ece:.3f}) — 예측확률 0.7–0.9 구간이 실제 위험을 과대평가. "
            f"운영 시 확률값보다 위험등급(threshold {s1.get('best_threshold')}) 사용 권장. 재보정(Platt/Isotonic) 필요."
        )
    if "careunit" not in df.columns and "first_careunit" not in df.columns:
        rep.known_failures.append(
            "ICU/ER/ward subgroup 분석 불가 — 데이터셋에 careunit 컬럼 없음(4.1 요건 부분 미충족). "
            "대체로 성별·연령분위·결측수준 subgroup 으로 검증함."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ai_model_validation_report.md").write_text(rep.to_markdown(), encoding="utf-8")
    (OUT_DIR / "ai_model_validation_report.json").write_text(rep.to_json(), encoding="utf-8")
    print(f"[OK] 리포트 저장 → {OUT_DIR / 'ai_model_validation_report.md'}")
    print(f"     Stage1 AUROC={m1.auroc:.4f} AUPRC={m1.auprc:.4f} ECE={m1.ece:.4f}")
    print(f"     Stage2 AUROC={m2.auroc:.4f} AUPRC={m2.auprc:.4f}")
    print(f"     known_failures={len(rep.known_failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
