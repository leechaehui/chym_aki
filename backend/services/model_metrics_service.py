"""AKI 2-stage 모델 운영 성능 지표 — 의료진 표시용(읽기 전용).

검증 리포트(ai_model_validation_report.json)의 AUROC/AUPRC/ECE·known_failures 와,
운영 threshold 에서의 혼동행렬 지표(정확도/Precision/민감도/위음성률)를 합쳐 제공한다.

정확도·Precision·FNR 은 **테스트셋(test_final.csv)** 에서 산출한다.
라이브 코호트(train+valid+test)로 계산하면 학습 데이터가 섞여 낙관적 편향이 생기므로 금지.
프로세스당 1회만 계산하고 캐시한다.
"""
from __future__ import annotations

import functools
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parent.parent
MODELS = BACKEND / "ml_models"
REPORT_JSON = BACKEND / "docs" / "vv" / "ai_model_validation_report.json"
TEST_CSV = BACKEND.parent / "AKI" / "preprocessing" / "final_dataset" / "test_final.csv"


def available() -> bool:
    return (MODELS / "stage1_LR_full.pkl").exists() and TEST_CSV.exists() and REPORT_JSON.exists()


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """임계값 적용 후 혼동행렬 기반 지표(정확도/Precision/민감도/위음성률/특이도)."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    n = tp + fp + fn + tn
    pos = tp + fn  # 실제 양성
    return {
        "n": n,
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / pos if pos else 0.0,            # 민감도(Sensitivity)
        "fnr": fn / pos if pos else 0.0,               # 위음성률 = 1 - 민감도
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


@functools.lru_cache(maxsize=1)
def model_metrics() -> dict:
    """모델 성능 지표(운영 threshold 혼동행렬 + 리포트 AUROC/AUPRC/ECE·known_failures)."""
    report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    ds = report["dataset"]
    th1 = float(ds["stage1_threshold"])
    th2 = float(ds["stage2_threshold"])

    s1 = pickle.load(open(MODELS / "stage1_LR_full.pkl", "rb"))
    
    # scikit-learn 버전 호환성 패치 (multi_class 속성 유실 해결)
    lr = s1["model"]
    if not hasattr(lr, "multi_class"):
        lr.multi_class = "auto"
        
    s2 = pickle.load(open(MODELS / "stage2_LGBM_v13_full_classweight.pkl", "rb"))
    feat = list(s1["feature_cols"])
    df = pd.read_csv(TEST_CSV)

    # Stage1: Non-AKI vs AKI (전체 테스트셋).
    y1 = (df["aki_stage"] >= 1).astype(int).to_numpy()
    p_aki = s1["model"].predict_proba(df[feat].to_numpy(dtype=float))[:, 1]
    m1 = _confusion(y1, (p_aki >= th1).astype(int))

    # Stage2: Stage1 vs Stage2+3 (실제 AKI 환자 내에서만 평가 — 리포트와 동일 정의).
    aki = df["aki_stage"] >= 1
    y2 = (df.loc[aki, "aki_stage"] >= 2).astype(int).to_numpy()
    p_sev = s2["model"].predict_proba(df.loc[aki, feat])[:, 1]
    m2 = _confusion(y2, (p_sev >= th2).astype(int))

    sec = report["sections"]
    s1_rep = sec["Stage1 (LR) — Non-AKI vs AKI"]
    s2_rep = sec["Stage2 (LGBM) — Stage1 vs Stage2+3 (AKI 내)"]

    return {
        "generatedAt": report.get("generated_at"),
        "dataset": {
            "source": "test_final.csv",
            "nRows": ds["n_rows"],
            "akiPrevalence": ds["aki_prevalence"],
            "stage1Threshold": th1,
            "stage2Threshold": th2,
        },
        "stage1": {
            "label": "AKI 발생 여부 (Non-AKI vs AKI)",
            **{k: m1[k] for k in ("n", "accuracy", "precision", "recall", "fnr", "specificity",
                                  "tp", "fp", "fn", "tn")},
            "auroc": s1_rep["auroc"], "auprc": s1_rep["auprc"], "ece": s1_rep["ece"],
        },
        "stage2": {
            "label": "중증도 (Stage1 vs Stage2-3, AKI 내)",
            **{k: m2[k] for k in ("n", "accuracy", "precision", "recall", "fnr", "specificity",
                                  "tp", "fp", "fn", "tn")},
            "auroc": s2_rep["auroc"], "auprc": s2_rep["auprc"], "ece": s2_rep["ece"],
        },
        "knownFailures": report.get("known_failures", []),
    }
