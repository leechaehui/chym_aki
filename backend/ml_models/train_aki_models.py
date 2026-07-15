"""AKI 2-stage model training script.

Following the model architecture and threshold search logic of the original training scripts 
(07_modeling_LR_full.py / 09_modeling_LGBM_v13_full_classweight.py), this script trains 
directly from the preprocessed final CSV files (AKI/preprocessing/final_dataset/*_final.csv) 
instead of intermediate .npy files.

Output bundles (same format loaded by backend ai_draft/aki_model.py):
  stage1_LR_full.pkl                  : {"model","best_threshold","feature_cols","metrics"}
  stage2_LGBM_v13_full_classweight.pkl: {"model","best_threshold","feature_cols","version","metrics"}

실행:
  cd backend && .venv\\Scripts\\python.exe ml_models/train_aki_models.py
"""
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight  # noqa: E402

# ----------------------------------------------------------
# 경로
# ----------------------------------------------------------
HERE = Path(__file__).resolve().parent          # backend/ml_models
REPO = HERE.parent.parent                        # repo root (예: C:/team/chym_aki) — __file__ 기준 동적
DATA_DIR = REPO / "AKI" / "preprocessing" / "final_dataset"
# final_aki_model.py 가 참조하는 위치에도 복사한다.
AKI_MODELS_DIR = REPO / "AKI" / "modeling" / "models"

# 원본 스크립트와 동일한 35개 피처.
FEATURE_COLS = [
    "map_mean", "map_min", "map_below65_hours",
    "sbp_min", "sbp_mean", "shock_index_mean",
    "hr_max", "hr_mean", "rr_max", "rr_mean",
    "temp_max", "temp_mean",
    "urine_output_sum", "urine_output_6h", "oliguria_flag",
    "creatinine_min", "creatinine_max", "creatinine_delta",
    "bun_max", "bun_cr_ratio",
    "lactate_max", "lactate_mean",
    "vasopressor_flag", "vasopressor_hours", "norepi_dose_max",
    "potassium_max", "potassium_mean",
    "bicarbonate_min", "bicarbonate_mean",
    "sodium_min", "sodium_max",
    "hemoglobin_min", "hemoglobin_mean",
    "spo2_min", "spo2_mean",
]


def load_split(name: str):
    df = pd.read_csv(DATA_DIR / f"{name}_final.csv")
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y_bin = df["aki_label"].to_numpy(dtype=int)       # Non-AKI=0, AKI=1
    y_stage = df["aki_stage"].to_numpy(dtype=int)      # 0/1/2/3
    return X, y_bin, y_stage


def search_threshold(y_true, prob, min_recall=0.75):
    """원본과 동일: recall>=min_recall 중 specificity 최대 임계값."""
    best_t, best_spec = 0.5, 0.0
    for t in np.arange(0.10, 0.91, 0.01):
        pred = (prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        if recall >= min_recall and spec > best_spec:
            best_spec, best_t = spec, t
    return round(float(best_t), 2)


# ==========================================================
# STAGE 1 — Logistic Regression (Non-AKI vs AKI)
# ==========================================================
def train_stage1(Xtr, ytr, Xvl, yvl, Xte, yte) -> dict:
    print("=" * 60)
    print("STAGE 1 : Logistic Regression (Non-AKI vs AKI)")
    print("=" * 60)
    model = LogisticRegression(
        class_weight="balanced", max_iter=1000, solver="lbfgs",
        random_state=42, n_jobs=-1,
    )
    model.fit(Xtr, ytr)

    vprob = model.predict_proba(Xvl)[:, 1]
    tprob = model.predict_proba(Xte)[:, 1]
    best_t = search_threshold(yvl, vprob)

    val_auroc = roc_auc_score(yvl, vprob)
    test_auroc = roc_auc_score(yte, tprob)
    vpred = (vprob >= best_t).astype(int)
    tn, fp, fn, tp = confusion_matrix(yvl, vpred).ravel()
    print(f"  best_threshold : {best_t}")
    print(f"  valid AUROC    : {val_auroc:.4f}  | test AUROC: {test_auroc:.4f}")
    print(f"  valid sens/spec: {tp/(tp+fn):.3f} / {tn/(tn+fp):.3f}")

    return {
        "model": model,
        "best_threshold": best_t,
        "feature_cols": FEATURE_COLS,
        "metrics": {
            "valid_auroc": float(val_auroc),
            "valid_auprc": float(average_precision_score(yvl, vprob)),
            "test_auroc": float(test_auroc),
        },
    }


# ==========================================================
# STAGE 2 — LightGBM (Stage1 vs Stage2+3), class_weight
# ==========================================================
def train_stage2(Xtr, ystr, Xvl, ysvl, Xte, yste) -> dict:
    print("=" * 60)
    print("STAGE 2 : LightGBM (Stage1 vs Stage2+3, class_weight)")
    print("=" * 60)
    from lightgbm import LGBMClassifier

    # AKI 환자만(stage>0) 사용. 라벨: Stage1=0, Stage2+3=1
    tr_m, vl_m, te_m = ystr > 0, ysvl > 0, yste > 0
    Xtr2, Xvl2, Xte2 = Xtr[tr_m], Xvl[vl_m], Xte[te_m]
    ytr2 = (ystr[tr_m] >= 2).astype(int)
    yvl2 = (ysvl[vl_m] >= 2).astype(int)
    yte2 = (yste[te_m] >= 2).astype(int)
    print(f"  train Stage1={int((ytr2==0).sum())} Stage2+3={int((ytr2==1).sum())}")

    classes = np.unique(ytr2)
    weights = compute_class_weight("balanced", classes=classes, y=ytr2)
    cw = dict(zip(classes, weights))
    sw = np.array([cw[y] for y in ytr2])

    # 원본은 Optuna 50 trial. 설치되어 있으면 경량 탐색(20), 없으면 합리적 고정 파라미터.
    params = _optuna_or_default(Xtr2, ytr2, Xvl2, yvl2, sw)
    model = LGBMClassifier(**params)
    model.fit(Xtr2, ytr2, sample_weight=sw)

    vprob = model.predict_proba(Xvl2)[:, 1]
    tprob = model.predict_proba(Xte2)[:, 1]
    best_t = search_threshold(yvl2, vprob)

    val_auroc = roc_auc_score(yvl2, vprob)
    test_auroc = roc_auc_score(yte2, tprob)
    vpred = (vprob >= best_t).astype(int)
    val_s23r = recall_score(yvl2, vpred, pos_label=1, zero_division=0)
    print(f"  best_threshold : {best_t}")
    print(f"  valid AUROC    : {val_auroc:.4f}  | test AUROC: {test_auroc:.4f}")
    print(f"  valid S2+3 recall: {val_s23r:.3f}")

    return {
        "model": model,
        "best_threshold": best_t,
        "feature_cols": FEATURE_COLS,
        "version": "v13_full_classweight",
        "metrics": {
            "valid_auroc": float(val_auroc),
            "valid_s23r": float(val_s23r),
            "test_auroc": float(test_auroc),
            "valid_f1": float(f1_score(yvl2, vpred, zero_division=0)),
            "valid_precision": float(precision_score(yvl2, vpred, zero_division=0)),
        },
    }


def _optuna_or_default(Xtr, ytr, Xvl, yvl, sw):
    base = {
        "objective": "binary", "metric": "binary_logloss",
        "random_state": 42, "n_jobs": -1, "verbose": -1,
    }
    try:
        import optuna
        from lightgbm import LGBMClassifier

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            p = {
                **base,
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 60),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 1.0),
            }
            m = LGBMClassifier(**p)
            m.fit(Xtr, ytr, sample_weight=sw)
            pred = m.predict(Xvl)
            s23r = recall_score(yvl, pred, pos_label=1, zero_division=0)
            macro = f1_score(yvl, pred, average="macro")
            return s23r * 0.6 + macro * 0.4

        print("  Optuna 탐색(50 trials)...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)
        return {**base, **study.best_params}
    except ImportError:
        print("  optuna 미설치 → 고정 파라미터 사용")
        return {
            **base, "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
            "num_leaves": 40, "min_child_samples": 20, "subsample": 0.8,
            "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 0.1,
        }


def main():
    print(f"DATA_DIR: {DATA_DIR}")
    Xtr, ytr, ystr = load_split("train")
    Xvl, yvl, ysvl = load_split("valid")
    Xte, yte, yste = load_split("test")
    print(f"train={Xtr.shape} valid={Xvl.shape} test={Xte.shape}\n")

    stage1 = train_stage1(Xtr, ytr, Xvl, yvl, Xte, yte)
    stage2 = train_stage2(Xtr, ystr, Xvl, ysvl, Xte, yste)

    AKI_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    targets = {
        "stage1_LR_full.pkl": stage1,
        "stage2_LGBM_v13_full_classweight.pkl": stage2,
    }
    for fname, bundle in targets.items():
        for out_dir in (HERE, AKI_MODELS_DIR):
            with open(out_dir / fname, "wb") as f:
                pickle.dump(bundle, f)
            print(f"saved → {out_dir / fname}")

    print("\nDONE")


if __name__ == "__main__":
    main()
