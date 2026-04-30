"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
xgb_model/train.py  —  XGBoost AKI 예측 모델 학습 파이프라인 (SHAP 포함)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[개선 사항]
  ① F1 / Precision / Recall 지표 추가  (Youden 임계값 기준)
  ② 수치형 결측 → missing_flag 보존 후 KNN Imputation
  ③ map_missing 외 주요 변수 결측 패턴을 피처로 추가
  ④ ★ KNN Imputer를 CV 루프 안으로 이동 (누수 방지)
     Fold별로 train만 fit_transform, val은 transform만 적용한다.
  ⑤ ★★ KNN Imputer를 Optuna objective 루프 안으로도 이동 (누수 완전 차단)
     기존: 전체 X를 impute 후 Optuna에 전달 → val 정보 혼입
     수정: objective 내부 각 fold에서 train만으로 fit → val에 transform만
  ⑥ ★★★ KNN imputation을 루프 전 1회 사전 계산 후 재사용 (속도 대폭 향상)
     기존: Optuna 50 trials × 3 folds = 150회 KNN fit
     수정: 3회(inner) + 5회(CV) = 8회만 fit → 누수 없음, 속도 향상
  ⑦ ★★★★ Optuna/CV에서 KNN 완전 제거 (속도 최적화)
     XGBoost tree_method="hist"는 NaN을 내부적으로 처리한다.
     missing_flag 피처로 결측 패턴은 이미 보존되어 있으므로 KNN 불필요.
     KNN은 추론 시 재사용을 위해 최종 모델 학습 전 1회만 수행한다.

[전체 파이프라인 흐름]
  DB 로드 → missing_flag 생성 → 전처리
  → Optuna 탐색 (NaN 그대로, fold split만 사전 계산)
  → 5-Fold CV  (NaN 그대로, fold split만 사전 계산)
  → 최종 모델 학습 (전체 KNN imputed X_final)
  → 아티팩트 저장 → SHAP 분석

실행 방법:
    python train.py                        # 기본 실행
    python train.py --trials 100           # Optuna 시도 횟수 지정
    python train.py --db-uri postgresql://user:pw@localhost:5432/mimic4

출력 아티팩트:
    model/xgb_aki.json          XGBoost 모델 가중치
    model/threshold.txt         최적 분류 임계값 (Youden's J)
    model/feature_names.csv     학습에 사용된 피처 목록 (순서 포함)
    model/label_encoders.pkl    LabelEncoder 인스턴스 딕셔너리
    model/knn_imputer.pkl       KNNImputer 인스턴스 (추론 시 재사용)
    output/eval_metrics.txt     5-Fold CV 성능 지표 (F1/Precision/Recall 포함)
    output/feature_importance.csv  XGBoost feature importance
    output/track_importance.csv    트랙별 기여도 (SCR-03~07 그룹)
    output/shap_summary_plot.png   SHAP Beeswarm 플롯
    output/shap_summary_bar.png    SHAP 막대 그래프 (피처 기여도)
    output/shap_values.npy         SHAP values (numpy)
    output/shap_base_values.npy    SHAP base values (numpy)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import pickle
import argparse
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import sqlalchemy
import xgboost as xgb
import optuna
import shap
import matplotlib.pyplot as plt

from optuna.samplers import TPESampler
from optuna.pruners  import MedianPruner

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.impute import KNNImputer

from feature_config import (
    ALL_FEATURES,
    FEAT_LAB,
    FEAT_ISCHEMIC,
    FEAT_DRUG,
    FEAT_RULE,
    TARGET,
)
from preprocessing import preprocess_for_training

warnings.filterwarnings("ignore")

os.makedirs("model",  exist_ok=True)
os.makedirs("output", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 전역 상수 / 설정값
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DB_URI = os.getenv(
    "DATABASE_URL",
    "postgresql://bio4:bio4@localhost:5432/mimic4"
)

N_FOLDS         = 5
N_OPTUNA_TRIALS = 50
EARLY_STOP      = 30
RANDOM_STATE    = 42
KNN_NEIGHBORS   = 5

MISSING_FLAG_COLS = [
    "map",
    "hemoglobin",
    "potassium",
    "creatinine",
    "urine_output",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────────────────────────────────────────

def load_master_features_from_db(db_uri: str) -> pd.DataFrame:
    print("[데이터 로드] cdss_master_features ...")
    engine = sqlalchemy.create_engine(db_uri)
    df = pd.read_sql("SELECT * FROM aki_project.cdss_master_features", engine)
    print(f"  로드 완료: {len(df):,}행 × {df.shape[1]}열")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. 결측 플래그 생성
# ─────────────────────────────────────────────────────────────────────────────

def add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    """결측 패턴이 임상적 의미를 갖는 컬럼에 대해 이진 _missing 플래그를 추가한다.

    반드시 KNN imputation 이전에 호출해야 한다.
    """
    df = df.copy()
    added = []

    for col in MISSING_FLAG_COLS:
        flag_col = f"{col}_missing"
        if col in df.columns and flag_col not in df.columns:
            df[flag_col] = df[col].isna().astype(np.int8)
            added.append(flag_col)

    if added:
        print(f"  [missing_flag] 추가된 플래그 피처: {added}")
    else:
        print("  [missing_flag] 새로 추가할 플래그 없음 (이미 모두 존재)")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. KNN Imputation (최종 모델 학습 전 전체 데이터에만 1회 사용)
# ─────────────────────────────────────────────────────────────────────────────

def fit_knn_imputer(X: pd.DataFrame) -> tuple[pd.DataFrame, KNNImputer]:
    """수치형 컬럼의 결측값을 KNN Imputation으로 채우고, Imputer를 저장한다.

    [사용 시점]
    - 최종 모델 학습 직전에만 1회 호출한다.
    - Optuna / CV 단계에서는 호출하지 않는다.
      XGBoost(tree_method="hist")가 NaN을 내부적으로 처리하고,
      missing_flag 피처로 결측 패턴이 이미 보존되어 있기 때문이다.

    Args:
        X: 전처리된 피처 DataFrame (NaN 포함)

    Returns:
        (imputed_X, fitted_imputer)
    """
    numeric_cols = [
        c for c in X.select_dtypes(include=[np.number]).columns
        if not c.endswith("_missing")
    ]
    non_numeric_cols = [c for c in X.columns if c not in numeric_cols]

    print(f"\n[KNN Imputation] 수치형 {len(numeric_cols)}개 컬럼 대상 (k={KNN_NEIGHBORS})")
    before_missing = X[numeric_cols].isna().sum().sum()
    print(f"  imputation 전 총 결측 셀: {before_missing:,}")

    imputer       = KNNImputer(n_neighbors=KNN_NEIGHBORS)
    X_imputed_arr = imputer.fit_transform(X[numeric_cols])

    X_imputed_num = pd.DataFrame(X_imputed_arr, columns=numeric_cols, index=X.index)

    after_missing = X_imputed_num.isna().sum().sum()
    print(f"  imputation 후 총 결측 셀: {after_missing:,}")

    X_result = pd.concat([X_imputed_num, X[non_numeric_cols]], axis=1)[X.columns]

    imputer_path = "model/knn_imputer.pkl"
    with open(imputer_path, "wb") as f:
        pickle.dump({"imputer": imputer, "numeric_cols": numeric_cols}, f)
    print(f"  [저장] {imputer_path}")

    return X_result, imputer


# ─────────────────────────────────────────────────────────────────────────────
# 4. Fold 사전 계산 (KNN 없음 — XGBoost 내부 NaN 처리 활용)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PrecomputedFold:
    """split이 완료된 fold 데이터를 담는 컨테이너."""
    X_tr:  np.ndarray
    X_val: np.ndarray
    y_tr:  np.ndarray
    y_val: np.ndarray


def precompute_folds(
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
) -> list[PrecomputedFold]:
    """fold split을 미리 계산해 numpy 배열로 저장한다.

    [설계 근거]
    - XGBoost(tree_method="hist")는 NaN을 자체적으로 처리한다.
      각 split에서 최적 분기 방향을 NaN 샘플에 자동 할당하므로
      CV / Optuna 단계에서 별도 imputation이 불필요하다.
    - missing_flag 피처(_missing 접미사)로 결측 패턴 정보가 이미 보존되어 있다.
    - KNN imputation은 추론 시 재사용을 위해 최종 모델 학습 직전 1회만 수행한다.

    Args:
        X: NaN이 포함된 원본 X_raw
        y: 타겟 Series
        cv: StratifiedKFold 인스턴스 (random_state 고정 필수)

    Returns:
        PrecomputedFold 리스트 (fold 순서 보존)
    """
    folds: list[PrecomputedFold] = []
    for tr_idx, val_idx in cv.split(X, y):
        folds.append(PrecomputedFold(
            X_tr  = X.iloc[tr_idx].values,
            X_val = X.iloc[val_idx].values,
            y_tr  = y.iloc[tr_idx].values,
            y_val = y.iloc[val_idx].values,
        ))
    return folds


# ─────────────────────────────────────────────────────────────────────────────
# 5. Optuna 하이퍼파라미터 탐색
# ─────────────────────────────────────────────────────────────────────────────

def build_xgb_params_from_optuna_trial(trial: optuna.Trial, scale_pos_weight: float) -> dict:
    return {
        "max_depth":          trial.suggest_int("max_depth",           4,    8),
        "learning_rate":      trial.suggest_float("learning_rate",     0.01, 0.2,  log=True),
        "n_estimators":       trial.suggest_int("n_estimators",      100,  800),
        "subsample":          trial.suggest_float("subsample",         0.6,  1.0),
        "colsample_bytree":   trial.suggest_float("colsample_bytree",  0.5,  1.0),
        "min_child_weight":   trial.suggest_int("min_child_weight",    1,   20),
        "gamma":              trial.suggest_float("gamma",             0.0,  5.0),
        "reg_alpha":          trial.suggest_float("reg_alpha",         1e-4, 10.0, log=True),
        "reg_lambda":         trial.suggest_float("reg_lambda",        1e-4, 10.0, log=True),
        "scale_pos_weight":   scale_pos_weight,
        "tree_method":        "hist",
        "eval_metric":        "aucpr",
        "use_label_encoder":  False,
        "random_state":       RANDOM_STATE,
        "n_jobs":             -1,
    }


def run_optuna_hyperparameter_search(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = N_OPTUNA_TRIALS,
) -> dict:
    """Optuna TPE 샘플러로 XGBoost 하이퍼파라미터를 탐색하고 최적값을 반환한다.

    [KNN 제거 근거]
    XGBoost는 NaN을 내부적으로 처리하므로 Optuna 탐색 단계에서 KNN imputation이
    불필요하다. fold split만 사전 계산해 모든 trial에서 재사용한다.

    Args:
        X: NaN이 포함된 원본 X_raw
        y: AKI 타겟 Series
        n_trials: Optuna 탐색 횟수
    """
    neg_count        = (y == 0).sum()
    pos_count        = (y == 1).sum()
    scale_pos_weight = float(neg_count / max(pos_count, 1))

    cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    print("\n[Optuna] inner fold 사전 계산 중 ...")
    precomputed = precompute_folds(X, y, cv_inner)
    print(f"  완료: {len(precomputed)}개 fold 준비")

    def objective(trial: optuna.Trial) -> float:
        params    = build_xgb_params_from_optuna_trial(trial, scale_pos_weight)
        fold_aucs = []

        for fold_i, fold in enumerate(precomputed):
            model = xgb.XGBClassifier(**params, early_stopping_rounds=EARLY_STOP)
            model.fit(
                fold.X_tr, fold.y_tr,
                eval_set=[(fold.X_val, fold.y_val)],
                verbose=False,
            )

            prob = model.predict_proba(fold.X_val)[:, 1]
            fold_aucs.append(roc_auc_score(fold.y_val, prob))

            trial.report(np.mean(fold_aucs), fold_i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(fold_aucs)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_STATE),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"\n[Optuna] {n_trials}회 탐색 시작 ...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params.update({
        "scale_pos_weight":  scale_pos_weight,
        "tree_method":       "hist",
        "eval_metric":       "aucpr",
        "use_label_encoder": False,
        "random_state":      RANDOM_STATE,
        "n_jobs":            -1,
    })

    print(f"  최적 AUROC: {study.best_value:.4f}")
    print(f"  최적 파라미터: {best_params}")
    return best_params


# ─────────────────────────────────────────────────────────────────────────────
# 6. 5-Fold CV 성능 평가
# ─────────────────────────────────────────────────────────────────────────────

def run_stratified_kfold_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    best_params: dict,
) -> dict:
    """Optuna 최적 파라미터로 5-Fold CV를 수행해 최종 성능을 평가한다.

    [KNN 제거 근거]
    XGBoost는 NaN을 내부적으로 처리하므로 CV 단계에서도 KNN imputation이 불필요하다.
    fold split만 사전 계산해 재사용한다.

    Args:
        X: NaN이 포함된 원본 X_raw
        y: AKI 타겟 Series
        best_params: Optuna 최적 하이퍼파라미터
    """
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    metrics = {k: [] for k in ["auroc", "auprc", "f1", "precision", "recall", "threshold"]}

    print("\n[5-Fold CV] fold 사전 계산 중 ...")
    precomputed = precompute_folds(X, y, cv)
    print(f"  완료: {len(precomputed)}개 fold 준비")

    print("[5-Fold CV] 최적 파라미터로 성능 평가 ...")
    for fold_i, fold in enumerate(precomputed, 1):
        model = xgb.XGBClassifier(**best_params, early_stopping_rounds=EARLY_STOP)
        model.fit(
            fold.X_tr, fold.y_tr,
            eval_set=[(fold.X_val, fold.y_val)],
            verbose=False,
        )

        prob      = model.predict_proba(fold.X_val)[:, 1]
        auroc     = roc_auc_score(fold.y_val, prob)
        auprc     = average_precision_score(fold.y_val, prob)
        threshold = _find_youden_threshold(fold.y_val, prob)
        pred      = (prob >= threshold).astype(int)
        f1        = f1_score(fold.y_val,        pred, zero_division=0)
        precision = precision_score(fold.y_val, pred, zero_division=0)
        recall    = recall_score(fold.y_val,    pred, zero_division=0)

        for key, val in zip(
            ["auroc", "auprc", "f1", "precision", "recall", "threshold"],
            [auroc,   auprc,   f1,   precision,   recall,   threshold],
        ):
            metrics[key].append(val)

        print(
            f"  Fold {fold_i}: "
            f"AUROC={auroc:.4f}  AUPRC={auprc:.4f}  "
            f"F1={f1:.4f}  Prec={precision:.4f}  Recall={recall:.4f}  "
            f"Threshold={threshold:.3f}"
        )

    print(f"\n  [CV 요약]")
    for key, label in [
        ("auroc",     "AUROC    "),
        ("auprc",     "AUPRC    "),
        ("f1",        "F1       "),
        ("precision", "Precision"),
        ("recall",    "Recall   "),
    ]:
        vals = metrics[key]
        print(f"  {label}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print(f"  최적 임계값 평균: {np.mean(metrics['threshold']):.3f}")

    return metrics


def _find_youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """ROC 곡선에서 Youden's J 통계량이 최대가 되는 분류 임계값을 반환한다."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx])


# ─────────────────────────────────────────────────────────────────────────────
# 7. 최종 모델 학습 (전체 데이터)
# ─────────────────────────────────────────────────────────────────────────────

def train_final_model_on_full_data(
    X: pd.DataFrame,
    y: pd.Series,
    best_params: dict,
    best_threshold: float,
) -> xgb.XGBClassifier:
    """CV로 검증된 최적 파라미터로 전체 데이터에서 최종 모델을 학습한다."""
    print("\n[최종 모델 학습] 전체 데이터로 재학습 ...")

    params = {k: v for k, v in best_params.items() if k != "early_stopping_rounds"}

    model = xgb.XGBClassifier(**params)
    model.fit(X, y, verbose=False)

    model_path = os.getenv("XGB_MODEL_PATH", "model/xgb_aki.json")
    model.save_model(model_path)
    print(f"  [저장] {model_path}")

    threshold_path = os.getenv("XGB_THRESHOLD_PATH", "model/threshold.txt")
    with open(threshold_path, "w", encoding="utf-8") as f:
        f.write(str(best_threshold))
    print(f"  [저장] {threshold_path}  (threshold={best_threshold:.3f})")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# 8. Feature Importance 저장
# ─────────────────────────────────────────────────────────────────────────────

def save_feature_importance_reports(
    model: xgb.XGBClassifier,
    feature_names: list[str],
) -> None:
    """XGBoost 피처 중요도를 개별 피처 단위·CDSS 트랙 단위로 CSV에 저장한다."""
    importance = model.get_booster().get_score(importance_type="gain")

    df_imp = pd.DataFrame([
        {"feature": f, "importance_gain": importance.get(f, 0.0)}
        for f in feature_names
    ]).sort_values("importance_gain", ascending=False)

    df_imp.to_csv("output/feature_importance.csv", index=False, encoding="utf-8")
    print("\n  [저장] output/feature_importance.csv")
    print("  상위 10 피처:")
    print(df_imp.head(10).to_string(index=False))

    TRACK_GROUPS = {
        "SCR-03 약물":     FEAT_DRUG,
        "SCR-04 혈액검사": FEAT_LAB,
        "SCR-05 허혈":     FEAT_ISCHEMIC,
        "SCR-06 규칙":     FEAT_RULE,
    }
    track_rows = []
    for track_name, feat_list in TRACK_GROUPS.items():
        total = sum(importance.get(f, 0.0) for f in feat_list)
        track_rows.append({"track": track_name, "total_gain": round(total, 2)})

    df_track = pd.DataFrame(track_rows).sort_values("total_gain", ascending=False)
    df_track["pct"] = (df_track["total_gain"] / df_track["total_gain"].sum() * 100).round(1)
    df_track.to_csv("output/track_importance.csv", index=False, encoding="utf-8")
    print("\n  [저장] output/track_importance.csv")
    print(df_track.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 9. 평가 지표 저장
# ─────────────────────────────────────────────────────────────────────────────

def save_evaluation_metrics(cv_metrics: dict, best_params: dict) -> None:
    """5-Fold CV 성능 지표를 텍스트 파일에 저장한다."""
    label_map = {
        "auroc":     "AUROC    ",
        "auprc":     "AUPRC    ",
        "f1":        "F1       ",
        "precision": "Precision",
        "recall":    "Recall   ",
        "threshold": "Threshold",
    }

    lines = ["=== AKI XGBoost 5-Fold CV 결과 ===\n"]
    for key, label in label_map.items():
        vals = cv_metrics[key]
        lines.append(f"{label}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    lines += ["", "=== 최적 하이퍼파라미터 ==="]
    lines += [f"  {k}: {v}" for k, v in best_params.items()]

    with open("output/eval_metrics.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n  [저장] output/eval_metrics.txt")


# ─────────────────────────────────────────────────────────────────────────────
# 10. SHAP 분석
# ─────────────────────────────────────────────────────────────────────────────

def save_shap_analysis(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    feature_names: list[str],
) -> None:
    """SHAP TreeExplainer로 모델 해석성을 분석하고 플롯·수치 파일을 저장한다."""
    print("\n[SHAP 분석] TreeExplainer로 SHAP values 계산 중 ...")

    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        base_value  = explainer.expected_value

        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            base_value  = base_value[1] if isinstance(base_value, (list, np.ndarray)) else base_value

        print("  [1/3] SHAP Summary Plot (Beeswarm) 생성 중 ...")
        plt.figure(figsize=(14, 8))
        shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="dot", show=False)
        plt.tight_layout()
        plt.savefig("output/shap_summary_plot.png", dpi=300, bbox_inches="tight")
        print("    [저장] output/shap_summary_plot.png")
        plt.close()

        print("  [2/3] SHAP Feature Importance Bar 생성 중 ...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig("output/shap_summary_bar.png", dpi=300, bbox_inches="tight")
        print("    [저장] output/shap_summary_bar.png")
        plt.close()

        print("  [3/3] SHAP values 저장 중 ...")
        np.save("output/shap_values.npy",      shap_values)
        np.save("output/shap_base_values.npy", np.array(base_value))

        meta = {
            "feature_names":     feature_names,
            "shap_values_shape": shap_values.shape,
            "base_value":        float(base_value) if np.isscalar(base_value)
                                 else float(np.mean(base_value)),
        }
        np.save("output/shap_meta.npy", meta, allow_pickle=True)
        print("    [저장] output/shap_values.npy / shap_base_values.npy / shap_meta.npy")

        print("\n  [SHAP 통계] 상위 10 중요 피처:")
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_features  = np.argsort(mean_abs_shap)[-10:][::-1]
        for rank, idx in enumerate(top_features, 1):
            feat     = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
            shap_imp = mean_abs_shap[idx]
            print(f"    {rank:2d}. {feat:30s} → SHAP importance: {shap_imp:.4f}")

    except Exception as e:
        print(f"\n  ⚠️  SHAP 분석 중 오류 발생: {e}")
        print("  계속 진행합니다...")


# ─────────────────────────────────────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────────────────────────────────────

def main(db_uri: str, n_trials: int) -> None:
    """XGBoost AKI 예측 모델의 전체 학습 파이프라인을 실행한다.

    [파이프라인 흐름]
      1. DB 로드
      2. missing_flag 생성
      3. 전처리 (인코딩·클리핑·피처 선택)  → X_raw (NaN 포함)
      4. Optuna 탐색: fold split 3회 사전 계산 → 50 trials 재사용 (KNN 없음)
      5. 5-Fold CV:  fold split 5회 사전 계산 → 평가에 재사용  (KNN 없음)
      6. 최종 모델: X_raw 전체 KNN imputation → X_final + imputer 저장
      7. 리포트 저장
      8. SHAP 분석

    [KNN fit 횟수 비교]
      v4 (사전계산): 3 + 5 + 1 =  9회
      v6 (KNN 제거): 0 + 0 + 1 =  1회  ← 현재
    """
    print("=" * 70)
    print("AKI CDSS XGBoost 학습 파이프라인 v6 (Optuna/CV KNN 제거)")
    print("=" * 70)

    # 1. DB 로드
    df_raw = load_master_features_from_db(db_uri)

    # 2. missing_flag 생성 (imputation 전에 반드시 실행)
    df_flagged = add_missing_flags(df_raw)

    # 3. 전처리: 인코딩·클리핑·피처 선택
    X_raw, y, feature_names, encoders = preprocess_for_training(df_flagged)

    # 4. Optuna 탐색: fold split 사전 계산 후 모든 trial 재사용 (KNN 없음)
    best_params = run_optuna_hyperparameter_search(X_raw, y, n_trials)

    # 5. 5-Fold CV: fold split 사전 계산 후 평가 (KNN 없음)
    cv_metrics     = run_stratified_kfold_cross_validation(X_raw, y, best_params)
    best_threshold = float(np.mean(cv_metrics["threshold"]))

    # 6. 최종 모델: 전체 데이터 KNN imputation (추론용 imputer 저장 목적)
    print("\n[최종 모델용 imputation] 전체 데이터로 fit_transform + imputer 저장 ...")
    X_final, knn_imputer = fit_knn_imputer(X_raw.copy())

    # feature_names: imputation 후 컬럼 순서 기준으로 갱신
    feature_names = list(X_final.columns)

    final_model = train_final_model_on_full_data(X_final, y, best_params, best_threshold)

    # 7. 리포트 저장
    save_feature_importance_reports(final_model, feature_names)
    save_evaluation_metrics(cv_metrics, best_params)

    # 8. SHAP 분석
    save_shap_analysis(final_model, X_final, feature_names)

    # 최종 아티팩트 확인
    print("\n" + "=" * 70)
    print("학습 완료. 생성된 아티팩트:")
    artifact_paths = [
        "model/xgb_aki.json",
        "model/threshold.txt",
        "model/feature_names.csv",
        "model/label_encoders.pkl",
        "model/knn_imputer.pkl",
        "output/eval_metrics.txt",
        "output/feature_importance.csv",
        "output/track_importance.csv",
        "output/shap_summary_plot.png",
        "output/shap_summary_bar.png",
        "output/shap_values.npy",
        "output/shap_base_values.npy",
    ]
    for path in artifact_paths:
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"  {exists} {path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AKI XGBoost 학습 파이프라인 v6 (Optuna/CV KNN 제거)"
    )
    parser.add_argument(
        "--db-uri",
        default=DEFAULT_DB_URI,
        help="SQLAlchemy DB URI",
    )
    parser.add_argument(
        "--trials",
        default=N_OPTUNA_TRIALS,
        type=int,
        help=f"Optuna 탐색 횟수 (기본값: {N_OPTUNA_TRIALS})",
    )
    args = parser.parse_args()
    main(db_uri=args.db_uri, n_trials=args.trials)