"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
xgb_model/train.py  —  XGBoost AKI 예측 모델 학습 파이프라인 (SHAP 포함)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

실행 방법:
    python train.py                        # 기본 실행
    python train.py --trials 100           # Optuna 시도 횟수 지정
    python train.py --db-uri postgresql://user:pw@localhost:5432/mimic4

출력 아티팩트:
    model/xgb_aki.json          XGBoost 모델 가중치
    model/threshold.txt         최적 분류 임계값 (Youden's J)
    model/feature_names.csv     학습에 사용된 피처 목록 (순서 포함)
    model/label_encoders.pkl    LabelEncoder 인스턴스 딕셔너리
    output/eval_metrics.txt     5-Fold CV 성능 지표
    output/feature_importance.csv  XGBoost feature importance
    output/track_importance.csv    트랙별 기여도 (SCR-03~07 그룹)
    output/shap_summary_plot.png   SHAP Beeswarm 플롯
    output/shap_summary_bar.png    SHAP 막대 그래프 (피처 기여도)
    output/shap_values.npy         SHAP values (numpy)
    output/shap_base_values.npy    SHAP base values (numpy)

설계 원칙:
  - Optuna TPE로 하이퍼파라미터를 탐색하고, Pruner로 조기 종료한다.
  - StratifiedKFold 5-Fold CV로 AKI 불균형(일반적으로 20~30%)에서
    각 Fold의 양성 비율을 유지한다.
  - scale_pos_weight로 클래스 불균형을 보정한다.
  - early_stopping_rounds로 과적합을 방지한다.
  - SHAP TreeExplainer로 모델 해석성을 분석한다.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import argparse
import warnings
from typing import Optional

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
from sklearn.metrics import roc_auc_score, average_precision_score

from feature_config   import ALL_FEATURES, FEAT_LAB, FEAT_ISCHEMIC, FEAT_DRUG, FEAT_RULE, TARGET
from preprocessing    import preprocess_for_training

warnings.filterwarnings("ignore")
os.makedirs("model",  exist_ok=True)
os.makedirs("output", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DB_URI  = os.getenv(
    "DATABASE_URL",
    "postgresql://bio4:bio4@localhost:5432/mimic4"
)
N_FOLDS        = 5     # StratifiedKFold 분할 수
N_OPTUNA_TRIALS= 50    # Optuna 탐색 횟수 (시간 여유 있으면 100~200 권장)
EARLY_STOP     = 30    # early_stopping_rounds
RANDOM_STATE   = 42


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────────────────────────────────────

def load_master_features_from_db(db_uri: str) -> pd.DataFrame:
    """cdss_master_features 전체를 DB에서 로드한다.

    전처리(필터링·인코딩·클리핑)는 preprocess_for_training()에서 수행한다.

    Args:
        db_uri: SQLAlchemy 연결 문자열

    Returns:
        원본 DataFrame (전처리 전)
    """
    print("[데이터 로드] cdss_master_features ...")
    engine = sqlalchemy.create_engine(db_uri)
    df = pd.read_sql("SELECT * FROM aki_project.cdss_master_features", engine)
    print(f"  로드 완료: {len(df):,}행 × {df.shape[1]}열")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Optuna 하이퍼파라미터 탐색
# ─────────────────────────────────────────────────────────────────────────────

def build_xgb_params_from_optuna_trial(trial: optuna.Trial, scale_pos_weight: float) -> dict:
    """Optuna Trial에서 XGBoost 하이퍼파라미터를 샘플링한다.

    탐색 공간 설계 근거:
      max_depth  4~8    : 너무 깊으면 과적합, 너무 얕으면 과소적합
      learning_rate      : 낮을수록 정밀하나 학습 시간 증가 → early_stopping으로 보상
      subsample, colsample_bytree: 무작위성으로 과적합 방지 (Random Forest 효과)
      gamma, reg_alpha, reg_lambda: 정규화로 트리 과성장 억제
      scale_pos_weight   : AKI 불균형 보정 (음성/양성 비율)
    """
    return {
        "max_depth":          trial.suggest_int("max_depth",        4, 8),
        "learning_rate":      trial.suggest_float("learning_rate",  0.01, 0.2,  log=True),
        "n_estimators":       trial.suggest_int("n_estimators",   100, 800),
        "subsample":          trial.suggest_float("subsample",     0.6, 1.0),
        "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight":   trial.suggest_int("min_child_weight", 1, 20),
        "gamma":              trial.suggest_float("gamma",         0.0, 5.0),
        "reg_alpha":          trial.suggest_float("reg_alpha",     1e-4, 10.0, log=True),
        "reg_lambda":         trial.suggest_float("reg_lambda",    1e-4, 10.0, log=True),
        "scale_pos_weight":   scale_pos_weight,   # 클래스 불균형 보정
        "tree_method":        "hist",              # GPU 없는 환경에서 빠른 학습
        "eval_metric":        "aucpr",             # AKI 불균형에서 AUPRC가 더 적합
        "use_label_encoder":  False,
        "random_state":       RANDOM_STATE,
        "n_jobs":             -1,
    }


def run_optuna_hyperparameter_search(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = N_OPTUNA_TRIALS,
) -> dict:
    """Optuna TPE로 XGBoost 하이퍼파라미터를 탐색한다.

    각 Trial에서 3-Fold CV AUROC를 최대화하는 파라미터를 찾는다.
    MedianPruner로 성능이 낮은 Trial은 조기 종료해 탐색 시간을 줄인다.

    Args:
        X:        전처리된 피처 DataFrame
        y:        타겟 Series
        n_trials: Optuna 시도 횟수 (기본 50)

    Returns:
        최적 하이퍼파라미터 딕셔너리
    """
    neg_count       = (y == 0).sum()
    pos_count       = (y == 1).sum()
    scale_pos_weight= float(neg_count / max(pos_count, 1))

    # 3-Fold CV (탐색 속도 우선, 최종 평가는 5-Fold로 별도 수행)
    cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params = build_xgb_params_from_optuna_trial(trial, scale_pos_weight)
        fold_aucs = []

        for fold_i, (tr_idx, val_idx) in enumerate(cv_inner.split(X, y)):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            model = xgb.XGBClassifier(**params, early_stopping_rounds=EARLY_STOP)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            prob = model.predict_proba(X_val)[:, 1]
            fold_aucs.append(roc_auc_score(y_val, prob))

            # Pruner: 이 Fold 결과가 중간값 미만이면 Trial 조기 종료
            trial.report(np.mean(fold_aucs), fold_i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(fold_aucs)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_STATE),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    # Optuna 로그 최소화 (진행 상황만 표시)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"\n[Optuna] {n_trials}회 탐색 시작 ...")
    study.fit = study.optimize  # IDE 자동완성 우회
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params["scale_pos_weight"] = scale_pos_weight
    best_params["tree_method"]      = "hist"
    best_params["eval_metric"]      = "aucpr"
    best_params["use_label_encoder"]= False
    best_params["random_state"]     = RANDOM_STATE
    best_params["n_jobs"]           = -1

    print(f"  최적 AUROC: {study.best_value:.4f}")
    print(f"  최적 파라미터: {best_params}")
    return best_params


# ─────────────────────────────────────────────────────────────────────────────
# 5-Fold 교차 검증 + 최종 모델 학습
# ─────────────────────────────────────────────────────────────────────────────

def run_stratified_kfold_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    best_params: dict,
) -> tuple[list[float], list[float], list[float]]:
    """5-Fold StratifiedKFold CV로 모델 성능을 최종 평가한다.

    각 Fold에서 AUROC·AUPRC·최적 임계값(Youden's J)을 계산하고,
    전체 CV 결과의 평균±표준편차를 출력한다.

    Args:
        X:           전처리된 피처 DataFrame
        y:           타겟 Series
        best_params: Optuna 최적 파라미터

    Returns:
        (fold_aurocs, fold_auprcs, fold_thresholds) 리스트
    """
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_aurocs, fold_auprcs, fold_thresholds = [], [], []

    print(f"\n[5-Fold CV] 최적 파라미터로 성능 평가 ...")
    for fold_i, (tr_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(**best_params, early_stopping_rounds=EARLY_STOP)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        prob   = model.predict_proba(X_val)[:, 1]
        auroc  = roc_auc_score(y_val, prob)
        auprc  = average_precision_score(y_val, prob)

        # Youden's J 최적 임계값: sensitivity + specificity 최대화
        threshold = _find_youden_threshold(y_val, prob)

        fold_aurocs.append(auroc)
        fold_auprcs.append(auprc)
        fold_thresholds.append(threshold)
        print(f"  Fold {fold_i}: AUROC={auroc:.4f}  AUPRC={auprc:.4f}  Threshold={threshold:.3f}")

    print(f"\n  [CV 요약]")
    print(f"  AUROC : {np.mean(fold_aurocs):.4f} ± {np.std(fold_aurocs):.4f}")
    print(f"  AUPRC : {np.mean(fold_auprcs):.4f} ± {np.std(fold_auprcs):.4f}")
    print(f"  최적 임계값 평균: {np.mean(fold_thresholds):.3f}")
    return fold_aurocs, fold_auprcs, fold_thresholds


def _find_youden_threshold(y_true: pd.Series, y_prob: np.ndarray) -> float:
    """Youden's J 통계량으로 최적 분류 임계값을 탐색한다.

    Youden's J = Sensitivity + Specificity - 1 을 최대화하는 threshold.
    AKI 탐지에서 민감도와 특이도의 균형을 맞추는 데 사용한다.

    Args:
        y_true: 실제 레이블
        y_prob: AKI 예측 확률 (0~1)

    Returns:
        최적 분류 임계값 (float)
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # Youden's J = tpr + (1 - fpr) - 1 = tpr - fpr
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx])


# ─────────────────────────────────────────────────────────────────────────────
# 최종 모델 학습 (전체 데이터)
# ─────────────────────────────────────────────────────────────────────────────

def train_final_model_on_full_data(
    X: pd.DataFrame,
    y: pd.Series,
    best_params: dict,
    best_threshold: float,
) -> xgb.XGBClassifier:
    """CV로 확인된 최적 파라미터로 전체 데이터에서 최종 모델을 학습한다.

    CV Fold에서 얻은 평균 n_estimators를 최종 학습에 사용한다.
    (early_stopping은 전체 데이터에서 사용 불가 → n_estimators 고정)

    모델 저장: model/xgb_aki.json
    임계값 저장: model/threshold.txt

    Args:
        X:              전처리된 피처 DataFrame (전체)
        y:              타겟 Series (전체)
        best_params:    Optuna 최적 파라미터
        best_threshold: CV에서 계산된 평균 Youden 임계값

    Returns:
        학습 완료된 XGBClassifier 인스턴스
    """
    print("\n[최종 모델 학습] 전체 데이터로 재학습 ...")

    # early_stopping 없이 고정 n_estimators로 학습
    params = {k: v for k, v in best_params.items() if k != "early_stopping_rounds"}

    model = xgb.XGBClassifier(**params)
    model.fit(X, y, verbose=False)

    # 모델 저장
    model_path = os.getenv("XGB_MODEL_PATH", "model/xgb_aki.json")
    model.save_model(model_path)
    print(f"  [저장] {model_path}")

    # 임계값 저장
    threshold_path = os.getenv("XGB_THRESHOLD_PATH", "model/threshold.txt")
    with open(threshold_path, "w") as f:
        f.write(str(best_threshold))
    print(f"  [저장] {threshold_path}  (threshold={best_threshold:.3f})")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Feature Importance 저장
# ─────────────────────────────────────────────────────────────────────────────

def save_feature_importance_reports(
    model: xgb.XGBClassifier,
    feature_names: list[str],
) -> None:
    """피처 중요도를 개별 피처·트랙별로 CSV에 저장한다.

    저장 파일:
      output/feature_importance.csv  개별 피처 importance (gain 기준)
      output/track_importance.csv    SCR 화면별 트랙 기여도 합산

    gain 방식을 사용하는 이유:
      weight(분기 횟수)보다 gain(분기로 인한 손실 감소량)이
      임상적 중요도 해석에 더 의미 있다.
    """
    # 개별 피처 importance
    importance = model.get_booster().get_score(importance_type="gain")
    df_imp = pd.DataFrame([
        {"feature": f, "importance_gain": importance.get(f, 0.0)}
        for f in feature_names
    ]).sort_values("importance_gain", ascending=False)

    df_imp.to_csv("output/feature_importance.csv", index=False)
    print("\n  [저장] output/feature_importance.csv")
    print("  상위 10 피처:")
    print(df_imp.head(10).to_string(index=False))

    # 트랙별 기여도 합산 (SCR 화면 대응)
    TRACK_GROUPS = {
        "SCR-03 약물":   FEAT_DRUG,
        "SCR-04 혈액검사": FEAT_LAB,
        "SCR-05 허혈":   FEAT_ISCHEMIC,
        "SCR-06 규칙":   FEAT_RULE,
    }
    track_rows = []
    for track_name, feat_list in TRACK_GROUPS.items():
        total = sum(importance.get(f, 0.0) for f in feat_list)
        track_rows.append({"track": track_name, "total_gain": round(total, 2)})

    df_track = pd.DataFrame(track_rows).sort_values("total_gain", ascending=False)
    df_track["pct"] = (df_track["total_gain"] / df_track["total_gain"].sum() * 100).round(1)
    df_track.to_csv("output/track_importance.csv", index=False)
    print("\n  [저장] output/track_importance.csv")
    print(df_track.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# SHAP 분석
# ─────────────────────────────────────────────────────────────────────────────

def save_shap_analysis(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    feature_names: list[str],
) -> None:
    """SHAP values로 모델 해석성을 분석하고 시각화한다.

    저장 파일:
      output/shap_summary_plot.png    SHAP Beeswarm 플롯 (개별 샘플)
      output/shap_summary_bar.png     SHAP 막대 그래프 (평균 기여도)
      output/shap_values.npy          SHAP values (numpy 형식)
      output/shap_base_values.npy     SHAP base values (numpy 형식)

    설명:
      - SHAP: SHapley Additive exPlanations
        각 피처가 예측에 얼마나 기여했는지를 게임 이론으로 해석
      - Beeswarm 플롯: 각 샘플의 SHAP value 분포 시각화
      - 막대 그래프: 전체 데이터에서 평균 피처 기여도
    """
    print("\n[SHAP 분석] TreeExplainer로 SHAP values 계산 중 ...")

    try:
        # TreeExplainer 생성 (XGBoost용 최적화된 Explainer)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        base_value = explainer.expected_value

        # 이진 분류: shap_values는 리스트 → [0] = neg, [1] = pos (AKI)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # AKI=1에 대한 SHAP values
            base_value = base_value[1] if isinstance(base_value, (list, np.ndarray)) else base_value

        # ─────────────────────────────────────────────────────────────────────
        # 1. SHAP Summary Plot (Beeswarm)
        # ─────────────────────────────────────────────────────────────────────
        print("  [1/3] SHAP Summary Plot (Beeswarm) 생성 중 ...")
        plt.figure(figsize=(14, 8))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_names,
            plot_type="dot",  # Beeswarm 방식
            show=False
        )
        plt.tight_layout()
        plt.savefig("output/shap_summary_plot.png", dpi=300, bbox_inches='tight')
        print("    [저장] output/shap_summary_plot.png")
        plt.close()

        # ─────────────────────────────────────────────────────────────────────
        # 2. SHAP Summary Plot (Bar)
        # ─────────────────────────────────────────────────────────────────────
        print("  [2/3] SHAP Feature Importance Bar (평균 기여도) 생성 중 ...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_names,
            plot_type="bar",  # 막대 그래프
            show=False
        )
        plt.tight_layout()
        plt.savefig("output/shap_summary_bar.png", dpi=300, bbox_inches='tight')
        print("    [저장] output/shap_summary_bar.png")
        plt.close()

        # ─────────────────────────────────────────────────────────────────────
        # 3. SHAP Values 저장 (numpy 형식 - 추후 분석용)
        # ─────────────────────────────────────────────────────────────────────
        print("  [3/3] SHAP values 저장 중 ...")
        np.save("output/shap_values.npy", shap_values)
        np.save("output/shap_base_values.npy", np.array(base_value))

        # 메타데이터도 함께 저장
        meta = {
            "feature_names": feature_names,
            "shap_values_shape": shap_values.shape,
            "base_value": float(base_value) if np.isscalar(base_value) else float(np.mean(base_value)),
        }
        np.save("output/shap_meta.npy", meta, allow_pickle=True)

        print("    [저장] output/shap_values.npy")
        print("    [저장] output/shap_base_values.npy")
        print("    [저장] output/shap_meta.npy (메타데이터)")

        # ─────────────────────────────────────────────────────────────────────
        # SHAP 통계 출력
        # ─────────────────────────────────────────────────────────────────────
        print("\n  [SHAP 통계]")
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_features = np.argsort(mean_abs_shap)[-10:][::-1]  # 상위 10
        print("  상위 10 중요 피처 (SHAP 기준):")
        for rank, idx in enumerate(top_features, 1):
            feat = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
            shap_imp = mean_abs_shap[idx]
            print(f"    {rank:2d}. {feat:30s} → SHAP importance: {shap_imp:.4f}")

    except Exception as e:
        print(f"\n  ⚠️  SHAP 분석 중 오류 발생: {e}")
        print("  계속 진행합니다...")


def save_evaluation_metrics(
    fold_aurocs: list[float],
    fold_auprcs: list[float],
    fold_thresholds: list[float],
    best_params: dict,
) -> None:
    """CV 성능 지표를 텍스트 파일에 저장한다.

    저장 파일: output/eval_metrics.txt
    """
    lines = [
        "=== AKI XGBoost 5-Fold CV 결과 ===\n",
        f"AUROC : {np.mean(fold_aurocs):.4f} ± {np.std(fold_aurocs):.4f}",
        f"AUPRC : {np.mean(fold_auprcs):.4f} ± {np.std(fold_auprcs):.4f}",
        f"Threshold (Youden) : {np.mean(fold_thresholds):.3f} ± {np.std(fold_thresholds):.3f}",
        "",
        "=== 최적 하이퍼파라미터 ===",
    ] + [f"  {k}: {v}" for k, v in best_params.items()]

    with open("output/eval_metrics.txt", "w") as f:
        f.write("\n".join(lines))
    print("\n  [저장] output/eval_metrics.txt")


# ─────────────────────────────────────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────────────────────────────────────

def main(db_uri: str, n_trials: int) -> None:
    """XGBoost 학습 전체 파이프라인 실행 (SHAP 포함).

    실행 순서:
      1. DB에서 데이터 로드
      2. 전처리 (필터·인코딩·클리핑·피처 선택)
      3. Optuna 하이퍼파라미터 탐색
      4. 5-Fold CV 성능 평가
      5. 전체 데이터로 최종 모델 학습
      6. 아티팩트 저장 (모델·임계값·피처목록·인코더·성능지표)
      7. SHAP 분석 (모델 해석성)
    """
    print("=" * 60)
    print("AKI CDSS XGBoost 학습 파이프라인 (SHAP 포함)")
    print("=" * 60)

    # 1. 데이터 로드
    df_raw = load_master_features_from_db(db_uri)

    # 2. 전처리 (filter + encode + clip + select + save artifacts)
    X, y, feature_names, encoders = preprocess_for_training(df_raw)

    # 3. Optuna 탐색
    best_params = run_optuna_hyperparameter_search(X, y, n_trials)

    # 4. 5-Fold CV
    fold_aurocs, fold_auprcs, fold_thresholds = run_stratified_kfold_cross_validation(
        X, y, best_params
    )
    best_threshold = float(np.mean(fold_thresholds))

    # 5. 최종 모델 학습
    final_model = train_final_model_on_full_data(X, y, best_params, best_threshold)

    # 6. 리포트 저장
    save_feature_importance_reports(final_model, feature_names)
    save_evaluation_metrics(fold_aurocs, fold_auprcs, fold_thresholds, best_params)

    # 7. SHAP 분석 (새로 추가)
    save_shap_analysis(final_model, X, feature_names)

    print("\n" + "=" * 60)
    print("학습 완료. 생성된 아티팩트:")
    for path in [
        "model/xgb_aki.json",
        "model/threshold.txt",
        "model/feature_names.csv",
        "model/label_encoders.pkl",
        "output/eval_metrics.txt",
        "output/feature_importance.csv",
        "output/track_importance.csv",
        "output/shap_summary_plot.png",
        "output/shap_summary_bar.png",
        "output/shap_values.npy",
        "output/shap_base_values.npy",
    ]:
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"  {exists} {path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AKI XGBoost 학습 파이프라인 (SHAP 포함)")
    parser.add_argument("--db-uri",  default=DEFAULT_DB_URI,  help="SQLAlchemy DB URI")
    parser.add_argument("--trials",  default=N_OPTUNA_TRIALS, type=int, help="Optuna 탐색 횟수")
    args = parser.parse_args()
    main(db_uri=args.db_uri, n_trials=args.trials)