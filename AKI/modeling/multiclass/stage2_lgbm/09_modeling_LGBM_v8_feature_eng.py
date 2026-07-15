# ==========================================================
# 09_modeling_LGBM_v8_feature_eng.py
#
# Stage 2 : AKI Stage Classification
# Method : 피처 엔지니어링 (파생 피처 추가)
# Feature : Excluded 28개 + 파생 피처 추가
# Author : A
# ==========================================================

import os
import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
import pandas as pd
import optuna
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus'] = False
optuna.logging.set_verbosity(optuna.logging.WARNING)

from lightgbm import LGBMClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    classification_report, confusion_matrix
)

# ==========================================================
# SETTINGS
# ==========================================================

OUT_DIR = "outputs/LGBM_stage_v8_feature_eng"
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs("models", exist_ok=True)

ORIGINAL_FEATURE_COLS = [
    'map_mean', 'map_min', 'map_below65_hours',
    'sbp_min', 'sbp_mean', 'shock_index_mean',
    'hr_max', 'hr_mean', 'rr_max', 'rr_mean',
    'temp_max', 'temp_mean',
    'urine_output_sum', 'urine_output_6h', 'oliguria_flag',
    'creatinine_min', 'creatinine_max', 'creatinine_delta',
    'bun_max', 'bun_cr_ratio',
    'lactate_max', 'lactate_mean',
    'vasopressor_flag', 'vasopressor_hours', 'norepi_dose_max',
    'potassium_max', 'potassium_mean',
    'bicarbonate_min', 'bicarbonate_mean',
    'sodium_min', 'sodium_max',
    'hemoglobin_min', 'hemoglobin_mean',
    'spo2_min', 'spo2_mean'
]

REMOVE_COLS = [
    'urine_output_sum', 'urine_output_6h', 'oliguria_flag',
    'creatinine_min', 'creatinine_max', 'creatinine_delta',
    'bun_cr_ratio'
]

keep_idx = [
    i for i, col in enumerate(ORIGINAL_FEATURE_COLS)
    if col not in REMOVE_COLS
]
BASE_FEATURE_COLS = [
    col for col in ORIGINAL_FEATURE_COLS
    if col not in REMOVE_COLS
]

# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 70)
print("LOAD DATA")
print("=" * 70)

X_train_orig = np.load("data/X_train.npy")
X_valid_orig = np.load("data/X_valid.npy")
X_test_orig  = np.load("data/X_test.npy")

y_stage_train = np.load("data/y_stage_train.npy")
y_stage_valid = np.load("data/y_stage_valid.npy")
y_stage_test  = np.load("data/y_stage_test.npy")

train_mask = y_stage_train > 0
valid_mask = y_stage_valid > 0
test_mask  = y_stage_test  > 0

X_base_tr = X_train_orig[train_mask][:, keep_idx]
X_base_vl = X_valid_orig[valid_mask][:, keep_idx]
X_base_te = X_test_orig[test_mask][:, keep_idx]

y_tr = y_stage_train[train_mask] - 1
y_vl = y_stage_valid[valid_mask] - 1
y_te = y_stage_test[test_mask]   - 1

# 피처 이름 → 인덱스 매핑
feat_idx = {col: i for i, col in enumerate(BASE_FEATURE_COLS)}

# ==========================================================
# 파생 피처 생성 함수
# ==========================================================

def add_derived_features(X, feature_cols):
    fi = {col: i for i, col in enumerate(feature_cols)}

    derived = []
    derived_names = []

    # 1. 신장-혈역학 상호작용
    # MAP 최저치 × BUN 최대치 → 저관류로 인한 신장 손상
    map_min  = X[:, fi['map_min']]
    bun_max  = X[:, fi['bun_max']]
    d1 = map_min * bun_max
    derived.append(d1)
    derived_names.append("map_min_x_bun_max")

    # 2. 쇼크 지속 강도
    # 승압제 용량 × 투여 시간
    norepi = X[:, fi['norepi_dose_max']]
    vhours = X[:, fi['vasopressor_hours']]
    d2 = norepi * vhours
    derived.append(d2)
    derived_names.append("shock_severity")

    # 3. 전해질 이상 복합 지표
    # 칼륨 최대 - 중탄산염 최소 (고칼륨혈증 + 대사성 산증)
    k_max   = X[:, fi['potassium_max']]
    bic_min = X[:, fi['bicarbonate_min']]
    d3 = k_max - bic_min
    derived.append(d3)
    derived_names.append("hyperkalemia_acidosis")

    # 4. 빈혈-저산소 복합
    hgb_min  = X[:, fi['hemoglobin_min']]
    spo2_min = X[:, fi['spo2_min']]
    d4 = hgb_min * spo2_min
    derived.append(d4)
    derived_names.append("anemia_hypoxia")

    # 5. 혈역학 불안정도
    # SBP 범위 (최대-최소 대신 mean-min으로 근사)
    sbp_mean = X[:, fi['sbp_mean']]
    sbp_min  = X[:, fi['sbp_min']]
    d5 = sbp_mean - sbp_min
    derived.append(d5)
    derived_names.append("sbp_variability")

    # 6. 체액 과부하 지표
    # 나트륨 최대 - 나트륨 최소
    na_max = X[:, fi['sodium_max']]
    na_min = X[:, fi['sodium_min']]
    d6 = na_max - na_min
    derived.append(d6)
    derived_names.append("sodium_variability")

    # 7. 젖산-혈압 복합 지표
    lactate = X[:, fi['lactate_max']]
    map_val = X[:, fi['map_mean']]
    d7 = lactate / (map_val + 1e-6)
    derived.append(d7)
    derived_names.append("lactate_map_ratio")

    # 8. MAP 저하 시간 × 승압제 여부
    map_hrs  = X[:, fi['map_below65_hours']]
    vaso_flg = X[:, fi['vasopressor_flag']]
    d8 = map_hrs * vaso_flg
    derived.append(d8)
    derived_names.append("hypotension_vasopressor")

    derived_arr = np.column_stack(derived)
    X_new = np.hstack([X, derived_arr])
    new_cols = feature_cols + derived_names

    return X_new, new_cols

# ==========================================================
# 파생 피처 추가
# ==========================================================

X_tr, FEATURE_COLS = add_derived_features(X_base_tr, BASE_FEATURE_COLS)
X_vl, _            = add_derived_features(X_base_vl, BASE_FEATURE_COLS)
X_te, _            = add_derived_features(X_base_te, BASE_FEATURE_COLS)

print("=" * 70)
print("파생 피처 추가 결과")
print("=" * 70)
print(f"기본 피처 : {len(BASE_FEATURE_COLS)}개")
print(f"파생 피처 : {len(FEATURE_COLS) - len(BASE_FEATURE_COLS)}개")
print(f"최종 피처 : {len(FEATURE_COLS)}개")
print()
print("추가된 파생 피처:")
for col in FEATURE_COLS[len(BASE_FEATURE_COLS):]:
    print(f"  + {col}")
print()

# ==========================================================
# CLASS WEIGHT
# ==========================================================

classes = np.unique(y_tr)
weights = compute_class_weight(
    class_weight="balanced", classes=classes, y=y_tr
)
class_weight_dict = dict(zip(classes, weights))
sample_weight     = np.array(
    [class_weight_dict[y] for y in y_tr]
)

print("CLASS WEIGHT")
for cls, w in class_weight_dict.items():
    print(f"  Stage {cls+1} : {w:.4f}")
print()

# ==========================================================
# OPTUNA
# ==========================================================

trial_history = []

def objective(trial):
    params = {
        "objective"        : "multiclass",
        "num_class"        : 3,
        "metric"           : "multi_logloss",
        "n_estimators"     : trial.suggest_int("n_estimators", 100, 500),
        "max_depth"        : trial.suggest_int("max_depth", 3, 8),
        "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves"       : trial.suggest_int("num_leaves", 20, 60),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "subsample"        : trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha"        : trial.suggest_float("reg_alpha", 0, 1.0),
        "reg_lambda"       : trial.suggest_float("reg_lambda", 0, 1.0),
        "random_state"     : 42,
        "n_jobs"           : -1,
        "verbose"          : -1,
    }
    model = LGBMClassifier(**params)
    model.fit(X_tr, y_tr, sample_weight=sample_weight)
    pred     = model.predict(X_vl)
    macro_f1 = f1_score(y_vl, pred, average="macro")
    trial_history.append([trial.number, macro_f1])
    print(f"Trial {trial.number:03d} | Macro F1 = {macro_f1:.5f}")
    return macro_f1

print("=" * 70)
print("OPTUNA TUNING v8 feature_eng (50 trials)")
print("=" * 70)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=False)

trial_df = pd.DataFrame(trial_history, columns=["trial","macro_f1"])
trial_df.to_csv(f"{OUT_DIR}/optuna_history.csv", index=False)

print()
print("=" * 70)
print("BEST RESULT")
print("=" * 70)
print(f"Best Macro F1 : {study.best_value:.5f}")
print()
for k, v in study.best_params.items():
    print(f"  {k:<25} : {v}")
print()

# ==========================================================
# FINAL MODEL
# ==========================================================

print("=" * 70)
print("FINAL MODEL TRAINING")
print("=" * 70)

best_params = study.best_params
best_params.update({
    "objective"   : "multiclass",
    "num_class"   : 3,
    "metric"      : "multi_logloss",
    "random_state": 42,
    "n_jobs"      : -1,
    "verbose"     : -1,
})

model = LGBMClassifier(**best_params)
model.fit(X_tr, y_tr, sample_weight=sample_weight)
print("학습 완료")
print()

# ==========================================================
# EVALUATION
# ==========================================================

valid_pred = model.predict(X_vl)
test_pred  = model.predict(X_te)

accuracy    = accuracy_score(y_vl, valid_pred)
weighted_f1 = f1_score(y_vl, valid_pred, average='weighted')
macro_f1    = f1_score(y_vl, valid_pred, average='macro')
s1_recall   = recall_score(y_vl, valid_pred, labels=[0], average=None)[0]
s2_recall   = recall_score(y_vl, valid_pred, labels=[1], average=None)[0]
s3_recall   = recall_score(y_vl, valid_pred, labels=[2], average=None)[0]

test_accuracy    = accuracy_score(y_te, test_pred)
test_weighted_f1 = f1_score(y_te, test_pred, average='weighted')
test_macro_f1    = f1_score(y_te, test_pred, average='macro')
ts1_recall = recall_score(y_te, test_pred, labels=[0], average=None)[0]
ts2_recall = recall_score(y_te, test_pred, labels=[1], average=None)[0]
ts3_recall = recall_score(y_te, test_pred, labels=[2], average=None)[0]

print("=" * 70)
print("VALIDATION RESULT")
print("=" * 70)
print(f"Accuracy      : {accuracy:.4f}")
print(f"Weighted F1   : {weighted_f1:.4f}")
print(f"Macro F1      : {macro_f1:.4f}")
print()
print(f"Stage1 Recall : {s1_recall:.4f}")
print(f"Stage2 Recall : {s2_recall:.4f}  ← 핵심 지표")
print(f"Stage3 Recall : {s3_recall:.4f}")
print()
print(classification_report(
    y_vl, valid_pred,
    target_names=["Stage1","Stage2","Stage3"]
))

print("=" * 70)
print("TEST RESULT")
print("=" * 70)
print(f"Accuracy      : {test_accuracy:.4f}")
print(f"Weighted F1   : {test_weighted_f1:.4f}")
print(f"Macro F1      : {test_macro_f1:.4f}")
print()
print(f"Stage1 Recall : {ts1_recall:.4f}")
print(f"Stage2 Recall : {ts2_recall:.4f}  ← 핵심 지표")
print(f"Stage3 Recall : {ts3_recall:.4f}")

cm = confusion_matrix(y_vl, valid_pred)
print()
print("Confusion Matrix (Validation)")
print(cm)
print()

# ==========================================================
# FIGURES
# ==========================================================

print("=" * 70)
print("GENERATE FIGURES")
print("=" * 70)

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Stage1','Stage2','Stage3'],
            yticklabels=['Stage1','Stage2','Stage3'])
plt.title("Validation Confusion Matrix\n(LGBM v8 Feature Engineering)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=300)
plt.close()

cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(7, 6))
sns.heatmap(cm_pct, annot=True, fmt='.1%', cmap='Blues',
            xticklabels=['Stage1','Stage2','Stage3'],
            yticklabels=['Stage1','Stage2','Stage3'])
plt.title("Normalized Confusion Matrix\n(LGBM v8 Feature Engineering)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix_percent.png", dpi=300)
plt.close()

metrics = {
    "Accuracy"   : accuracy,
    "Weighted F1": weighted_f1,
    "Macro F1"   : macro_f1,
    "S1 Recall"  : s1_recall,
    "S2 Recall"  : s2_recall,
    "S3 Recall"  : s3_recall,
}
colors = ["#185FA5","#185FA5","#185FA5","#4A90D9","#E24B4A","#E67E22"]
plt.figure(figsize=(10, 5))
bars = plt.bar(metrics.keys(), metrics.values(), color=colors)
plt.ylim(0, 1.1)
for bar, val in zip(bars, metrics.values()):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.02,
             f"{val:.4f}", ha="center", fontsize=10)
plt.title("Validation Performance Summary\n(LGBM v8 Feature Engineering)")
plt.ylabel("Score")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/performance_summary.png", dpi=300)
plt.close()

recall_vals = [s1_recall, s2_recall, s3_recall]
plt.figure(figsize=(7, 5))
bars = plt.bar(["Stage1 Recall","Stage2 Recall","Stage3 Recall"],
               recall_vals, color=["#185FA5","#E24B4A","#E67E22"])
plt.ylim(0, 1.1)
for bar, val in zip(bars, recall_vals):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.02,
             f"{val:.4f}", ha="center", fontsize=12)
plt.axhline(0.30, color="red", linestyle="--", alpha=0.5, label="목표 0.30")
plt.title("Per-class Recall (LGBM v8 Feature Engineering)")
plt.ylabel("Recall")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/per_class_recall.png", dpi=300)
plt.close()

# 피처 중요도 (파생 피처 포함)
importance_df = pd.DataFrame({
    "feature"   : FEATURE_COLS,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)
importance_df.to_csv(f"{OUT_DIR}/feature_importance.csv", index=False)

# 파생 피처만 별도 표시
derived_cols = FEATURE_COLS[len(BASE_FEATURE_COLS):]
derived_imp  = importance_df[importance_df["feature"].isin(derived_cols)]
print("파생 피처 중요도:")
print(derived_imp[["feature","importance"]].to_string(index=False))
print()

top15 = importance_df.head(15)
plt.figure(figsize=(10, 8))
colors_bar = ["#E24B4A" if col in derived_cols else "#185FA5"
              for col in top15["feature"]]
plt.barh(top15["feature"], top15["importance"], color=colors_bar)
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("Top 15 Feature Importance (LGBM v8)\n빨간색=파생 피처")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(trial_df["trial"], trial_df["macro_f1"],
         marker="o", markersize=4, color="#185FA5")
plt.xlabel("Trial")
plt.ylabel("Macro F1")
plt.title("Optuna Trial History (LGBM v8)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/optuna_history.png", dpi=300)
plt.close()

print(f"Saved → {OUT_DIR}")
print()

# ==========================================================
# 결과 CSV
# ==========================================================

results_df = pd.DataFrame({
    "Metric": ["Accuracy","Weighted F1","Macro F1",
               "Stage1 Recall","Stage2 Recall","Stage3 Recall"],
    "Valid" : [accuracy, weighted_f1, macro_f1,
               s1_recall, s2_recall, s3_recall],
    "Test"  : [test_accuracy, test_weighted_f1, test_macro_f1,
               ts1_recall, ts2_recall, ts3_recall],
})
results_df.to_csv(f"{OUT_DIR}/results.csv", index=False)

# ==========================================================
# SAVE MODEL
# ==========================================================

with open("models/stage2_LGBM_v8_feature_eng.pkl", "wb") as f:
    pickle.dump({
        "model"            : model,
        "feature_cols"     : FEATURE_COLS,
        "base_feature_cols": BASE_FEATURE_COLS,
        "derived_cols"     : derived_cols,
        "remove_cols"      : REMOVE_COLS,
        "best_params"      : best_params,
        "version"          : "v8_feature_eng",
        "metrics": {
            "valid_macro_f1"  : macro_f1,
            "valid_s1_recall" : s1_recall,
            "valid_s2_recall" : s2_recall,
            "valid_s3_recall" : s3_recall,
            "test_macro_f1"   : test_macro_f1,
            "test_s2_recall"  : ts2_recall,
        }
    }, f)

print("=" * 70)
print("MODEL SAVED : models/stage2_LGBM_v8_feature_eng.pkl")
print("=" * 70)
print("FINISHED")
