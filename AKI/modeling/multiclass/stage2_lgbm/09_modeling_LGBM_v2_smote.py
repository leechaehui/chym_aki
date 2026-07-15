# v2: SMOTE
#     Train만 합성 데이터 생성
#     class_weight 없이 학습

# ==========================================================
# 09_modeling_LGBM_v2_smote.py
#
# Stage 2 : AKI Stage Classification
# Multi-class Classification
# Stage 1 / Stage 2 / Stage 3
#
# Model : LightGBM
# Method : SMOTE (합성 데이터 생성)
# Feature : Excluded (28개, KDIGO 7개 제거)
#
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
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    classification_report, confusion_matrix
)

# ==========================================================
# SETTINGS
# ==========================================================

OUT_DIR = "outputs/LGBM_stage_v2_smote"
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
FEATURE_COLS = [
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

# AKI 환자만 추출
train_mask = y_stage_train > 0
valid_mask = y_stage_valid > 0
test_mask  = y_stage_test  > 0

X_stage_train = X_train_orig[train_mask][:, keep_idx]
X_stage_valid = X_valid_orig[valid_mask][:, keep_idx]
X_stage_test  = X_test_orig[test_mask][:, keep_idx]

y_tr = y_stage_train[train_mask] - 1
y_vl = y_stage_valid[valid_mask] - 1
y_te = y_stage_test[test_mask]   - 1

print(f"X_stage_train : {X_stage_train.shape}")
print(f"X_stage_valid : {X_stage_valid.shape}")
print(f"X_stage_test  : {X_stage_test.shape}")
print(f"Feature Count : {len(FEATURE_COLS)}")
print()

print("=" * 70)
print("Stage 분포 (SMOTE 적용 전)")
print("=" * 70)
print("Train")
print(pd.Series(y_tr).value_counts().sort_index()
      .rename({0:"Stage1",1:"Stage2",2:"Stage3"}))
print()

# ==========================================================
# SMOTE 적용 (Train only)
# ==========================================================

print("=" * 70)
print("SMOTE 적용 (Train only)")
print("=" * 70)

smote = SMOTE(random_state=42)
X_stage_train_sm, y_tr_sm = smote.fit_resample(
    X_stage_train, y_tr
)

print("After SMOTE:")
print(pd.Series(y_tr_sm).value_counts().sort_index()
      .rename({0:"Stage1",1:"Stage2",2:"Stage3"}))
print(f"Train shape: {X_stage_train_sm.shape}")
print()

# ==========================================================
# OPTUNA OBJECTIVE
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
    model.fit(X_stage_train_sm, y_tr_sm)

    pred     = model.predict(X_stage_valid)
    macro_f1 = f1_score(y_vl, pred, average="macro")

    trial_history.append([trial.number, macro_f1])
    print(f"Trial {trial.number:03d} | Macro F1 = {macro_f1:.5f}")

    return macro_f1

# ==========================================================
# OPTUNA TUNING
# ==========================================================

print("=" * 70)
print("OPTUNA TUNING v2 SMOTE (50 trials)")
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
model.fit(X_stage_train_sm, y_tr_sm)
print("학습 완료")
print()

# ==========================================================
# VALIDATION 평가
# ==========================================================

valid_pred = model.predict(X_stage_valid)

accuracy    = accuracy_score(y_vl, valid_pred)
weighted_f1 = f1_score(y_vl, valid_pred, average='weighted')
macro_f1    = f1_score(y_vl, valid_pred, average='macro')
s1_recall   = recall_score(y_vl, valid_pred, labels=[0], average=None)[0]
s2_recall   = recall_score(y_vl, valid_pred, labels=[1], average=None)[0]
s3_recall   = recall_score(y_vl, valid_pred, labels=[2], average=None)[0]

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

# ==========================================================
# TEST 평가
# ==========================================================

test_pred = model.predict(X_stage_test)

test_accuracy    = accuracy_score(y_te, test_pred)
test_weighted_f1 = f1_score(y_te, test_pred, average='weighted')
test_macro_f1    = f1_score(y_te, test_pred, average='macro')
ts1_recall = recall_score(y_te, test_pred, labels=[0], average=None)[0]
ts2_recall = recall_score(y_te, test_pred, labels=[1], average=None)[0]
ts3_recall = recall_score(y_te, test_pred, labels=[2], average=None)[0]

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
plt.title("Validation Confusion Matrix\n(LGBM v2 SMOTE)")
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
plt.title("Normalized Confusion Matrix\n(LGBM v2 SMOTE)")
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
plt.title("Validation Performance Summary\n(LGBM v2 SMOTE)")
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
plt.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="목표 0.5")
plt.title("Per-class Recall (LGBM v2 SMOTE)")
plt.ylabel("Recall")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/per_class_recall.png", dpi=300)
plt.close()

importance_df = pd.DataFrame({
    "feature"   : FEATURE_COLS,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)
importance_df.to_csv(f"{OUT_DIR}/feature_importance.csv", index=False)

top15 = importance_df.head(15)
plt.figure(figsize=(10, 8))
plt.barh(top15["feature"], top15["importance"], color="#185FA5")
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("Top 15 Feature Importance\n(LGBM v2 SMOTE)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(trial_df["trial"], trial_df["macro_f1"],
         marker="o", markersize=4, color="#185FA5")
plt.xlabel("Trial")
plt.ylabel("Macro F1")
plt.title("Optuna Trial History (LGBM v2 SMOTE)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/optuna_history.png", dpi=300)
plt.close()

# SMOTE 전후 분포 비교
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
before = pd.Series(y_tr).value_counts().sort_index()
after  = pd.Series(y_tr_sm).value_counts().sort_index()
axes[0].bar(["Stage1","Stage2","Stage3"], before.values,
            color=["#185FA5","#E24B4A","#E67E22"])
axes[0].set_title("Before SMOTE")
axes[0].set_ylabel("Count")
for i, v in enumerate(before.values):
    axes[0].text(i, v+30, str(v), ha='center')
axes[1].bar(["Stage1","Stage2","Stage3"], after.values,
            color=["#185FA5","#E24B4A","#E67E22"])
axes[1].set_title("After SMOTE")
axes[1].set_ylabel("Count")
for i, v in enumerate(after.values):
    axes[1].text(i, v+30, str(v), ha='center')
plt.suptitle("SMOTE 적용 전후 Stage 분포 비교")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/smote_distribution.png", dpi=300)
plt.close()

print("Top 15 Feature Importance")
print(top15[["feature","importance"]].to_string(index=False))
print()
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

with open("models/stage2_LGBM_v2_smote.pkl", "wb") as f:
    pickle.dump({
        "model"       : model,
        "feature_cols": FEATURE_COLS,
        "remove_cols" : REMOVE_COLS,
        "best_params" : best_params,
        "version"     : "v2_smote",
        "metrics": {
            "valid_accuracy"   : accuracy,
            "valid_weighted_f1": weighted_f1,
            "valid_macro_f1"   : macro_f1,
            "valid_s1_recall"  : s1_recall,
            "valid_s2_recall"  : s2_recall,
            "valid_s3_recall"  : s3_recall,
            "test_accuracy"    : test_accuracy,
            "test_weighted_f1" : test_weighted_f1,
            "test_macro_f1"    : test_macro_f1,
            "test_s1_recall"   : ts1_recall,
            "test_s2_recall"   : ts2_recall,
            "test_s3_recall"   : ts3_recall,
        }
    }, f)

print("=" * 70)
print("MODEL SAVED : models/stage2_LGBM_v2_smote.pkl")
print("=" * 70)
print("FINISHED")