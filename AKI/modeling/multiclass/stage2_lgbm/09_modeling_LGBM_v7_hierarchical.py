# ==========================================================
# 09_modeling_LGBM_v7_hierarchical.py
#
# Stage 2 : AKI Stage Classification
# Method : 계층적 분류 (Hierarchical Classification)
#
# Step 1: Stage1 vs (Stage2+3) 이진 분류
# Step 2: Stage2 vs Stage3 이진 분류
#
# Feature : Excluded (28개, KDIGO 7개 제거)
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

OUT_DIR = "outputs/LGBM_stage_v7_hierarchical"
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

train_mask = y_stage_train > 0
valid_mask = y_stage_valid > 0
test_mask  = y_stage_test  > 0

X_tr = X_train_orig[train_mask][:, keep_idx]
X_vl = X_valid_orig[valid_mask][:, keep_idx]
X_te = X_test_orig[test_mask][:, keep_idx]

y_tr = y_stage_train[train_mask] - 1
y_vl = y_stage_valid[valid_mask] - 1
y_te = y_stage_test[test_mask]   - 1

print(f"Train : {X_tr.shape} | Stage 분포: "
      f"S1={sum(y_tr==0)} S2={sum(y_tr==1)} S3={sum(y_tr==2)}")
print(f"Valid : {X_vl.shape} | Stage 분포: "
      f"S1={sum(y_vl==0)} S2={sum(y_vl==1)} S3={sum(y_vl==2)}")
print(f"Test  : {X_te.shape} | Stage 분포: "
      f"S1={sum(y_te==0)} S2={sum(y_te==1)} S3={sum(y_te==2)}")
print()

# ==========================================================
# STEP 1 데이터: Stage1(0) vs Stage2+3(1)
# ==========================================================

print("=" * 70)
print("STEP 1: Stage1 vs (Stage2+3) 이진 분류")
print("=" * 70)

y_step1_tr = (y_tr >= 1).astype(int)  # Stage2+3=1, Stage1=0
y_step1_vl = (y_vl >= 1).astype(int)
y_step1_te = (y_te >= 1).astype(int)

print(f"Train: Stage1={sum(y_step1_tr==0)}, Stage2+3={sum(y_step1_tr==1)}")
print(f"Valid: Stage1={sum(y_step1_vl==0)}, Stage2+3={sum(y_step1_vl==1)}")
print()

# Step1 class weight
w1 = compute_class_weight("balanced",
     classes=np.array([0,1]), y=y_step1_tr)
sw1 = np.array([{0:w1[0],1:w1[1]}[y] for y in y_step1_tr])
print(f"Step1 class weight: Stage1={w1[0]:.4f}, Stage2+3={w1[1]:.4f}")
print()

# Step1 Optuna
trial_history_1 = []

def objective_step1(trial):
    params = {
        "objective"        : "binary",
        "metric"           : "binary_logloss",
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
    m = LGBMClassifier(**params)
    m.fit(X_tr, y_step1_tr, sample_weight=sw1)
    pred  = m.predict(X_vl)
    score = recall_score(y_step1_vl, pred, labels=[1], average=None)[0]
    trial_history_1.append([trial.number, score])
    print(f"  Trial {trial.number:03d} | Stage2+3 Recall = {score:.5f}")
    return score

print("Optuna Step1 (30 trials)")
study1 = optuna.create_study(direction="maximize")
study1.optimize(objective_step1, n_trials=30)

print(f"\nBest Stage2+3 Recall : {study1.best_value:.5f}")

bp1 = study1.best_params
bp1.update({"objective":"binary","metric":"binary_logloss",
            "random_state":42,"n_jobs":-1,"verbose":-1})

model_step1 = LGBMClassifier(**bp1)
model_step1.fit(X_tr, y_step1_tr, sample_weight=sw1)

# Step1 Threshold 탐색
proba1_vl = model_step1.predict_proba(X_vl)[:, 1]
best_t1, best_s1_score = 0.5, 0
for t in np.arange(0.10, 0.90, 0.01):
    pred = (proba1_vl >= t).astype(int)
    s = recall_score(y_step1_vl, pred, labels=[1], average=None)[0]
    f = f1_score(y_step1_vl, pred, average="macro")
    if s >= 0.70 and f > best_s1_score:
        best_s1_score = f
        best_t1 = t

pred1_vl = (proba1_vl >= best_t1).astype(int)
pred1_te = (model_step1.predict_proba(X_te)[:, 1] >= best_t1).astype(int)

s23_recall = recall_score(y_step1_vl, pred1_vl, labels=[1], average=None)[0]
s1_prec    = recall_score(y_step1_vl, pred1_vl, labels=[0], average=None)[0]

print(f"\nStep1 Best Threshold : {best_t1:.2f}")
print(f"Stage1 Recall        : {s1_prec:.4f}")
print(f"Stage2+3 Recall      : {s23_recall:.4f}")
print()

# ==========================================================
# STEP 2 데이터: Stage2(0) vs Stage3(1) — Stage2+3 환자만
# ==========================================================

print("=" * 70)
print("STEP 2: Stage2 vs Stage3 이진 분류")
print("=" * 70)

step2_mask_tr = y_tr >= 1
step2_mask_vl = y_vl >= 1
step2_mask_te = y_te >= 1

X_tr2 = X_tr[step2_mask_tr]
X_vl2 = X_vl[step2_mask_vl]
X_te2 = X_te[step2_mask_te]

y_step2_tr = (y_tr[step2_mask_tr] == 2).astype(int)  # Stage3=1, Stage2=0
y_step2_vl = (y_vl[step2_mask_vl] == 2).astype(int)
y_step2_te = (y_te[step2_mask_te] == 2).astype(int)

print(f"Train: Stage2={sum(y_step2_tr==0)}, Stage3={sum(y_step2_tr==1)}")
print(f"Valid: Stage2={sum(y_step2_vl==0)}, Stage3={sum(y_step2_vl==1)}")
print()

w2 = compute_class_weight("balanced",
     classes=np.array([0,1]), y=y_step2_tr)
sw2 = np.array([{0:w2[0],1:w2[1]}[y] for y in y_step2_tr])
print(f"Step2 class weight: Stage2={w2[0]:.4f}, Stage3={w2[1]:.4f}")
print()

# Step2 Optuna
trial_history_2 = []

def objective_step2(trial):
    params = {
        "objective"        : "binary",
        "metric"           : "binary_logloss",
        "n_estimators"     : trial.suggest_int("n_estimators", 100, 500),
        "max_depth"        : trial.suggest_int("max_depth", 3, 8),
        "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves"       : trial.suggest_int("num_leaves", 20, 60),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        "subsample"        : trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha"        : trial.suggest_float("reg_alpha", 0, 1.0),
        "reg_lambda"       : trial.suggest_float("reg_lambda", 0, 1.0),
        "random_state"     : 42,
        "n_jobs"           : -1,
        "verbose"          : -1,
    }
    m = LGBMClassifier(**params)
    m.fit(X_tr2, y_step2_tr, sample_weight=sw2)
    pred  = m.predict(X_vl2)
    # Stage2 Recall 최대화 (Stage2=0이므로 labels=[0])
    s2r   = recall_score(y_step2_vl, pred, labels=[0], average=None)[0]
    macro = f1_score(y_step2_vl, pred, average="macro")
    score = s2r * 0.6 + macro * 0.4
    trial_history_2.append([trial.number, s2r, macro, score])
    print(f"  Trial {trial.number:03d} | S2 Recall={s2r:.4f} | Macro={macro:.4f}")
    return score

print("Optuna Step2 (30 trials)")
study2 = optuna.create_study(direction="maximize")
study2.optimize(objective_step2, n_trials=30)

print(f"\nBest Score : {study2.best_value:.5f}")

bp2 = study2.best_params
bp2.update({"objective":"binary","metric":"binary_logloss",
            "random_state":42,"n_jobs":-1,"verbose":-1})

model_step2 = LGBMClassifier(**bp2)
model_step2.fit(X_tr2, y_step2_tr, sample_weight=sw2)

# Step2 Threshold 탐색
proba2_vl = model_step2.predict_proba(X_vl2)[:, 1]
best_t2, best_s2_score = 0.5, 0
for t in np.arange(0.10, 0.90, 0.01):
    pred = (proba2_vl >= t).astype(int)
    s2r  = recall_score(y_step2_vl, pred, labels=[0], average=None)[0]
    s3r  = recall_score(y_step2_vl, pred, labels=[1], average=None)[0]
    score = s2r * 0.6 + s3r * 0.4
    if score > best_s2_score:
        best_s2_score = score
        best_t2 = t

pred2_vl = (model_step2.predict_proba(X_vl2)[:, 1] >= best_t2).astype(int)
pred2_te = (model_step2.predict_proba(X_te2)[:, 1] >= best_t2).astype(int)

s2_recall_step = recall_score(y_step2_vl, pred2_vl, labels=[0], average=None)[0]
s3_recall_step = recall_score(y_step2_vl, pred2_vl, labels=[1], average=None)[0]

print(f"\nStep2 Best Threshold : {best_t2:.2f}")
print(f"Stage2 Recall (Step2): {s2_recall_step:.4f}")
print(f"Stage3 Recall (Step2): {s3_recall_step:.4f}")
print()

# ==========================================================
# 계층적 최종 예측 조합
# ==========================================================

print("=" * 70)
print("최종 예측 조합")
print("=" * 70)

def hierarchical_predict(X, model1, t1, model2, t2):
    # Step1: Stage1 vs Stage2+3
    proba1 = model1.predict_proba(X)[:, 1]
    step1  = (proba1 >= t1).astype(int)

    final_pred = np.zeros(len(X), dtype=int)

    # Stage1으로 판정된 경우
    final_pred[step1 == 0] = 0

    # Stage2+3으로 판정된 경우 → Step2로 세분화
    idx_23 = np.where(step1 == 1)[0]
    if len(idx_23) > 0:
        X_23   = X[idx_23]
        proba2 = model2.predict_proba(X_23)[:, 1]
        step2  = (proba2 >= t2).astype(int)
        final_pred[idx_23[step2 == 0]] = 1  # Stage2
        final_pred[idx_23[step2 == 1]] = 2  # Stage3

    return final_pred

valid_pred = hierarchical_predict(
    X_vl, model_step1, best_t1, model_step2, best_t2
)
test_pred  = hierarchical_predict(
    X_te, model_step1, best_t1, model_step2, best_t2
)

# ==========================================================
# 최종 평가
# ==========================================================

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
print("VALIDATION RESULT (계층적 분류)")
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
plt.title("Validation Confusion Matrix\n(LGBM v7 Hierarchical)")
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
plt.title("Normalized Confusion Matrix\n(LGBM v7 Hierarchical)")
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
plt.title("Validation Performance Summary\n(LGBM v7 Hierarchical)")
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
plt.title("Per-class Recall (LGBM v7 Hierarchical)")
plt.ylabel("Recall")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/per_class_recall.png", dpi=300)
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
# SAVE
# ==========================================================

with open("models/stage2_LGBM_v7_hierarchical.pkl", "wb") as f:
    pickle.dump({
        "model_step1"  : model_step1,
        "model_step2"  : model_step2,
        "threshold_t1" : best_t1,
        "threshold_t2" : best_t2,
        "feature_cols" : FEATURE_COLS,
        "version"      : "v7_hierarchical",
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
print("MODEL SAVED : models/stage2_LGBM_v7_hierarchical.pkl")
print("=" * 70)
print("FINISHED")