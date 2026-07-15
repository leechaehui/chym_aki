# ==========================================================
# 09_modeling_LGBM_v9_binary_ensemble.py
#
# Stage 2 : AKI Stage Classification
# Method : 이진 앙상블
#           Model A: Stage2 vs Non-Stage2
#           Model B: Stage3 vs Non-Stage3
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

OUT_DIR = "outputs/LGBM_stage_v9_binary_ensemble"
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

print(f"Train : {X_tr.shape} S1={sum(y_tr==0)} S2={sum(y_tr==1)} S3={sum(y_tr==2)}")
print(f"Valid : {X_vl.shape} S1={sum(y_vl==0)} S2={sum(y_vl==1)} S3={sum(y_vl==2)}")
print()

# ==========================================================
# MODEL A: Stage2 vs Non-Stage2
# ==========================================================

print("=" * 70)
print("MODEL A: Stage2 vs Non-Stage2")
print("=" * 70)

y_a_tr = (y_tr == 1).astype(int)  # Stage2=1, 나머지=0
y_a_vl = (y_vl == 1).astype(int)
y_a_te = (y_te == 1).astype(int)

print(f"Train: Non-Stage2={sum(y_a_tr==0)}, Stage2={sum(y_a_tr==1)}")
print(f"Valid: Non-Stage2={sum(y_a_vl==0)}, Stage2={sum(y_a_vl==1)}")

wa = compute_class_weight("balanced",
     classes=np.array([0,1]), y=y_a_tr)
swa = np.array([{0:wa[0],1:wa[1]}[y] for y in y_a_tr])
print(f"Weight: Non-Stage2={wa[0]:.4f}, Stage2={wa[1]:.4f}")
print()

trial_hist_a = []

def objective_a(trial):
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
    m.fit(X_tr, y_a_tr, sample_weight=swa)
    pred = m.predict(X_vl)
    s2r  = recall_score(y_a_vl, pred, labels=[1], average=None)[0]
    mf1  = f1_score(y_a_vl, pred, average="macro")
    sc   = s2r * 0.7 + mf1 * 0.3
    trial_hist_a.append([trial.number, s2r, mf1])
    print(f"  Trial {trial.number:03d} | S2 Recall={s2r:.4f} | Macro={mf1:.4f}")
    return sc

print("Optuna Model A (30 trials)")
study_a = optuna.create_study(direction="maximize")
study_a.optimize(objective_a, n_trials=30)

bp_a = study_a.best_params
bp_a.update({"objective":"binary","metric":"binary_logloss",
             "random_state":42,"n_jobs":-1,"verbose":-1})

model_a = LGBMClassifier(**bp_a)
model_a.fit(X_tr, y_a_tr, sample_weight=swa)

# Model A Threshold 탐색
proba_a_vl = model_a.predict_proba(X_vl)[:, 1]
best_ta, best_ta_score = 0.5, 0
for t in np.arange(0.05, 0.60, 0.01):
    pred = (proba_a_vl >= t).astype(int)
    s2r  = recall_score(y_a_vl, pred, labels=[1], average=None)[0]
    mf1  = f1_score(y_a_vl, pred, average="macro")
    sc   = s2r * 0.7 + mf1 * 0.3
    if sc > best_ta_score:
        best_ta_score = sc
        best_ta = t

proba_a_te = model_a.predict_proba(X_te)[:, 1]
s2r_a = recall_score(y_a_vl,
    (proba_a_vl >= best_ta).astype(int),
    labels=[1], average=None)[0]
print(f"\nModel A Best Threshold : {best_ta:.2f}")
print(f"Stage2 Recall          : {s2r_a:.4f}")
print()

# ==========================================================
# MODEL B: Stage3 vs Non-Stage3
# ==========================================================

print("=" * 70)
print("MODEL B: Stage3 vs Non-Stage3")
print("=" * 70)

y_b_tr = (y_tr == 2).astype(int)  # Stage3=1, 나머지=0
y_b_vl = (y_vl == 2).astype(int)
y_b_te = (y_te == 2).astype(int)

print(f"Train: Non-Stage3={sum(y_b_tr==0)}, Stage3={sum(y_b_tr==1)}")
print(f"Valid: Non-Stage3={sum(y_b_vl==0)}, Stage3={sum(y_b_vl==1)}")

wb = compute_class_weight("balanced",
     classes=np.array([0,1]), y=y_b_tr)
swb = np.array([{0:wb[0],1:wb[1]}[y] for y in y_b_tr])
print(f"Weight: Non-Stage3={wb[0]:.4f}, Stage3={wb[1]:.4f}")
print()

trial_hist_b = []

def objective_b(trial):
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
    m.fit(X_tr, y_b_tr, sample_weight=swb)
    pred = m.predict(X_vl)
    s3r  = recall_score(y_b_vl, pred, labels=[1], average=None)[0]
    mf1  = f1_score(y_b_vl, pred, average="macro")
    sc   = s3r * 0.6 + mf1 * 0.4
    trial_hist_b.append([trial.number, s3r, mf1])
    print(f"  Trial {trial.number:03d} | S3 Recall={s3r:.4f} | Macro={mf1:.4f}")
    return sc

print("Optuna Model B (30 trials)")
study_b = optuna.create_study(direction="maximize")
study_b.optimize(objective_b, n_trials=30)

bp_b = study_b.best_params
bp_b.update({"objective":"binary","metric":"binary_logloss",
             "random_state":42,"n_jobs":-1,"verbose":-1})

model_b = LGBMClassifier(**bp_b)
model_b.fit(X_tr, y_b_tr, sample_weight=swb)

proba_b_vl = model_b.predict_proba(X_vl)[:, 1]
best_tb, best_tb_score = 0.5, 0
for t in np.arange(0.05, 0.90, 0.01):
    pred = (proba_b_vl >= t).astype(int)
    s3r  = recall_score(y_b_vl, pred, labels=[1], average=None)[0]
    mf1  = f1_score(y_b_vl, pred, average="macro")
    sc   = s3r * 0.6 + mf1 * 0.4
    if sc > best_tb_score:
        best_tb_score = sc
        best_tb = t

proba_b_te = model_b.predict_proba(X_te)[:, 1]
s3r_b = recall_score(y_b_vl,
    (proba_b_vl >= best_tb).astype(int),
    labels=[1], average=None)[0]
print(f"\nModel B Best Threshold : {best_tb:.2f}")
print(f"Stage3 Recall          : {s3r_b:.4f}")
print()

# ==========================================================
# 앙상블 예측 조합 — 수정 버전
# Stage3 우선순위 + threshold 조정
# ==========================================================

print("=" * 70)
print("앙상블 예측 조합 (수정: Stage3 우선)")
print("=" * 70)

def ensemble_predict_v2(proba_a, proba_b, ta, tb):
    n = len(proba_a)
    pred = np.zeros(n, dtype=int)  # 기본 Stage1

    is_s2 = proba_a >= ta
    is_s3 = proba_b >= tb

    # Stage2 먼저 적용
    pred[is_s2] = 1

    # Stage3 나중에 적용 (Stage3 우선)
    pred[is_s3] = 2

    return pred

# threshold 조합 탐색
print("Threshold 조합 탐색")
print(f"{'ta':>6} {'tb':>6} | {'Macro':>6} {'S1':>6} {'S2':>6} {'S3':>6}")
print("-" * 50)

best_combo = {"score": 0}

for ta in [0.30, 0.35, 0.40, 0.45, 0.50]:
    for tb in [0.20, 0.25, 0.30, 0.35, 0.40]:
        pred = ensemble_predict_v2(proba_a_vl, proba_b_vl, ta, tb)
        mf1  = f1_score(y_vl, pred, average="macro")
        s1r  = recall_score(y_vl, pred, labels=[0], average=None)[0]
        s2r  = recall_score(y_vl, pred, labels=[1], average=None)[0]
        s3r  = recall_score(y_vl, pred, labels=[2], average=None)[0]

        # 세 클래스 모두 0.20 이상이면 점수 계산
        if s1r > 0.10 and s2r > 0.10 and s3r > 0.10:
            score = mf1
            if score > best_combo["score"]:
                best_combo = {
                    "score": score, "ta": ta, "tb": tb,
                    "mf1": mf1, "s1r": s1r,
                    "s2r": s2r, "s3r": s3r
                }

        print(f"{ta:>6.2f} {tb:>6.2f} | "
              f"{mf1:>6.4f} {s1r:>6.3f} {s2r:>6.3f} {s3r:>6.3f}")

print()
if best_combo["score"] > 0:
    print(f"최적 조합: ta={best_combo['ta']}, tb={best_combo['tb']}")
    print(f"Macro F1  : {best_combo['mf1']:.4f}")
    print(f"S1 Recall : {best_combo['s1r']:.4f}")
    print(f"S2 Recall : {best_combo['s2r']:.4f}")
    print(f"S3 Recall : {best_combo['s3r']:.4f}")
    best_ta = best_combo["ta"]
    best_tb = best_combo["tb"]
else:
    print("모든 클래스 0.10 이상 달성 불가 → 기본값 사용")
    best_ta = 0.40
    best_tb = 0.30

valid_pred = ensemble_predict_v2(
    proba_a_vl, proba_b_vl, best_ta, best_tb
)
test_pred = ensemble_predict_v2(
    proba_a_te, proba_b_te, best_ta, best_tb
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
print("VALIDATION RESULT (이진 앙상블)")
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
plt.title("Validation Confusion Matrix\n(LGBM v9 Binary Ensemble)")
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
plt.title("Normalized Confusion Matrix\n(LGBM v9 Binary Ensemble)")
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
plt.title("Validation Performance Summary\n(LGBM v9 Binary Ensemble)")
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
plt.axhline(0.30, color="red", linestyle="--",
            alpha=0.5, label="목표 0.30")
plt.title("Per-class Recall (LGBM v9 Binary Ensemble)")
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
# SAVE MODEL
# ==========================================================

with open("models/stage2_LGBM_v9_binary_ensemble.pkl", "wb") as f:
    pickle.dump({
        "model_a"     : model_a,
        "model_b"     : model_b,
        "threshold_a" : best_ta,
        "threshold_b" : best_tb,
        "feature_cols": FEATURE_COLS,
        "version"     : "v9_binary_ensemble",
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
print("MODEL SAVED : models/stage2_LGBM_v9_binary_ensemble.pkl")
print("=" * 70)
print("FINISHED")