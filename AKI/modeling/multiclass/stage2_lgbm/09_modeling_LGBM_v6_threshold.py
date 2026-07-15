# ==========================================================
# 09_modeling_LGBM_v6_threshold.py
#
# Stage 2 : AKI Stage Classification
# Multi-class Classification
#
# Model : LightGBM
# Method : v1 모델 재활용 + Stage2 Threshold 조정
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
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus'] = False

from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    classification_report, confusion_matrix
)

# ==========================================================
# SETTINGS
# ==========================================================

OUT_DIR = "outputs/LGBM_stage_v6_threshold"
os.makedirs(OUT_DIR,  exist_ok=True)

# ==========================================================
# v1 모델 로딩
# ==========================================================

print("=" * 70)
print("v1 모델 로딩")
print("=" * 70)

with open("models/stage2_LGBM_v1_classweight.pkl", "rb") as f:
    v1_data = pickle.load(f)

model        = v1_data["model"]
FEATURE_COLS = v1_data["feature_cols"]
REMOVE_COLS  = v1_data["remove_cols"]

print(f"모델 버전 : {v1_data['version']}")
print(f"피처 수   : {len(FEATURE_COLS)}")
print()

# ==========================================================
# 데이터 로딩
# ==========================================================

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

keep_idx = [
    i for i, col in enumerate(ORIGINAL_FEATURE_COLS)
    if col not in REMOVE_COLS
]

X_valid_orig = np.load("data/X_valid.npy")
X_test_orig  = np.load("data/X_test.npy")
y_stage_valid = np.load("data/y_stage_valid.npy")
y_stage_test  = np.load("data/y_stage_test.npy")

valid_mask = y_stage_valid > 0
test_mask  = y_stage_test  > 0

X_stage_valid = X_valid_orig[valid_mask][:, keep_idx]
X_stage_test  = X_test_orig[test_mask][:, keep_idx]

y_vl = y_stage_valid[valid_mask] - 1
y_te = y_stage_test[test_mask]   - 1

print(f"X_stage_valid : {X_stage_valid.shape}")
print(f"X_stage_test  : {X_stage_test.shape}")
print()

# ==========================================================
# 확률 예측
# ==========================================================

valid_proba = model.predict_proba(X_stage_valid)
test_proba  = model.predict_proba(X_stage_test)

# valid_proba shape: (N, 3)
# 열 0 = Stage1 확률
# 열 1 = Stage2 확률
# 열 2 = Stage3 확률

print("=" * 70)
print("기본 예측 (threshold 없음)")
print("=" * 70)

default_pred = np.argmax(valid_proba, axis=1)
print(f"Macro F1      : {f1_score(y_vl, default_pred, average='macro'):.4f}")
print(f"Stage2 Recall : {recall_score(y_vl, default_pred, labels=[1], average=None)[0]:.4f}")
print()

# ==========================================================
# THRESHOLD 탐색
# ==========================================================

print("=" * 70)
print("Stage2 Threshold 탐색")
print("=" * 70)

threshold_results = []

for t2 in np.arange(0.05, 0.60, 0.01):
    pred = np.argmax(valid_proba, axis=1).copy()

    # Stage2 확률이 t2 이상이면 Stage2로 판정
    stage2_mask = valid_proba[:, 1] >= t2
    pred[stage2_mask] = 1

    macro_f1  = f1_score(y_vl, pred, average="macro")
    s1_recall = recall_score(y_vl, pred, labels=[0], average=None)[0]
    s2_recall = recall_score(y_vl, pred, labels=[1], average=None)[0]
    s3_recall = recall_score(y_vl, pred, labels=[2], average=None)[0]

    threshold_results.append([
        round(t2, 2), macro_f1,
        s1_recall, s2_recall, s3_recall
    ])

    print(f"t2={t2:.2f} | "
          f"Macro F1={macro_f1:.4f} | "
          f"S1={s1_recall:.3f} | "
          f"S2={s2_recall:.3f} | "
          f"S3={s3_recall:.3f}")

threshold_df = pd.DataFrame(
    threshold_results,
    columns=["threshold","macro_f1",
             "s1_recall","s2_recall","s3_recall"]
)
threshold_df.to_csv(f"{OUT_DIR}/threshold_search.csv", index=False)

# ==========================================================
# 최적 Threshold 선정
# 기준: Stage2 Recall 0.3 이상 + Macro F1 최대
# ==========================================================

print()
print("=" * 70)
print("최적 Threshold 선정")
print("=" * 70)

# 기준 1: S2 Recall 0.30 이상 + Macro F1 최대
cond1 = threshold_df[threshold_df["s2_recall"] >= 0.30]
if len(cond1) > 0:
    best_row = cond1.loc[cond1["macro_f1"].idxmax()]
    print("기준: Stage2 Recall ≥ 0.30 + Macro F1 최대")
else:
    best_row = threshold_df.loc[threshold_df["s2_recall"].idxmax()]
    print("기준: Stage2 Recall 최대 (0.30 달성 불가)")

best_threshold = best_row["threshold"]
print(f"Best Threshold : {best_threshold}")
print(f"Macro F1       : {best_row['macro_f1']:.4f}")
print(f"Stage2 Recall  : {best_row['s2_recall']:.4f}")
print()

# ==========================================================
# 최적 Threshold 적용 — Validation
# ==========================================================

def apply_threshold(proba, t2):
    pred = np.argmax(proba, axis=1).copy()
    pred[proba[:, 1] >= t2] = 1
    return pred

valid_pred = apply_threshold(valid_proba, best_threshold)
test_pred  = apply_threshold(test_proba,  best_threshold)

accuracy    = accuracy_score(y_vl, valid_pred)
weighted_f1 = f1_score(y_vl, valid_pred, average='weighted')
macro_f1    = f1_score(y_vl, valid_pred, average='macro')
s1_recall   = recall_score(y_vl, valid_pred, labels=[0], average=None)[0]
s2_recall   = recall_score(y_vl, valid_pred, labels=[1], average=None)[0]
s3_recall   = recall_score(y_vl, valid_pred, labels=[2], average=None)[0]

print("=" * 70)
print(f"VALIDATION RESULT (threshold={best_threshold})")
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

# 1. Threshold 탐색 그래프
plt.figure(figsize=(10, 6))
plt.plot(threshold_df["threshold"], threshold_df["macro_f1"],
         label="Macro F1", color="#185FA5", marker="o", markersize=3)
plt.plot(threshold_df["threshold"], threshold_df["s2_recall"],
         label="Stage2 Recall", color="#E24B4A", marker="o", markersize=3)
plt.plot(threshold_df["threshold"], threshold_df["s1_recall"],
         label="Stage1 Recall", color="#4A90D9", marker="o", markersize=3)
plt.plot(threshold_df["threshold"], threshold_df["s3_recall"],
         label="Stage3 Recall", color="#E67E22", marker="o", markersize=3)
plt.axvline(best_threshold, color="black", linestyle="--",
            label=f"Best={best_threshold}")
plt.axhline(0.30, color="#E24B4A", linestyle=":", alpha=0.5,
            label="S2 목표 (0.30)")
plt.xlabel("Stage2 Threshold")
plt.ylabel("Score")
plt.title("Stage2 Threshold 탐색 결과 (LGBM v6)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/threshold_search.png", dpi=300)
plt.close()

# 2. Confusion Matrix
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Stage1','Stage2','Stage3'],
            yticklabels=['Stage1','Stage2','Stage3'])
plt.title(f"Validation Confusion Matrix\n(LGBM v6 threshold={best_threshold})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=300)
plt.close()

# 3. Normalized Confusion Matrix
cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(7, 6))
sns.heatmap(cm_pct, annot=True, fmt='.1%', cmap='Blues',
            xticklabels=['Stage1','Stage2','Stage3'],
            yticklabels=['Stage1','Stage2','Stage3'])
plt.title(f"Normalized Confusion Matrix\n(LGBM v6 threshold={best_threshold})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix_percent.png", dpi=300)
plt.close()

# 4. Performance Summary
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
plt.title(f"Validation Performance Summary\n(LGBM v6 threshold={best_threshold})")
plt.ylabel("Score")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/performance_summary.png", dpi=300)
plt.close()

# 5. Per-class Recall
recall_vals = [s1_recall, s2_recall, s3_recall]
plt.figure(figsize=(7, 5))
bars = plt.bar(["Stage1 Recall","Stage2 Recall","Stage3 Recall"],
               recall_vals, color=["#185FA5","#E24B4A","#E67E22"])
plt.ylim(0, 1.1)
for bar, val in zip(bars, recall_vals):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.02,
             f"{val:.4f}", ha="center", fontsize=12)
plt.axhline(0.30, color="red", linestyle="--", alpha=0.5,
            label="목표 0.30")
plt.title(f"Per-class Recall (LGBM v6 threshold={best_threshold})")
plt.ylabel("Recall")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/per_class_recall.png", dpi=300)
plt.close()

# 6. v1 기본 vs v6 threshold 비교
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
default_cm = confusion_matrix(y_vl, default_pred)
sns.heatmap(default_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['S1','S2','S3'],
            yticklabels=['S1','S2','S3'])
axes[0].set_title("v1 기본 (argmax)")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
            xticklabels=['S1','S2','S3'],
            yticklabels=['S1','S2','S3'])
axes[1].set_title(f"v6 threshold={best_threshold}")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")
plt.suptitle("Confusion Matrix 비교: v1 기본 vs v6 Threshold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/comparison_cm.png", dpi=300)
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

with open("models/stage2_LGBM_v6_threshold.pkl", "wb") as f:
    pickle.dump({
        "model"         : model,
        "feature_cols"  : FEATURE_COLS,
        "best_threshold": best_threshold,
        "version"       : "v6_threshold",
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
print("MODEL SAVED : models/stage2_LGBM_v6_threshold.pkl")
print("=" * 70)
print("FINISHED")