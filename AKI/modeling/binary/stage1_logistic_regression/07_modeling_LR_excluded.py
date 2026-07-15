# ==========================================================
# 07_modeling_LR_excluded.py
#
# Stage 1 : AKI Prediction
# Binary Classification
# Model : Logistic Regression
# Feature : Excluded (29개, KDIGO 6개 제거)
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

matplotlib.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus'] = False

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve
)

# ==========================================================
# SETTINGS
# ==========================================================

OUT_DIR = "outputs/LR_excluded"
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

EXCLUDED_FEATURES = [
    'urine_output_sum', 'urine_output_6h', 'oliguria_flag',
    'creatinine_min', 'creatinine_max', 'creatinine_delta'
]

selected_idx = [
    i for i, col in enumerate(ORIGINAL_FEATURE_COLS)
    if col not in EXCLUDED_FEATURES
]
FEATURE_COLS = [
    col for col in ORIGINAL_FEATURE_COLS
    if col not in EXCLUDED_FEATURES
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
y_train = np.load("data/y_train.npy")
y_valid = np.load("data/y_valid.npy")
y_test  = np.load("data/y_test.npy")

X_train = X_train_orig[:, selected_idx]
X_valid = X_valid_orig[:, selected_idx]
X_test  = X_test_orig[:, selected_idx]

print(f"X_train : {X_train.shape}")
print(f"X_valid : {X_valid.shape}")
print(f"X_test  : {X_test.shape}")
print()
print("AKI Ratio")
print(f"Train : {y_train.mean():.4f}")
print(f"Valid : {y_valid.mean():.4f}")
print(f"Test  : {y_test.mean():.4f}")
print()

print("=" * 70)
print("FEATURE EXCLUSION")
print("=" * 70)
print("Removed Features")
for col in EXCLUDED_FEATURES:
    print(f"  - {col}")
print()
print(f"Final Feature Count : {len(FEATURE_COLS)}")
print()

n_pos = int(y_train.sum())
n_neg = int(len(y_train) - n_pos)
print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)
print(f"AKI     : {n_pos:,}")
print(f"Non-AKI : {n_neg:,}")
print()

# ==========================================================
# MODEL TRAINING
# ==========================================================

print("=" * 70)
print("LOGISTIC REGRESSION TRAINING")
print("=" * 70)

model = LogisticRegression(
    class_weight = 'balanced',
    max_iter     = 1000,
    solver       = 'lbfgs',
    random_state = 42,
    n_jobs       = -1
)
model.fit(X_train, y_train)
print("학습 완료")
print()

train_prob = model.predict_proba(X_train)[:, 1]
valid_prob = model.predict_proba(X_valid)[:, 1]
test_prob  = model.predict_proba(X_test)[:, 1]

# ==========================================================
# THRESHOLD SEARCH
# ==========================================================

print("=" * 70)
print("THRESHOLD OPTIMIZATION")
print("=" * 70)

threshold_results = []
best_threshold    = 0.5
best_specificity  = 0

for threshold in np.arange(0.10, 0.91, 0.01):
    pred = (valid_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_valid, pred).ravel()
    precision   = precision_score(y_valid, pred, zero_division=0)
    recall      = recall_score(y_valid, pred, zero_division=0)
    specificity = tn / (tn + fp)
    threshold_results.append([threshold, precision, recall, specificity])
    if recall >= 0.75 and specificity > best_specificity:
        best_specificity = specificity
        best_threshold   = threshold

threshold_df = pd.DataFrame(
    threshold_results,
    columns=["threshold", "precision", "recall", "specificity"]
)
threshold_df.to_csv(f"{OUT_DIR}/threshold_search.csv", index=False)
print(f"Best Threshold : {best_threshold:.2f}")
print(f"Specificity    : {best_specificity:.4f}")
print()

# ==========================================================
# EVALUATION
# ==========================================================

valid_pred = (valid_prob >= best_threshold).astype(int)
test_pred  = (test_prob  >= best_threshold).astype(int)

train_auroc = roc_auc_score(y_train, train_prob)
train_auprc = average_precision_score(y_train, train_prob)
val_auroc   = roc_auc_score(y_valid, valid_prob)
val_auprc   = average_precision_score(y_valid, valid_prob)
val_prec    = precision_score(y_valid, valid_pred, zero_division=0)
val_rec     = recall_score(y_valid, valid_pred, zero_division=0)
val_f1      = f1_score(y_valid, valid_pred, zero_division=0)

tn, fp, fn, tp  = confusion_matrix(y_valid, valid_pred).ravel()
val_sensitivity = tp / (tp + fn)
val_specificity = tn / (tn + fp)

test_auroc  = roc_auc_score(y_test, test_prob)
test_auprc  = average_precision_score(y_test, test_prob)
test_prec   = precision_score(y_test, test_pred, zero_division=0)
test_rec    = recall_score(y_test, test_pred, zero_division=0)
test_f1     = f1_score(y_test, test_pred, zero_division=0)
tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, test_pred).ravel()
test_sensitivity = tp_t / (tp_t + fn_t)
test_specificity = tn_t / (tn_t + fp_t)

print("=" * 70)
print("TRAIN RESULT")
print("=" * 70)
print(f"AUROC : {train_auroc:.4f}")
print(f"AUPRC : {train_auprc:.4f}")
print()

print("=" * 70)
print("VALIDATION RESULT")
print("=" * 70)
print(f"AUROC       : {val_auroc:.4f}")
print(f"AUPRC       : {val_auprc:.4f}")
print(f"Precision   : {val_prec:.4f}")
print(f"Recall      : {val_rec:.4f}")
print(f"F1-score    : {val_f1:.4f}")
print(f"Sensitivity : {val_sensitivity:.4f}")
print(f"Specificity : {val_specificity:.4f}")
print()
print(f"Train AUROC      : {train_auroc:.4f}")
print(f"Validation AUROC : {val_auroc:.4f}")
print(f"Train AUPRC      : {train_auprc:.4f}")
print(f"Validation AUPRC : {val_auprc:.4f}")
print()

print("=" * 70)
print("TEST RESULT")
print("=" * 70)
print(f"AUROC       : {test_auroc:.4f}")
print(f"AUPRC       : {test_auprc:.4f}")
print(f"Precision   : {test_prec:.4f}")
print(f"Recall      : {test_rec:.4f}")
print(f"F1-score    : {test_f1:.4f}")
print(f"Sensitivity : {test_sensitivity:.4f}")
print(f"Specificity : {test_specificity:.4f}")
print()

cm = confusion_matrix(y_valid, valid_pred)
print("Confusion Matrix (Validation)")
print(cm)
print()

# ==========================================================
# FIGURES
# ==========================================================

print("=" * 70)
print("GENERATE FIGURES")
print("=" * 70)

# 1. Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Validation Confusion Matrix (LR Excluded)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/validation_confusion_matrix.png", dpi=300)
plt.close()

# 2. Normalized Confusion Matrix
cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(6, 5))
plt.imshow(cm_pct, interpolation="nearest", cmap="Blues")
plt.colorbar()
for i in range(2):
    for j in range(2):
        plt.text(j, i, f"{cm_pct[i,j]:.1%}",
                 ha="center", va="center", fontsize=13)
plt.xticks([0, 1], ["Non-AKI", "AKI"])
plt.yticks([0, 1], ["Non-AKI", "AKI"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Normalized Confusion Matrix (LR Excluded)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix_percent.png", dpi=300)
plt.close()

# 3. ROC Curve
fpr, tpr, _ = roc_curve(y_valid, valid_prob)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUROC = {val_auroc:.4f}", color="#185FA5")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (LR Excluded)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/roc_curve.png", dpi=300)
plt.close()

# 4. PR Curve
precisions, recalls, _ = precision_recall_curve(y_valid, valid_prob)
plt.figure(figsize=(6, 6))
plt.plot(recalls, precisions, color="#185FA5")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"PR Curve (LR Excluded) | AUPRC={val_auprc:.4f}")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/pr_curve.png", dpi=300)
plt.close()

# 5. Probability Distribution
plt.figure(figsize=(8, 5))
plt.hist(valid_prob[y_valid == 0], bins=50,
         alpha=0.5, label="Non-AKI", color="#4A90D9")
plt.hist(valid_prob[y_valid == 1], bins=50,
         alpha=0.5, label="AKI",     color="#E24B4A")
plt.axvline(best_threshold, color="black", linestyle="--",
            label=f"Threshold={best_threshold:.2f}")
plt.xlabel("Predicted Probability")
plt.ylabel("Count")
plt.title("Probability Distribution (LR Excluded)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/probability_distribution.png", dpi=300)
plt.close()

# 6. Threshold Optimization
plt.figure(figsize=(8, 5))
plt.plot(threshold_df["threshold"], threshold_df["recall"],
         label="Recall (Sensitivity)", color="#E24B4A")
plt.plot(threshold_df["threshold"], threshold_df["specificity"],
         label="Specificity", color="#185FA5")
plt.axvline(best_threshold, color="black", linestyle="--",
            label=f"Best={best_threshold:.2f}")
plt.axhline(0.75, color="#E24B4A", linestyle=":", alpha=0.5,
            label="민감도 기준 (0.75)")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Threshold Optimization (LR Excluded)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/threshold_optimization.png", dpi=300)
plt.close()

# 7. Performance Summary
metrics = {
    "AUROC": val_auroc, "AUPRC": val_auprc,
    "Precision": val_prec, "Recall": val_rec,
    "F1": val_f1, "Specificity": val_specificity
}
plt.figure(figsize=(9, 5))
bars = plt.bar(metrics.keys(), metrics.values(), color="#185FA5")
plt.ylim(0, 1.1)
for bar, val in zip(bars, metrics.values()):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.02,
             f"{val:.4f}", ha="center", fontsize=10)
plt.title("Validation Performance Summary (LR Excluded)")
plt.ylabel("Score")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/performance_summary.png", dpi=300)
plt.close()

# 8. Train / Valid / Test AUROC
plt.figure(figsize=(6, 5))
plt.bar(["Train", "Validation", "Test"],
        [train_auroc, val_auroc, test_auroc],
        color=["#4A90D9", "#185FA5", "#1F4080"])
plt.ylim(0.60, 0.80)
for i, v in enumerate([train_auroc, val_auroc, test_auroc]):
    plt.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=11)
plt.ylabel("AUROC")
plt.title("Train / Validation / Test AUROC (LR Excluded)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/train_val_test_auroc.png", dpi=300)
plt.close()

# 9. Feature Importance (Coefficient)
coef_df = pd.DataFrame({
    "feature"    : FEATURE_COLS,
    "coefficient": model.coef_[0]
}).sort_values("coefficient", key=abs, ascending=False)
coef_df.to_csv(f"{OUT_DIR}/feature_importance.csv", index=False)

top15  = coef_df.head(15)
colors = ["#E24B4A" if v > 0 else "#185FA5"
          for v in top15["coefficient"]]
plt.figure(figsize=(10, 8))
plt.barh(top15["feature"], top15["coefficient"], color=colors)
plt.gca().invert_yaxis()
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Coefficient")
plt.title("Top 15 Feature Coefficients (LR Excluded)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=300)
plt.close()

print("Top 15 Feature Coefficients")
print(top15[["feature", "coefficient"]].to_string(index=False))
print()
print(f"Saved → {OUT_DIR}")
print()

# ==========================================================
# SAVE MODEL
# ==========================================================

with open("models/stage1_LR_excluded.pkl", "wb") as f:
    pickle.dump({
        "model"          : model,
        "best_threshold" : best_threshold,
        "feature_cols"   : FEATURE_COLS,
        "excluded_cols"  : EXCLUDED_FEATURES,
        "metrics": {
            "valid_auroc"      : val_auroc,
            "valid_auprc"      : val_auprc,
            "valid_sensitivity": val_sensitivity,
            "valid_specificity": val_specificity,
            "valid_f1"         : val_f1,
            "test_auroc"       : test_auroc,
            "test_auprc"       : test_auprc,
            "test_sensitivity" : test_sensitivity,
            "test_specificity" : test_specificity,
            "test_f1"          : test_f1,
        }
    }, f)

print("=" * 70)
print("MODEL SAVED : models/stage1_LR_excluded.pkl")
print("=" * 70)
print("FINISHED")