# ==========================================================
# FINAL AKI DIAGNOSTIC MODEL
#
# Stage 1:
#   Logistic Regression
#   Non-AKI vs AKI
#
# Stage 2:
#   LightGBM
#   Stage1 vs Stage2+3
#
# Final Output:
#   0 = Non-AKI
#   1 = AKI Stage1
#   2 = AKI Stage2+
# ==========================================================

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = CURRENT_DIR / "models"

print("="*70)
print("DATA DIR :", DATA_DIR)
print("MODEL DIR:", MODEL_DIR)
print("="*70)
# ----------------------------------------------------------
# LOAD MODELS
# ----------------------------------------------------------

with open(MODEL_DIR / "stage1_LR_full.pkl", "rb") as f:
    lr_bundle = pickle.load(f)

with open(MODEL_DIR / "stage2_LGBM_v13_full_classweight.pkl", "rb") as f:
    lgbm_bundle = pickle.load(f)

lr_model = lr_bundle["model"]
lr_threshold = lr_bundle["best_threshold"]

lgbm_model = lgbm_bundle["model"]
lgbm_threshold = lgbm_bundle["best_threshold"]

feature_cols = lr_bundle["feature_cols"]

print("Model Loaded")
print(f"LR threshold   : {lr_threshold:.2f}")
print(f"LGBM threshold : {lgbm_threshold:.2f}")

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------

X_test = np.load(DATA_DIR / "X_test.npy")

# ----------------------------------------------------------
# PROBABILITY
# ----------------------------------------------------------

# Stage1 : AKI 여부
p_aki = lr_model.predict_proba(X_test)[:, 1]

# Stage2 : Stage2+3 여부
p_severe = lgbm_model.predict_proba(X_test)[:, 1]

# ----------------------------------------------------------
# SOFT PROBABILITY FUSION
# ----------------------------------------------------------

p_nonaki  = 1.0 - p_aki
p_stage1  = p_aki * (1.0 - p_severe)
p_stage23 = p_aki * p_severe

final_prob = np.column_stack([
    p_nonaki,
    p_stage1,
    p_stage23
])

# ----------------------------------------------------------
# FINAL CLASS
# ----------------------------------------------------------

final_pred = np.argmax(final_prob, axis=1)

# ----------------------------------------------------------
# LABEL
# ----------------------------------------------------------

label_map = {
    0: "Non-AKI",
    1: "AKI Stage1",
    2: "AKI Stage2+3"
}

result_df = pd.DataFrame({
    "P_NonAKI"   : p_nonaki,
    "P_Stage1"   : p_stage1,
    "P_Stage2+3" : p_stage23,
    "Prediction" : final_pred
})

result_df["Label"] = result_df["Prediction"].map(label_map)

print(result_df.head())

result_df.to_csv(
    "outputs/final_aki_prediction.csv",
    index=False
)

print()
print("=" * 70)
print("FINAL AKI MODEL COMPLETE")
print("=" * 70)
print("0 = Non-AKI")
print("1 = AKI Stage1")
print("2 = AKI Stage2+3")
print()
print("Saved -> outputs/final_aki_prediction.csv")


# ==========================================================
# FINAL MODEL EVALUATION
# Non-AKI / Stage1 / Stage2+3
# ==========================================================

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize

os.makedirs("outputs/final_model", exist_ok=True)

# ----------------------------------------------------------
# TRUE LABEL 생성
#
# y_stage:
# 0 = Non-AKI
# 1 = Stage1
# 2 = Stage2
# 3 = Stage3
#
# 최종:
# 0 = Non-AKI
# 1 = Stage1
# 2 = Stage2+3
# ----------------------------------------------------------

y_stage_test = np.load(DATA_DIR /"y_stage_test.npy")

y_true = np.zeros_like(y_stage_test)

y_true[y_stage_test == 1] = 1
y_true[(y_stage_test == 2) | (y_stage_test == 3)] = 2

# ----------------------------------------------------------
# 1. Confusion Matrix
# ----------------------------------------------------------

cm = confusion_matrix(y_true, final_pred)

plt.figure(figsize=(7,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Non-AKI","Stage1","Stage2+3"],
    yticklabels=["Non-AKI","Stage1","Stage2+3"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Final AKI Model Confusion Matrix")
plt.tight_layout()
plt.savefig(
    "outputs/final_model/confusion_matrix.png",
    dpi=300
)
plt.close()

# ----------------------------------------------------------
# 2. Normalized Confusion Matrix
# ----------------------------------------------------------

cm_pct = cm.astype(float) / cm.sum(axis=1)[:,None]

plt.figure(figsize=(7,6))
sns.heatmap(
    cm_pct,
    annot=True,
    fmt=".1%",
    cmap="Blues",
    xticklabels=["Non-AKI","Stage1","Stage2+3"],
    yticklabels=["Non-AKI","Stage1","Stage2+3"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Normalized Confusion Matrix")
plt.tight_layout()
plt.savefig(
    "outputs/final_model/confusion_matrix_percent.png",
    dpi=300
)
plt.close()

# ----------------------------------------------------------
# 3. ROC Curve (One-vs-Rest)
# ----------------------------------------------------------

y_bin = label_binarize(y_true, classes=[0,1,2])

plt.figure(figsize=(7,7))

class_names = [
    "Non-AKI",
    "Stage1",
    "Stage2+3"
]

for i in range(3):

    fpr, tpr, _ = roc_curve(
        y_bin[:,i],
        final_prob[:,i]
    )

    roc_auc = auc(fpr,tpr)

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{class_names[i]} (AUROC={roc_auc:.3f})"
    )

plt.plot([0,1],[0,1],"--",color="gray")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(
    "outputs/final_model/roc_curve_multiclass.png",
    dpi=300
)
plt.close()

# ----------------------------------------------------------
# 4. Precision Recall Curve
# ----------------------------------------------------------

plt.figure(figsize=(7,7))

for i in range(3):

    precision, recall, _ = precision_recall_curve(
        y_bin[:,i],
        final_prob[:,i]
    )

    ap = average_precision_score(
        y_bin[:,i],
        final_prob[:,i]
    )

    plt.plot(
        recall,
        precision,
        linewidth=2,
        label=f"{class_names[i]} (AUPRC={ap:.3f})"
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("One-vs-Rest Precision Recall Curve")
plt.legend()
plt.tight_layout()
plt.savefig(
    "outputs/final_model/pr_curve_multiclass.png",
    dpi=300
)
plt.close()

# ----------------------------------------------------------
# 5. Probability Distribution
# ----------------------------------------------------------

plt.figure(figsize=(10,6))

plt.hist(
    p_nonaki,
    bins=50,
    alpha=0.5,
    label="Non-AKI"
)

plt.hist(
    p_stage1,
    bins=50,
    alpha=0.5,
    label="Stage1"
)

plt.hist(
    p_stage23,
    bins=50,
    alpha=0.5,
    label="Stage2+3"
)

plt.xlabel("Predicted Probability")
plt.ylabel("Count")
plt.title("Probability Distribution")
plt.legend()

plt.tight_layout()

plt.savefig(
    "outputs/final_model/probability_distribution.png",
    dpi=300
)

plt.close()

# ----------------------------------------------------------
# 6. Detection Rate
# ----------------------------------------------------------

rates = []

for cls in [0,1,2]:

    mask = y_true == cls

    correct = (
        final_pred[mask] == cls
    ).sum()

    rate = correct / mask.sum()

    rates.append(rate)

plt.figure(figsize=(7,5))

bars = plt.bar(
    ["Non-AKI","Stage1","Stage2+3"],
    rates
)

plt.ylim(0,1.1)

for bar,val in zip(bars,rates):

    plt.text(
        bar.get_x()+bar.get_width()/2,
        val+0.02,
        f"{val:.3f}",
        ha="center"
    )

plt.ylabel("Detection Rate")
plt.title("Class-wise Detection Rate")

plt.tight_layout()

plt.savefig(
    "outputs/final_model/class_detection_rate.png",
    dpi=300
)

plt.close()

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------

summary_df = pd.DataFrame({
    "Class":[
        "Non-AKI",
        "Stage1",
        "Stage2+3"
    ],
    "DetectionRate":rates
})

summary_df.to_csv(
    "outputs/final_model/detection_rate.csv",
    index=False
)

print("="*70)
print("FINAL MODEL EVALUATION COMPLETE")
print("="*70)

print(summary_df)

print()
print("Saved:")
print(" - confusion_matrix.png")
print(" - confusion_matrix_percent.png")
print(" - roc_curve_multiclass.png")
print(" - pr_curve_multiclass.png")
print(" - probability_distribution.png")
print(" - class_detection_rate.png")

