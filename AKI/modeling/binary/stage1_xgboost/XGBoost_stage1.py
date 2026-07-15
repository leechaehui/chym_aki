# ==========================================================
# 07_stage1_xgboost.py
#
# Stage 1 : AKI Prediction
# Binary Classification
# 0 = Non-AKI
# 1 = AKI
#
# Model : XGBoost + Optuna
#
# Outputs
# - Optuna tuning result
# - Threshold optimization
# - Validation/Test metrics
# - ROC Curve
# - PR Curve
# - Confusion Matrix
# - Feature Importance
# - Probability Distribution
# - Trained Model
#
# Author : B
# ==========================================================

import os
import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve
)

# ==========================================================
# OUTPUT DIRECTORY
# ==========================================================

os.makedirs("outputs", exist_ok=True)

# ==========================================================
# FEATURE NAMES
# ==========================================================

FEATURE_COLS = [

    'map_mean',
    'map_min',
    'map_below65_hours',
    'sbp_min',
    'sbp_mean',
    'shock_index_mean',

    'hr_max',
    'hr_mean',
    'rr_max',
    'rr_mean',
    'temp_max',
    'temp_mean',

    'urine_output_sum',
    'urine_output_6h',
    'oliguria_flag',

    'creatinine_min',
    'creatinine_max',
    'creatinine_delta',

    'bun_max',
    'bun_cr_ratio',

    'lactate_max',
    'lactate_mean',

    'vasopressor_flag',
    'vasopressor_hours',
    'norepi_dose_max',

    'potassium_max',
    'potassium_mean',

    'bicarbonate_min',
    'bicarbonate_mean',

    'sodium_min',
    'sodium_max',

    'hemoglobin_min',
    'hemoglobin_mean',

    'spo2_min',
    'spo2_mean'
]


# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 70)
print("LOAD DATA")
print("=" * 70)

X_train = np.load("../data/X_train.npy")
X_valid = np.load("../data/X_valid.npy")
X_test = np.load("../data/X_test.npy")

y_train = np.load("../data/y_train.npy")
y_valid = np.load("../data/y_valid.npy")
y_test = np.load("../data/y_test.npy")

print(f"X_train : {X_train.shape}")
print(f"X_valid : {X_valid.shape}")
print(f"X_test  : {X_test.shape}")
print()

print("AKI Ratio")
print(f"Train : {y_train.mean():.4f}")
print(f"Valid : {y_valid.mean():.4f}")
print(f"Test  : {y_test.mean():.4f}")
print()

# ==========================================================
# FEATURE EXCLUSION
# ==========================================================

ORIGINAL_FEATURE_COLS = [

    'map_mean', 'map_min', 'map_below65_hours',
    'sbp_min', 'sbp_mean', 'shock_index_mean',

    'hr_max', 'hr_mean',
    'rr_max', 'rr_mean',
    'temp_max', 'temp_mean',

    'urine_output_sum',
    'urine_output_6h',
    'oliguria_flag',

    'creatinine_min',
    'creatinine_max',
    'creatinine_delta',

    'bun_max',
    'bun_cr_ratio',

    'lactate_max',
    'lactate_mean',

    'vasopressor_flag',
    'vasopressor_hours',
    'norepi_dose_max',

    'potassium_max',
    'potassium_mean',

    'bicarbonate_min',
    'bicarbonate_mean',

    'sodium_min',
    'sodium_max',

    'hemoglobin_min',
    'hemoglobin_mean',

    'spo2_min',
    'spo2_mean'
]

EXCLUDED_FEATURES = [

    'urine_output_sum',
    'urine_output_6h',
    'oliguria_flag',

    'creatinine_min',
    'creatinine_max',
    'creatinine_delta'

]

selected_idx = [

    i
    for i, col in enumerate(ORIGINAL_FEATURE_COLS)
    if col not in EXCLUDED_FEATURES

]

X_train = X_train[:, selected_idx]
X_valid = X_valid[:, selected_idx]
X_test = X_test[:, selected_idx]

FEATURE_COLS = [

    col
    for col in ORIGINAL_FEATURE_COLS
    if col not in EXCLUDED_FEATURES

]
print()
print("=" * 70)
print("FEATURE EXCLUSION")
print("=" * 70)

print("Removed Features")

for col in EXCLUDED_FEATURES:
    print(f" - {col}")

print()
print(f"Final Feature Count : {X_train.shape[1]}")
print()

print("Selected Features")

for col in FEATURE_COLS:
    print(col)
# ==========================================================
# CLASS IMBALANCE
# ==========================================================

n_pos = y_train.sum()
n_neg = len(y_train) - n_pos

scale_pos_weight = n_neg / n_pos

print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print(f"AKI      : {n_pos:,}")
print(f"Non-AKI  : {n_neg:,}")
print(f"scale_pos_weight : {scale_pos_weight:.3f}")
print()

# ==========================================================
# OPTUNA OBJECTIVE
# ==========================================================

trial_history = []

def objective(trial):

    params = {

        "objective": "binary:logistic",

        "n_estimators":
            trial.suggest_int(
                "n_estimators",
                300,
                1000
            ),

        "max_depth":
            trial.suggest_int(
                "max_depth",
                3,
                8
            ),

        "learning_rate":
            trial.suggest_float(
                "learning_rate",
                0.01,
                0.2,
                log=True
            ),

        "subsample":
            trial.suggest_float(
                "subsample",
                0.6,
                1.0
            ),

        "colsample_bytree":
            trial.suggest_float(
                "colsample_bytree",
                0.6,
                1.0
            ),

        "min_child_weight":
            trial.suggest_int(
                "min_child_weight",
                1,
                10
            ),

        "gamma":
            trial.suggest_float(
                "gamma",
                0,
                5
            ),

        "scale_pos_weight":
            scale_pos_weight,

        "eval_metric":
            "aucpr",

        "random_state":
            42,

        "n_jobs":
            -1
    }

    model = XGBClassifier(
        **params,
        early_stopping_rounds=50
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )

    prob = model.predict_proba(X_valid)[:, 1]

    score = average_precision_score(
        y_valid,
        prob
    )

    trial_history.append([
        trial.number,
        score
    ])

    print(
        f"Trial {trial.number:03d} | "
        f"AUPRC = {score:.5f}"
    )

    return score

# ==========================================================
# OPTUNA TUNING
# ==========================================================

print("=" * 70)
print("OPTUNA TUNING")
print("=" * 70)

study = optuna.create_study(
    direction="maximize"
)

study.optimize(
    objective,
    n_trials=30,
    show_progress_bar=True
)

trial_df = pd.DataFrame(
    trial_history,
    columns=["trial", "auprc"]
)

trial_df.to_csv(
    "outputs/optuna_history.csv",
    index=False
)

print()
print("=" * 70)
print("BEST RESULT")
print("=" * 70)

print(f"Best AUPRC : {study.best_value:.5f}")
print()

for k, v in study.best_params.items():
    print(f"{k:<20} : {v}")

# ==========================================================
# FINAL MODEL
# ==========================================================

best_params = study.best_params

model = XGBClassifier(
    **best_params,
    objective="binary:logistic",
    scale_pos_weight=scale_pos_weight,
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=False
)


# ==========================================================
# VALIDATION PROBABILITY
# ==========================================================

valid_prob = model.predict_proba(X_valid)[:, 1]

# ==========================================================
# THRESHOLD SEARCH
# ==========================================================

threshold_results = []

best_threshold = 0.5
best_specificity = 0

for threshold in np.arange(0.10, 0.91, 0.01):

    pred = (
        valid_prob >= threshold
    ).astype(int)

    precision = precision_score(
        y_valid,
        pred,
        zero_division=0
    )

    recall = recall_score(
        y_valid,
        pred,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_valid,
        pred
    ).ravel()

    specificity = tn / (tn + fp)

    threshold_results.append([
        threshold,
        precision,
        recall,
        specificity
    ])


    if recall >= 0.75:
        if specificity > best_specificity:
            best_specificity = specificity
            best_threshold = threshold

threshold_df = pd.DataFrame(
    threshold_results,
    columns=[
        "threshold",
        "precision",
        "recall",
        "specificity"
    ]
)

threshold_df.to_csv(
    "outputs/threshold_search.csv",
    index=False
)

print()
print("=" * 70)
print("THRESHOLD OPTIMIZATION")
print("=" * 70)

print(f"Best Threshold : {best_threshold:.2f}")
print(f"Specificity    : {best_specificity:.4f}")

# ==========================================================
# VALIDATION EVALUATION
# ==========================================================

valid_pred = (
    valid_prob >= best_threshold
).astype(int)

val_auroc = roc_auc_score(
    y_valid,
    valid_prob
)

val_auprc = average_precision_score(
    y_valid,
    valid_prob
)

val_precision = precision_score(
    y_valid,
    valid_pred
)

val_recall = recall_score(
    y_valid,
    valid_pred
)

val_f1 = f1_score(
    y_valid,
    valid_pred
)

print()
print("=" * 70)
print("VALIDATION RESULT")
print("=" * 70)

print(f"AUROC     : {val_auroc:.4f}")
print(f"AUPRC     : {val_auprc:.4f}")
print(f"Precision : {val_precision:.4f}")
print(f"Recall    : {val_recall:.4f}")
print(f"F1-score  : {val_f1:.4f}")

train_prob = model.predict_proba(X_train)[:,1]

train_auroc = roc_auc_score(
    y_train,
    train_prob
)

train_auprc = average_precision_score(
    y_train,
    train_prob
)

tn, fp, fn, tp = confusion_matrix(
    y_valid,
    valid_pred
).ravel()

val_sensitivity = tp / (tp + fn)
val_specificity = tn / (tn + fp)

print(f"Sensitivity : {val_sensitivity:.4f}")
print(f"Specificity : {val_specificity:.4f}")

print()
print("=" * 70)
print("TRAIN RESULT")
print("=" * 70)

print(f"AUROC : {train_auroc:.4f}")
print(f"AUPRC : {train_auprc:.4f}")

print()
print("Train vs Validation")

print(f"Train AUROC      : {train_auroc:.4f}")
print(f"Validation AUROC : {val_auroc:.4f}")

print(f"Train AUPRC      : {train_auprc:.4f}")
print(f"Validation AUPRC : {val_auprc:.4f}")

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    y_valid,
    valid_pred
)

print()
print("Confusion Matrix")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title(
    "Validation Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "outputs/validation_confusion_matrix.png",
    dpi=300
)

plt.close()

# ==========================================================
# ROC CURVE
# ==========================================================

fpr, tpr, _ = roc_curve(
    y_valid,
    valid_prob
)

plt.figure(figsize=(6,6))

plt.plot(
    fpr,
    tpr,
    label=f"AUROC={val_auroc:.4f}"
)

plt.plot(
    [0,1],
    [0,1],
    "--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.tight_layout()

plt.savefig(
    "outputs/roc_curve.png",
    dpi=300
)

plt.close()

# ==========================================================
# PR CURVE
# ==========================================================

precisions, recalls, _ = precision_recall_curve(
    y_valid,
    valid_prob
)

plt.figure(figsize=(6,6))

plt.plot(
    recalls,
    precisions
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")

plt.tight_layout()

plt.savefig(
    "outputs/pr_curve.png",
    dpi=300
)

plt.close()

# ==========================================================
# PROBABILITY DISTRIBUTION
# ==========================================================

plt.figure(figsize=(8,5))

plt.hist(
    valid_prob[y_valid == 0],
    bins=50,
    alpha=0.5,
    label="Non-AKI"
)

plt.hist(
    valid_prob[y_valid == 1],
    bins=50,
    alpha=0.5,
    label="AKI"
)

plt.legend()

plt.title(
    "Predicted Probability Distribution"
)

plt.tight_layout()

plt.savefig(
    "outputs/probability_distribution.png",
    dpi=300
)

plt.close()

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

print()
print("=" * 70)
print("FEATURE CHECK")
print("=" * 70)

print("X_train columns     :", X_train.shape[1])
print("FEATURE_COLS length :", len(FEATURE_COLS))
print("Importance length   :", len(model.feature_importances_))
print()


importance_df = pd.DataFrame({

    "feature":
        FEATURE_COLS,

    "importance":
        model.feature_importances_

})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

importance_df.to_csv(
    "outputs/feature_importance.csv",
    index=False
)

print()
print("=" * 70)
print("TOP 15 FEATURES")
print("=" * 70)

print(
    importance_df.head(15)
)

plt.figure(
    figsize=(10,8)
)

top15 = importance_df.head(15)

plt.barh(
    top15["feature"],
    top15["importance"]
)

plt.gca().invert_yaxis()

plt.title(
    "Top 15 Feature Importance"
)

plt.tight_layout()

plt.savefig(
    "outputs/feature_importance.png",
    dpi=300
)

plt.close()

# ==========================================================
# TEST EVALUATION
# ==========================================================

test_prob = model.predict_proba(X_test)[:,1]

test_pred = (
    test_prob >= best_threshold
).astype(int)

test_auroc = roc_auc_score(
    y_test,
    test_prob
)

test_auprc = average_precision_score(
    y_test,
    test_prob
)

test_precision = precision_score(
    y_test,
    test_pred
)

test_recall = recall_score(
    y_test,
    test_pred
)

test_f1 = f1_score(
    y_test,
    test_pred
)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_pred
).ravel()

test_sensitivity = tp / (tp + fn)
test_specificity = tn / (tn + fp)

print(f"Sensitivity : {test_sensitivity:.4f}")
print(f"Specificity : {test_specificity:.4f}")

print()
print("=" * 70)
print("TEST RESULT")
print("=" * 70)

print(f"AUROC     : {test_auroc:.4f}")
print(f"AUPRC     : {test_auprc:.4f}")
print(f"Precision : {test_precision:.4f}")
print(f"Recall    : {test_recall:.4f}")
print(f"F1-score  : {test_f1:.4f}")

# ==========================================================
# ADDITIONAL VISUALIZATION FOR PRESENTATION
# ==========================================================

print()
print("=" * 70)
print("GENERATE PRESENTATION FIGURES")
print("=" * 70)

# ----------------------------------------------------------
# 1. Performance Summary
# ----------------------------------------------------------

metrics = {
    "AUROC": val_auroc,
    "AUPRC": val_auprc,
    "Precision": val_precision,
    "Recall": val_recall,
    "F1": val_f1
}

plt.figure(figsize=(8, 5))

plt.bar(
    metrics.keys(),
    metrics.values()
)

plt.ylim(0, 1)

plt.title(
    "Validation Performance Summary"
)

plt.ylabel("Score")

plt.tight_layout()

plt.savefig(
    "outputs/performance_summary.png",
    dpi=300
)

plt.close()

# ----------------------------------------------------------
# 2. Threshold Optimization
# ----------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    threshold_df["threshold"],
    threshold_df["recall"],
    label="Recall"
)

plt.plot(
    threshold_df["threshold"],
    threshold_df["specificity"],
    label="Specificity"
)

plt.axvline(
    best_threshold,
    linestyle="--",
    label=f"Best={best_threshold:.2f}"
)

plt.xlabel("Threshold")
plt.ylabel("Score")

plt.title(
    "Threshold Optimization"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "outputs/threshold_optimization.png",
    dpi=300
)

plt.close()

# ----------------------------------------------------------
# 3. Train vs Validation AUROC
# ----------------------------------------------------------

plt.figure(figsize=(6, 5))

plt.bar(
    ["Train", "Validation"],
    [train_auroc, val_auroc]
)

plt.ylim(0.90, 1.00)

plt.ylabel("AUROC")

plt.title(
    "Train vs Validation AUROC"
)

plt.tight_layout()

plt.savefig(
    "outputs/train_vs_validation_auroc.png",
    dpi=300
)

plt.close()

# ----------------------------------------------------------
# 4. Normalized Confusion Matrix
# ----------------------------------------------------------

cm_percent = (
    cm.astype(float)
    / cm.sum(axis=1)[:, np.newaxis]
)

plt.figure(figsize=(6, 5))

plt.imshow(
    cm_percent,
    interpolation="nearest"
)

plt.colorbar()

for i in range(2):
    for j in range(2):

        plt.text(
            j,
            i,
            f"{cm_percent[i, j]:.1%}",
            ha="center",
            va="center"
        )

plt.xticks(
    [0, 1],
    ["Non-AKI", "AKI"]
)

plt.yticks(
    [0, 1],
    ["Non-AKI", "AKI"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title(
    "Normalized Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix_percent.png",
    dpi=300
)

plt.close()

# ----------------------------------------------------------
# 5. Top 10 Feature Importance
# ----------------------------------------------------------

top10 = importance_df.head(10)

plt.figure(figsize=(8, 6))

plt.barh(
    top10["feature"],
    top10["importance"]
)

plt.gca().invert_yaxis()

plt.title(
    "Top 10 Important Features"
)

plt.xlabel("Importance")

plt.tight_layout()

plt.savefig(
    "outputs/top10_importance.png",
    dpi=300
)

plt.close()

print("Saved:")
print(" - performance_summary.png")
print(" - threshold_optimization.png")
print(" - train_vs_validation_auroc.png")
print(" - confusion_matrix_percent.png")
print(" - top10_importance.png")
# ==========================================================
# SAVE MODEL
# ==========================================================

with open(
    "outputs/stage1_xgboost.pkl",
    "wb"
) as f:

    pickle.dump(
        model,
        f
    )

with open(
    "outputs/best_threshold.txt",
    "w"
) as f:

    f.write(
        str(best_threshold)
    )

print()
print("=" * 70)
print("MODEL SAVED")
print("=" * 70)

print("stage1_xgboost.pkl")
print("best_threshold.txt")

print()
print("FINISHED")

