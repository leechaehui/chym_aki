# ==========================================================
# CatBoost Stage Classification
# Stage1 vs Stage2 vs Stage3
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import os
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    classification_report,
    confusion_matrix
)

from catboost import CatBoostClassifier

import optuna

# ==========================================================
# 폴더 생성
# ==========================================================

os.makedirs("outputs", exist_ok=True)

# ==========================================================
# 데이터 로딩
# ==========================================================

X_train = np.load("../data/X_train.npy")
X_valid = np.load("../data/X_valid.npy")
X_test = np.load("../data/X_test.npy")

y_stage_train = np.load("../data/y_stage_train.npy")
y_stage_valid = np.load("../data/y_stage_valid.npy")
y_stage_test = np.load("../data/y_stage_test.npy")

# ==========================================================
# AKI 환자만 추출
# ==========================================================

train_mask = y_stage_train > 0
valid_mask = y_stage_valid > 0
test_mask = y_stage_test > 0

X_stage_train = X_train[train_mask]
X_stage_valid = X_valid[valid_mask]
X_stage_test = X_test[test_mask]

y_stage_train_mc = y_stage_train[train_mask]
y_stage_valid_mc = y_stage_valid[valid_mask]
y_stage_test_mc = y_stage_test[test_mask]

# ==========================================================
# 라벨 변환
# Stage1=0
# Stage2=1
# Stage3=2
# ==========================================================

y_stage_train_mc = y_stage_train_mc - 1
y_stage_valid_mc = y_stage_valid_mc - 1
y_stage_test_mc = y_stage_test_mc - 1

# ==========================================================
# Class Weight
# ==========================================================

classes = np.unique(y_stage_train_mc)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_stage_train_mc
)

class_weights = list(weights)

print("\nClass Weights")
print(class_weights)

# ==========================================================
# Optuna
# ==========================================================

def objective(trial):

    params = {

        "loss_function":"MultiClass",

        "iterations":
            trial.suggest_int(
                "iterations",
                300,
                1500
            ),

        "depth":
            trial.suggest_int(
                "depth",
                4,
                10
            ),

        "learning_rate":
            trial.suggest_float(
                "learning_rate",
                0.01,
                0.2,
                log=True
            ),

        "l2_leaf_reg":
            trial.suggest_float(
                "l2_leaf_reg",
                1,
                20
            ),

        "random_strength":
            trial.suggest_float(
                "random_strength",
                0,
                10
            ),

        "bagging_temperature":
            trial.suggest_float(
                "bagging_temperature",
                0,
                10
            ),

        "class_weights":
            class_weights,

        "random_seed":42,

        "verbose":0
    }

    model = CatBoostClassifier(**params)

    model.fit(
        X_stage_train,
        y_stage_train_mc
    )

    pred = model.predict(
        X_stage_valid
    )

    pred = pred.flatten()

    macro_f1 = f1_score(
        y_stage_valid_mc,
        pred,
        average="macro"
    )

    return macro_f1

print("=" * 70)
print("OPTUNA START")
print("=" * 70)

study = optuna.create_study(
    direction="maximize"
)

study.optimize(
    objective,
    n_trials=50
)

print("\nBest Params")
print(study.best_params)

print("\nBest Macro F1")
print(study.best_value)

# ==========================================================
# 최종 모델
# ==========================================================

best_params = study.best_params

cat = CatBoostClassifier(

    loss_function="MultiClass",

    class_weights=class_weights,

    random_seed=42,

    verbose=0,

    **best_params
)

cat.fit(
    X_stage_train,
    y_stage_train_mc
)
# ==========================================================
# Threshold Tuning
# ==========================================================

valid_proba = cat.predict_proba(
    X_stage_valid
)

best_threshold = 0.5
best_macro_f1 = 0

for threshold in np.arange(0.10, 0.91, 0.01):

    pred = []

    for p in valid_proba:

        # Stage2 확률이 threshold 이상이면
        # 강제로 Stage2

        if (
                p[1] >= threshold
                and p[1] > p[0]
        ):
            pred.append(1)

        else:
            pred.append(np.argmax(p))

    pred = np.array(pred)

    score = f1_score(
        y_stage_valid_mc,
        pred,
        average="macro"
    )

    if score > best_macro_f1:

        best_macro_f1 = score
        best_threshold = threshold

print()
print("="*70)
print("BEST THRESHOLD")
print("="*70)

print(best_threshold)
print(best_macro_f1)
# ==========================================================
# Validation
# ==========================================================

valid_proba = cat.predict_proba(
    X_stage_valid
)

valid_pred = []

for p in valid_proba:

    if p[1] >= best_threshold:
        valid_pred.append(1)

    else:
        valid_pred.append(np.argmax(p))

valid_pred = np.array(valid_pred)

valid_pred = valid_pred.flatten()

accuracy = accuracy_score(
    y_stage_valid_mc,
    valid_pred
)

weighted_f1 = f1_score(
    y_stage_valid_mc,
    valid_pred,
    average="weighted"
)

macro_f1 = f1_score(
    y_stage_valid_mc,
    valid_pred,
    average="macro"
)

stage1_recall = recall_score(
    y_stage_valid_mc,
    valid_pred,
    labels=[0],
    average=None
)[0]

stage2_recall = recall_score(
    y_stage_valid_mc,
    valid_pred,
    labels=[1],
    average=None
)[0]

stage3_recall = recall_score(
    y_stage_valid_mc,
    valid_pred,
    labels=[2],
    average=None
)[0]

print("\n" + "="*70)
print("VALIDATION RESULT")
print("="*70)

print(f"Accuracy      : {accuracy:.4f}")
print(f"Weighted F1   : {weighted_f1:.4f}")
print(f"Macro F1      : {macro_f1:.4f}")

print()
print(f"Stage1 Recall : {stage1_recall:.4f}")
print(f"Stage2 Recall : {stage2_recall:.4f}")
print(f"Stage3 Recall : {stage3_recall:.4f}")

# ==========================================================
# Classification Report
# ==========================================================

print()

print(classification_report(
    y_stage_valid_mc,
    valid_pred,
    target_names=[
        "Stage1",
        "Stage2",
        "Stage3"
    ]
))

# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(
    y_stage_valid_mc,
    valid_pred
)

plt.figure(figsize=(7,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=[
        'Stage1',
        'Stage2',
        'Stage3'
    ],
    yticklabels=[
        'Stage1',
        'Stage2',
        'Stage3'
    ]
)

plt.title(
    "CatBoost Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=300
)

plt.show()

# ==========================================================
# Feature Importance
# ==========================================================

FEATURE_NAMES = [

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

importance = pd.DataFrame({

    "Feature":FEATURE_NAMES,

    "Importance":
        cat.get_feature_importance()

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 20 Features")
print(
    importance.head(20)
)

importance.to_csv(
    "outputs/feature_importance.csv",
    index=False
)

# ==========================================================
# Test
# ==========================================================

test_proba = cat.predict_proba(
    X_stage_test
)

test_pred = []

for p in test_proba:

    if p[1] >= best_threshold:
        test_pred.append(1)

    else:
        test_pred.append(np.argmax(p))

test_pred = np.array(test_pred)

test_pred = test_pred.flatten()

test_accuracy = accuracy_score(
    y_stage_test_mc,
    test_pred
)

test_weighted_f1 = f1_score(
    y_stage_test_mc,
    test_pred,
    average="weighted"
)

test_macro_f1 = f1_score(
    y_stage_test_mc,
    test_pred,
    average="macro"
)

print("\n" + "="*70)
print("TEST RESULT")
print("="*70)

print(f"Accuracy      : {test_accuracy:.4f}")
print(f"Weighted F1   : {test_weighted_f1:.4f}")
print(f"Macro F1      : {test_macro_f1:.4f}")

# ==========================================================
# 저장
# ==========================================================

with open(
    "outputs/catboost_stage.pkl",
    "wb"
) as f:

    pickle.dump(
        cat,
        f
    )

print("\n저장 완료")

# ==========================================================
# 발표자료용 시각화
# ==========================================================

print("\n" + "="*70)
print("PRESENTATION FIGURES")
print("="*70)

# ==========================================================
# 1. Validation 성능 요약
# ==========================================================

metrics = pd.DataFrame({

    "Metric":[
        "Accuracy",
        "Weighted F1",
        "Macro F1"
    ],

    "Score":[
        accuracy,
        weighted_f1,
        macro_f1
    ]
})

plt.figure(figsize=(8,5))

bars = plt.bar(
    metrics["Metric"],
    metrics["Score"]
)

plt.ylim(0,1)

plt.ylabel("Score")

plt.title(
    "CatBoost Validation Performance"
)

for bar in bars:

    plt.text(
        bar.get_x()+bar.get_width()/2,
        bar.get_height()+0.01,
        f"{bar.get_height():.3f}",
        ha='center'
    )

plt.tight_layout()

plt.savefig(
    "outputs/catboost_performance.png",
    dpi=300
)

plt.show()


# ==========================================================
# 2. Stage Recall
# ==========================================================

recall_df = pd.DataFrame({

    "Stage":[
        "Stage1",
        "Stage2",
        "Stage3"
    ],

    "Recall":[
        stage1_recall,
        stage2_recall,
        stage3_recall
    ]
})

plt.figure(figsize=(8,5))

bars = plt.bar(
    recall_df["Stage"],
    recall_df["Recall"]
)

plt.ylim(0,1)

plt.ylabel("Recall")

plt.title(
    "Recall by AKI Stage"
)

for bar in bars:

    plt.text(
        bar.get_x()+bar.get_width()/2,
        bar.get_height()+0.01,
        f"{bar.get_height():.3f}",
        ha='center'
    )

plt.tight_layout()

plt.savefig(
    "outputs/stage_recall.png",
    dpi=300
)

plt.show()


# ==========================================================
# 3. Feature Importance Top15
# ==========================================================

top15 = (
    importance
    .head(15)
    .sort_values(
        "Importance",
        ascending=True
    )
)

plt.figure(figsize=(10,8))

plt.barh(
    top15["Feature"],
    top15["Importance"]
)

plt.xlabel("Importance")

plt.title(
    "Top 15 Feature Importance (CatBoost)"
)

plt.tight_layout()

plt.savefig(
    "outputs/feature_importance_top15.png",
    dpi=300
)

plt.show()




# ==========================================================
# 5. Stage Recall 비교 (발표 핵심)
# ==========================================================

plt.figure(figsize=(8,5))

bars = plt.bar(
    ["Stage1","Stage2","Stage3"],
    [
        stage1_recall,
        stage2_recall,
        stage3_recall
    ]
)

plt.ylabel("Recall")

plt.ylim(0,1)

plt.title(
    "CatBoost Recall by Stage"
)

for bar in bars:

    plt.text(
        bar.get_x()+bar.get_width()/2,
        bar.get_height()+0.01,
        f"{bar.get_height():.3f}",
        ha='center'
    )

plt.tight_layout()

plt.savefig(
    "outputs/catboost_stage_recall.png",
    dpi=300
)

plt.show()

print("\n발표용 Figure 저장 완료")
print("1. catboost_performance.png")
print("2. stage_recall.png")
print("3. feature_importance_top15.png")
print("4. model_comparison.png")
print("5. catboost_stage_recall.png")