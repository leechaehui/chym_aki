# ==========================================================
# 07_modeling_stage_xgb.py
# AKI Stage Classification
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

from xgboost import XGBClassifier
import optuna

# ==========================================================
# 저장 폴더 생성
# ==========================================================

os.makedirs("outputs", exist_ok=True)

# ==========================================================
# 데이터 로딩
# ==========================================================

print("=" * 70)
print("데이터 로딩")
print("=" * 70)

X_train = np.load("../data/X_train.npy")
X_valid = np.load("../data/X_valid.npy")
X_test = np.load("../data/X_test.npy")

y_stage_train = np.load("../data/y_stage_train.npy")
y_stage_valid = np.load("../data/y_stage_valid.npy")
y_stage_test = np.load("../data/y_stage_test.npy")

print("원본 데이터")
print(f"X_train : {X_train.shape}")
print(f"X_valid : {X_valid.shape}")
print(f"X_test  : {X_test.shape}")
print()

# ==========================================================
# AKI 환자만 추출
# ==========================================================

print("=" * 70)
print("AKI 환자만 추출")
print("=" * 70)

train_mask = y_stage_train > 0
valid_mask = y_stage_valid > 0
test_mask = y_stage_test > 0

X_stage_train = X_train[train_mask]
X_stage_valid = X_valid[valid_mask]
X_stage_test = X_test[test_mask]

y_stage_train_mc = y_stage_train[train_mask]
y_stage_valid_mc = y_stage_valid[valid_mask]
y_stage_test_mc = y_stage_test[test_mask]

print("Stage 분포")

print("\nTrain")
print(pd.Series(y_stage_train_mc).value_counts().sort_index())

print("\nValid")
print(pd.Series(y_stage_valid_mc).value_counts().sort_index())

print("\nTest")
print(pd.Series(y_stage_test_mc).value_counts().sort_index())

# ==========================================================
# Stage 분포 시각화
# ==========================================================

stage_counts = pd.Series(
    y_stage_train_mc
).value_counts().sort_index()

stage_names = ["Stage1", "Stage2", "Stage3"]

plt.figure(figsize=(7,5))

sns.barplot(
    x=stage_names,
    y=stage_counts.values
)

plt.title("AKI Stage Distribution")

for i, v in enumerate(stage_counts.values):
    plt.text(
        i,
        v,
        str(v),
        ha='center'
    )

plt.tight_layout()

plt.savefig(
    "outputs/stage_distribution.png",
    dpi=300
)

plt.show()

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

class_weight_dict = dict(
    zip(classes, weights)
)
class_weight_dict = {
    0: 0.43,
    1: 4.0,
    2: 3.5
}

sample_weight = np.array(
    [
        class_weight_dict[y]
        for y in y_stage_train_mc
    ]
)



print()
print("=" * 70)
print("Class Mapping")
print("=" * 70)
print("0 -> Stage1")
print("1 -> Stage2")
print("2 -> Stage3")



# ==========================================================
# Optuna
# ==========================================================

print()
print("=" * 70)
print("Optuna Tuning Start")
print("=" * 70)

def objective(trial):

    params = {

        "objective":"multi:softprob",
        "num_class":3,

        "n_estimators":
            trial.suggest_int(
                "n_estimators",
                200,
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

        "random_state":42,
        "eval_metric":"mlogloss"
    }

    model = XGBClassifier(**params)

    model.fit(
        X_stage_train,
        y_stage_train_mc,
        sample_weight=sample_weight,
        verbose=False
    )

    pred = model.predict(
        X_stage_valid
    )

    macro_f1 = f1_score(
        y_stage_valid_mc,
        pred,
        average="macro"
    )

    return macro_f1

study = optuna.create_study(
    direction="maximize"
)

study.optimize(
    objective,
    n_trials=50
)

print()
print("=" * 70)
print("BEST PARAMS")
print("=" * 70)

print(study.best_params)

print()
print("BEST MACRO F1")
print(study.best_value)

# ==========================================================
# Optuna History
# ==========================================================

history = study.trials_dataframe()

plt.figure(figsize=(8,5))

plt.plot(
    history["number"],
    history["value"]
)

plt.title(
    "Optuna Optimization History"
)

plt.xlabel("Trial")
plt.ylabel("Macro F1")

plt.grid()

plt.tight_layout()

plt.savefig(
    "outputs/optuna_history.png",
    dpi=300
)

plt.show()

# ==========================================================
# 최종 모델
# ==========================================================

print()
print("=" * 70)
print("Final Model Training")
print("=" * 70)

best_params = study.best_params

xgb = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    random_state=42,
    eval_metric='mlogloss',
    **best_params
)

xgb.fit(
    X_stage_train,
    y_stage_train_mc,
    sample_weight=sample_weight
)

# ==========================================================
# Validation 평가
# ==========================================================

valid_pred = xgb.predict(
    X_stage_valid
)

accuracy = accuracy_score(
    y_stage_valid_mc,
    valid_pred
)

weighted_f1 = f1_score(
    y_stage_valid_mc,
    valid_pred,
    average='weighted'
)

macro_f1 = f1_score(
    y_stage_valid_mc,
    valid_pred,
    average='macro'
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

print()
print("=" * 70)
print("VALIDATION RESULT")
print("=" * 70)

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
    "Validation Confusion Matrix"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

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
        xgb.feature_importances_

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 20 Features")
print(
    importance.head(20)
)

top20 = (
    importance
    .head(20)
    .sort_values(
        "Importance",
        ascending=True
    )
)

plt.figure(figsize=(10,8))

plt.barh(
    top20["Feature"],
    top20["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.title(
    "Top 20 Feature Importance"
)

plt.tight_layout()

plt.savefig(
    "outputs/feature_importance_top20.png",
    dpi=300
)

plt.show()

importance.to_csv(
    "outputs/feature_importance.csv",
    index=False
)

# ==========================================================
# Test 평가
# ==========================================================

test_pred = xgb.predict(
    X_stage_test
)

test_accuracy = accuracy_score(
    y_stage_test_mc,
    test_pred
)

test_weighted_f1 = f1_score(
    y_stage_test_mc,
    test_pred,
    average='weighted'
)

test_macro_f1 = f1_score(
    y_stage_test_mc,
    test_pred,
    average='macro'
)

print()
print("=" * 70)
print("TEST RESULT")
print("=" * 70)

print(f"Accuracy      : {test_accuracy:.4f}")
print(f"Weighted F1   : {test_weighted_f1:.4f}")
print(f"Macro F1      : {test_macro_f1:.4f}")

# ==========================================================
# 결과 CSV
# ==========================================================

results = pd.DataFrame({

    "Metric":[
        "Accuracy",
        "Weighted F1",
        "Macro F1",
        "Stage1 Recall",
        "Stage2 Recall",
        "Stage3 Recall"
    ],

    "Score":[
        accuracy,
        weighted_f1,
        macro_f1,
        stage1_recall,
        stage2_recall,
        stage3_recall
    ]
})

results.to_csv(
    "outputs/stage_model_results.csv",
    index=False
)

# ==========================================================
# 모델 저장
# ==========================================================

with open(
    "outputs/xgb_stage_classifier.pkl",
    "wb"
) as f:
    pickle.dump(
        xgb,
        f
    )

print()
print("=" * 70)
print("저장 완료")
print("=" * 70)

print("outputs/")
print(" ├─ stage_distribution.png")
print(" ├─ confusion_matrix.png")
print(" ├─ feature_importance_top20.png")
print(" ├─ feature_importance.csv")
print(" ├─ optuna_history.png")
print(" ├─ stage_model_results.csv")
print(" └─ xgb_stage_classifier.pkl")