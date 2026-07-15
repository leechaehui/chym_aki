# ==========================================================
# 09_modeling_LGBM_v14_balanced_weight.py
# Stage1 vs Stage2+3 | Full 35개
# Method: 가중치 재조정 (Stage1 가중치 상향)
#         여러 가중치 조합 탐색
# ==========================================================

import os, warnings, pickle
warnings.filterwarnings("ignore")
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
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, roc_curve
)

OUT_DIR = "outputs/LGBM_v14_balanced_weight"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

FEATURE_COLS = [
    'map_mean','map_min','map_below65_hours',
    'sbp_min','sbp_mean','shock_index_mean',
    'hr_max','hr_mean','rr_max','rr_mean',
    'temp_max','temp_mean',
    'urine_output_sum','urine_output_6h','oliguria_flag',
    'creatinine_min','creatinine_max','creatinine_delta',
    'bun_max','bun_cr_ratio',
    'lactate_max','lactate_mean',
    'vasopressor_flag','vasopressor_hours','norepi_dose_max',
    'potassium_max','potassium_mean',
    'bicarbonate_min','bicarbonate_mean',
    'sodium_min','sodium_max',
    'hemoglobin_min','hemoglobin_mean',
    'spo2_min','spo2_mean'
]

print("=" * 70)
print("LGBM v14 | Balanced Weight | Stage1 vs Stage2+3")
print("=" * 70)

X_train_orig  = np.load("data/X_train.npy")
X_valid_orig  = np.load("data/X_valid.npy")
X_test_orig   = np.load("data/X_test.npy")
y_stage_train = np.load("data/y_stage_train.npy")
y_stage_valid = np.load("data/y_stage_valid.npy")
y_stage_test  = np.load("data/y_stage_test.npy")

train_mask = y_stage_train > 0
valid_mask = y_stage_valid > 0
test_mask  = y_stage_test  > 0

X_tr = X_train_orig[train_mask]
X_vl = X_valid_orig[valid_mask]
X_te = X_test_orig[test_mask]
y_stage_vl = y_stage_valid[valid_mask]

y_tr = (y_stage_train[train_mask] >= 2).astype(int)
y_vl = (y_stage_valid[valid_mask] >= 2).astype(int)
y_te = (y_stage_test[test_mask]   >= 2).astype(int)

# ==========================================================
# 가중치 조합 사전 탐색
# ==========================================================

print("=" * 70)
print("가중치 조합 사전 탐색 (기본 파라미터)")
print("=" * 70)

WEIGHT_CONFIGS = [
    {"name": "기존 (1:3.5)",    "w0": 0.64, "w1": 2.23},
    {"name": "균형 A (1:2.0)",  "w0": 1.00, "w1": 2.00},
    {"name": "균형 B (1:1.5)",  "w0": 1.00, "w1": 1.50},
    {"name": "균형 C (1:1.2)",  "w0": 1.00, "w1": 1.20},
    {"name": "동등 (1:1.0)",    "w0": 1.00, "w1": 1.00},
    {"name": "S1 중시 (1.5:2)", "w0": 1.50, "w1": 2.00},
    {"name": "S1 중시 (2:2)",   "w0": 2.00, "w1": 2.00},
]

base_params = {
    "objective":"binary","metric":"binary_logloss",
    "n_estimators":200,"max_depth":5,
    "learning_rate":0.05,"num_leaves":31,
    "random_state":42,"n_jobs":-1,"verbose":-1,
}

print(f"{'가중치 조합':>20} | {'S1 Recall':>10} | {'S2+3 Recall':>12} | {'F1':>8}")
print("-" * 60)

weight_results = []
for cfg in WEIGHT_CONFIGS:
    sw = np.array([cfg["w0"] if y==0 else cfg["w1"] for y in y_tr])
    m  = LGBMClassifier(**base_params)
    m.fit(X_tr, y_tr, sample_weight=sw)
    pred = m.predict(X_vl)
    tn, fp, fn, tp_ = confusion_matrix(y_vl, pred).ravel()
    s1r  = tn/(tn+fp) if (tn+fp)>0 else 0
    s23r = tp_/(tp_+fn) if (tp_+fn)>0 else 0
    f1   = f1_score(y_vl, pred, zero_division=0)
    weight_results.append([cfg["name"], cfg["w0"], cfg["w1"], s1r, s23r, f1])
    print(f"{cfg['name']:>20} | {s1r:>10.4f} | {s23r:>12.4f} | {f1:>8.4f}")

weight_df = pd.DataFrame(weight_results,
    columns=["name","w0","w1","s1r","s23r","f1"])
weight_df.to_csv(f"{OUT_DIR}/weight_search.csv", index=False)

# 최적 가중치 선정: S2+3 ≥ 0.75 + S1 최대
best_cfg = weight_df[weight_df["s23r"] >= 0.75]
if len(best_cfg) > 0:
    best_row = best_cfg.loc[best_cfg["s1r"].idxmax()]
else:
    best_row = weight_df.loc[weight_df["s1r"].idxmax()]

best_w0 = best_row["w0"]
best_w1 = best_row["w1"]
print(f"\n최적 가중치: Stage1={best_w0}, Stage2+3={best_w1}")
print(f"예상 S1={best_row['s1r']:.4f}, S2+3={best_row['s23r']:.4f}")
print()

# ==========================================================
# 최적 가중치로 Optuna 튜닝
# ==========================================================

sw_best = np.array([best_w0 if y==0 else best_w1 for y in y_tr])

trial_history = []
def objective(trial):
    params = {
        "objective":"binary","metric":"binary_logloss",
        "n_estimators":trial.suggest_int("n_estimators",100,500),
        "max_depth":trial.suggest_int("max_depth",3,8),
        "learning_rate":trial.suggest_float("learning_rate",0.01,0.2,log=True),
        "num_leaves":trial.suggest_int("num_leaves",20,60),
        "min_child_samples":trial.suggest_int("min_child_samples",5,30),
        "subsample":trial.suggest_float("subsample",0.6,1.0),
        "colsample_bytree":trial.suggest_float("colsample_bytree",0.6,1.0),
        "reg_alpha":trial.suggest_float("reg_alpha",0,1.0),
        "reg_lambda":trial.suggest_float("reg_lambda",0,1.0),
        "random_state":42,"n_jobs":-1,"verbose":-1,
    }
    m = LGBMClassifier(**params)
    m.fit(X_tr, y_tr, sample_weight=sw_best)
    pred = m.predict(X_vl)
    tn,fp,fn,tp_ = confusion_matrix(y_vl, pred).ravel()
    s1r  = tn/(tn+fp) if (tn+fp)>0 else 0
    s23r = tp_/(tp_+fn) if (tp_+fn)>0 else 0
    # S2+3 ≥ 0.75 보장하면서 S1 최대화
    if s23r < 0.75:
        return 0.0
    score = s1r * 0.5 + s23r * 0.5
    trial_history.append([trial.number, s1r, s23r, score])
    print(f"Trial {trial.number:03d} | S1={s1r:.4f} | S2+3={s23r:.4f} | Score={score:.4f}")
    return score

print("=" * 70)
print(f"OPTUNA (50 trials, 가중치 Stage1={best_w0}, Stage2+3={best_w1})")
print("=" * 70)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

trial_df = pd.DataFrame(trial_history,
    columns=["trial","s1r","s23r","score"])
trial_df.to_csv(f"{OUT_DIR}/optuna_history.csv", index=False)
print(f"\nBest Score: {study.best_value:.5f}")

bp = study.best_params
bp.update({"objective":"binary","metric":"binary_logloss",
           "random_state":42,"n_jobs":-1,"verbose":-1})
model = LGBMClassifier(**bp)
model.fit(X_tr, y_tr, sample_weight=sw_best)

valid_proba = model.predict_proba(X_vl)[:,1]
test_proba  = model.predict_proba(X_te)[:,1]

# Threshold 탐색 (S1 ≥ 0.50 + S2+3 최대)
best_t, best_score = 0.5, 0
for t in np.arange(0.10, 0.91, 0.01):
    pred = (valid_proba >= t).astype(int)
    tn,fp,fn,tp_ = confusion_matrix(y_vl, pred).ravel()
    s1r  = tn/(tn+fp) if (tn+fp)>0 else 0
    s23r = tp_/(tp_+fn) if (tp_+fn)>0 else 0
    score = s1r * 0.5 + s23r * 0.5
    if s23r >= 0.75 and score > best_score:
        best_score = score; best_t = t

valid_pred = (valid_proba >= best_t).astype(int)
test_pred  = (test_proba  >= best_t).astype(int)

tn,fp,fn,tp_ = confusion_matrix(y_vl, valid_pred).ravel()
val_s1r  = tn/(tn+fp); val_s23r = tp_/(tp_+fn)
val_auroc= roc_auc_score(y_vl, valid_proba)
val_auprc= average_precision_score(y_vl, valid_proba)
val_f1   = f1_score(y_vl, valid_pred, zero_division=0)

tn_t,fp_t,fn_t,tp_t = confusion_matrix(y_te, test_pred).ravel()
test_s1r  = tn_t/(tn_t+fp_t); test_s23r = tp_t/(tp_t+fn_t)
test_auroc= roc_auc_score(y_te, test_proba)
test_f1   = f1_score(y_te, test_pred, zero_division=0)

print(f"\nBest Threshold : {best_t:.2f}")
print("=" * 70)
print("VALIDATION RESULT")
print("=" * 70)
print(f"AUROC={val_auroc:.4f} | AUPRC={val_auprc:.4f} | F1={val_f1:.4f}")
print(f"Stage1 Recall  : {val_s1r:.4f}  ← 목표 0.50")
print(f"Stage2+3 Recall: {val_s23r:.4f}  ← 목표 0.75")
print()
print(classification_report(y_vl, valid_pred,
      target_names=["Stage1","Stage2+3"]))
print("Stage별 세부:")
for sv,sn in [(1,"Stage1"),(2,"Stage2"),(3,"Stage3")]:
    mask = y_stage_vl == sv
    if mask.sum() > 0:
        c = (valid_pred[mask]==0).sum() if sv==1 else (valid_pred[mask]==1).sum()
        print(f"  {sn}: {c}/{mask.sum()} ({c/mask.sum()*100:.1f}%)")
print()
print("TEST RESULT")
print(f"AUROC={test_auroc:.4f} | S1={test_s1r:.4f} | S2+3={test_s23r:.4f}")

cm = confusion_matrix(y_vl, valid_pred)
print(f"\nConfusion Matrix:\n{cm}")

# Figures
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Stage1","Stage2+3"],
            yticklabels=["Stage1","Stage2+3"])
plt.title(f"Confusion Matrix (v14 balanced_weight t={best_t})")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/confusion_matrix.png",dpi=300); plt.close()

cm_pct = cm.astype(float)/cm.sum(axis=1)[:,np.newaxis]
plt.figure(figsize=(6,5))
sns.heatmap(cm_pct, annot=True, fmt='.1%', cmap='Blues',
            xticklabels=["Stage1","Stage2+3"],
            yticklabels=["Stage1","Stage2+3"])
plt.title("Normalized CM (v14 balanced_weight)")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/confusion_matrix_percent.png",dpi=300); plt.close()

fpr,tpr,_=roc_curve(y_vl, valid_proba)
plt.figure(figsize=(6,6))
plt.plot(fpr,tpr,color="#185FA5",label=f"AUROC={val_auroc:.4f}")
plt.plot([0,1],[0,1],"--",color="gray")
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC Curve (v14 balanced_weight)")
plt.legend(); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/roc_curve.png",dpi=300); plt.close()

# 가중치 조합 비교 그래프
plt.figure(figsize=(12,6))
x = np.arange(len(weight_df))
w = 0.35
plt.bar(x-w/2, weight_df["s1r"],  w, label="Stage1 Recall",   color="#185FA5")
plt.bar(x+w/2, weight_df["s23r"], w, label="Stage2+3 Recall", color="#E24B4A")
plt.xticks(x, weight_df["name"], rotation=15, ha='right')
plt.ylim(0,1.1); plt.ylabel("Recall")
plt.axhline(0.50, color="#185FA5", linestyle="--", alpha=0.5, label="S1 목표")
plt.axhline(0.75, color="#E24B4A", linestyle="--", alpha=0.5, label="S2+3 목표")
plt.legend(); plt.title("가중치 조합별 Recall 비교")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/weight_comparison.png",dpi=300); plt.close()

importance_df = pd.DataFrame({"feature":FEATURE_COLS,
    "importance":model.feature_importances_}).sort_values("importance",ascending=False)
importance_df.to_csv(f"{OUT_DIR}/feature_importance.csv",index=False)
top15 = importance_df.head(15)
plt.figure(figsize=(10,8))
plt.barh(top15["feature"],top15["importance"],color="#185FA5")
plt.gca().invert_yaxis(); plt.title("Top 15 (v14 balanced_weight)")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/feature_importance.png",dpi=300); plt.close()

print(f"\nSaved → {OUT_DIR}")
pd.DataFrame({"Metric":["AUROC","AUPRC","F1","S1 Recall","S2+3 Recall"],
    "Valid":[val_auroc,val_auprc,val_f1,val_s1r,val_s23r],
    "Test":[test_auroc,roc_auc_score(y_te,test_proba),test_f1,test_s1r,test_s23r],
}).to_csv(f"{OUT_DIR}/results.csv",index=False)

with open("models/stage2_LGBM_v14_balanced_weight.pkl","wb") as f:
    pickle.dump({"model":model,"feature_cols":FEATURE_COLS,
                 "best_threshold":best_t,"best_w0":best_w0,"best_w1":best_w1,
                 "version":"v14_balanced_weight",
                 "metrics":{"valid_s1r":val_s1r,"valid_s23r":val_s23r,
                             "test_s1r":test_s1r,"test_s23r":test_s23r}},f)
print("MODEL SAVED : models/stage2_LGBM_v14_balanced_weight.pkl")
print("FINISHED")