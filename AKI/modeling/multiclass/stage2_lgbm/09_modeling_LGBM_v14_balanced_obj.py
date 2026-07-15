# ==========================================================
# 09_modeling_LGBM_v14_balanced_obj.py
# Stage1 vs Stage2+3 | Full 35개
# Method: Optuna 목표 함수 변경
#         S2+3×0.5 + S1×0.3 + Macro×0.2
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
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, roc_curve
)

OUT_DIR = "outputs/LGBM_v14_balanced_obj"
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
print("LGBM v14 | Balanced Objective | Stage1 vs Stage2+3")
print("목표: S2+3×0.5 + S1×0.3 + Macro×0.2")
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

classes = np.unique(y_tr)
weights = compute_class_weight("balanced", classes=classes, y=y_tr)
cw = dict(zip(classes, weights))
sw = np.array([cw[y] for y in y_tr])
print(f"Train: Stage1={sum(y_tr==0)}, Stage2+3={sum(y_tr==1)}")
print(f"Weight: Stage1={cw[0]:.4f}, Stage2+3={cw[1]:.4f}")
print()

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
    m.fit(X_tr, y_tr, sample_weight=sw)
    pred = m.predict(X_vl)
    tn,fp,fn,tp_ = confusion_matrix(y_vl, pred).ravel()
    s1r  = tn/(tn+fp) if (tn+fp)>0 else 0
    s23r = tp_/(tp_+fn) if (tp_+fn)>0 else 0
    macro = f1_score(y_vl, pred, average="macro")
    # S1 Recall을 목표 함수에 직접 포함
    score = s23r*0.5 + s1r*0.3 + macro*0.2
    trial_history.append([trial.number, s1r, s23r, macro, score])
    print(f"Trial {trial.number:03d} | S1={s1r:.4f} | S2+3={s23r:.4f} | Score={score:.4f}")
    return score

print("=" * 70)
print("OPTUNA (50 trials)")
print("=" * 70)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

trial_df = pd.DataFrame(trial_history,
    columns=["trial","s1r","s23r","macro","score"])
trial_df.to_csv(f"{OUT_DIR}/optuna_history.csv", index=False)
print(f"\nBest Score: {study.best_value:.5f}")

bp = study.best_params
bp.update({"objective":"binary","metric":"binary_logloss",
           "random_state":42,"n_jobs":-1,"verbose":-1})
model = LGBMClassifier(**bp)
model.fit(X_tr, y_tr, sample_weight=sw)

valid_proba = model.predict_proba(X_vl)[:,1]
test_proba  = model.predict_proba(X_te)[:,1]

# Threshold: S1 ≥ 0.50 + S2+3 최대
best_t, best_score = 0.5, 0
for t in np.arange(0.10, 0.91, 0.01):
    pred = (valid_proba >= t).astype(int)
    tn,fp,fn,tp_ = confusion_matrix(y_vl, pred).ravel()
    s1r  = tn/(tn+fp) if (tn+fp)>0 else 0
    s23r = tp_/(tp_+fn) if (tp_+fn)>0 else 0
    score = s1r*0.5 + s23r*0.5
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
print("VALIDATION RESULT")
print(f"AUROC={val_auroc:.4f} | F1={val_f1:.4f}")
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

# Optuna 목표 함수 비교 시각화
plt.figure(figsize=(10,5))
plt.plot(trial_df["trial"], trial_df["s1r"],
         label="S1 Recall", color="#185FA5", marker="o", markersize=3)
plt.plot(trial_df["trial"], trial_df["s23r"],
         label="S2+3 Recall", color="#E24B4A", marker="o", markersize=3)
plt.plot(trial_df["trial"], trial_df["score"],
         label="Score (0.5×S2+3 + 0.3×S1 + 0.2×Macro)",
         color="#E67E22", marker="o", markersize=3, linestyle="--")
plt.axhline(0.50, color="#185FA5", linestyle=":", alpha=0.5)
plt.axhline(0.75, color="#E24B4A", linestyle=":", alpha=0.5)
plt.xlabel("Trial"); plt.ylabel("Score")
plt.title("Optuna History (v14 balanced_obj)")
plt.legend(); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/optuna_history.png",dpi=300); plt.close()

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Stage1","Stage2+3"],yticklabels=["Stage1","Stage2+3"])
plt.title(f"Confusion Matrix (v14 balanced_obj t={best_t})")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/confusion_matrix.png",dpi=300); plt.close()

cm_pct=cm.astype(float)/cm.sum(axis=1)[:,np.newaxis]
plt.figure(figsize=(6,5))
sns.heatmap(cm_pct,annot=True,fmt='.1%',cmap='Blues',
            xticklabels=["Stage1","Stage2+3"],yticklabels=["Stage1","Stage2+3"])
plt.title("Normalized CM (v14 balanced_obj)")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/confusion_matrix_percent.png",dpi=300); plt.close()

fpr,tpr,_=roc_curve(y_vl,valid_proba)
plt.figure(figsize=(6,6))
plt.plot(fpr,tpr,color="#185FA5",label=f"AUROC={val_auroc:.4f}")
plt.plot([0,1],[0,1],"--",color="gray")
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC Curve (v14 balanced_obj)")
plt.legend(); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/roc_curve.png",dpi=300); plt.close()

importance_df=pd.DataFrame({"feature":FEATURE_COLS,
    "importance":model.feature_importances_}).sort_values("importance",ascending=False)
importance_df.to_csv(f"{OUT_DIR}/feature_importance.csv",index=False)
top15=importance_df.head(15)
plt.figure(figsize=(10,8))
plt.barh(top15["feature"],top15["importance"],color="#185FA5")
plt.gca().invert_yaxis(); plt.title("Top 15 (v14 balanced_obj)")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/feature_importance.png",dpi=300); plt.close()

print(f"\nSaved → {OUT_DIR}")
pd.DataFrame({"Metric":["AUROC","AUPRC","F1","S1 Recall","S2+3 Recall"],
    "Valid":[val_auroc,val_auprc,val_f1,val_s1r,val_s23r],
    "Test":[test_auroc,average_precision_score(y_te,test_proba),test_f1,test_s1r,test_s23r],
}).to_csv(f"{OUT_DIR}/results.csv",index=False)

with open("models/stage2_LGBM_v14_balanced_obj.pkl","wb") as f:
    pickle.dump({"model":model,"feature_cols":FEATURE_COLS,
                 "best_threshold":best_t,"version":"v14_balanced_obj",
                 "metrics":{"valid_s1r":val_s1r,"valid_s23r":val_s23r,
                             "test_s1r":test_s1r,"test_s23r":test_s23r}},f)
print("MODEL SAVED : models/stage2_LGBM_v14_balanced_obj.pkl")
print("FINISHED")