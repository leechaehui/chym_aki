# ==========================================================
# 09_modeling_LGBM_v13_full_classweight.py
# Stage1 vs Stage2+3 | Full 35개 | class_weight
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
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)

OUT_DIR = "outputs/LGBM_v13_full_classweight"
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

# ==========================================================
print("=" * 70)
print("LGBM v13 | Full 35개 | class_weight | Stage1 vs Stage2+3")
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
y_stage_te = y_stage_test[test_mask]

# Stage1=0, Stage2+3=1
y_tr = (y_stage_train[train_mask] >= 2).astype(int)
y_vl = (y_stage_valid[valid_mask] >= 2).astype(int)
y_te = (y_stage_test[test_mask]   >= 2).astype(int)

print(f"Train: Stage1={sum(y_tr==0)}, Stage2+3={sum(y_tr==1)}")
print(f"Valid: Stage1={sum(y_vl==0)}, Stage2+3={sum(y_vl==1)}")
print(f"Feature Count: {X_tr.shape[1]}")
print()

# class_weight
classes = np.unique(y_tr)
weights = compute_class_weight("balanced", classes=classes, y=y_tr)
cw = dict(zip(classes, weights))
sw = np.array([cw[y] for y in y_tr])
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
    pred  = m.predict(X_vl)
    s23r  = recall_score(y_vl, pred, labels=[1], average=None)[0]
    macro = f1_score(y_vl, pred, average="macro")
    score = s23r * 0.6 + macro * 0.4
    trial_history.append([trial.number, s23r, macro, score])
    print(f"Trial {trial.number:03d} | S2+3={s23r:.4f} | Macro={macro:.4f}")
    return score

print("OPTUNA (50 trials)")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
trial_df = pd.DataFrame(trial_history, columns=["trial","s23r","macro","score"])
trial_df.to_csv(f"{OUT_DIR}/optuna_history.csv", index=False)
print(f"\nBest Score: {study.best_value:.5f}")

bp = study.best_params
bp.update({"objective":"binary","metric":"binary_logloss",
           "random_state":42,"n_jobs":-1,"verbose":-1})
model = LGBMClassifier(**bp)
model.fit(X_tr, y_tr, sample_weight=sw)

valid_proba = model.predict_proba(X_vl)[:, 1]
test_proba  = model.predict_proba(X_te)[:, 1]

# Threshold 탐색
best_t, best_spec = 0.5, 0
for t in np.arange(0.10, 0.91, 0.01):
    pred = (valid_proba >= t).astype(int)
    tn,fp,fn,tp_ = confusion_matrix(y_vl, pred).ravel()
    s23r = tp_/(tp_+fn) if (tp_+fn)>0 else 0
    spec = tn/(tn+fp)   if (tn+fp)>0  else 0
    if s23r >= 0.75 and spec > best_spec:
        best_spec = spec; best_t = t

valid_pred = (valid_proba >= best_t).astype(int)
test_pred  = (test_proba  >= best_t).astype(int)

tn,fp,fn,tp_ = confusion_matrix(y_vl, valid_pred).ravel()
val_s23r = tp_/(tp_+fn); val_s1r = tn/(tn+fp)
val_auroc= roc_auc_score(y_vl, valid_proba)
val_auprc= average_precision_score(y_vl, valid_proba)
val_f1   = f1_score(y_vl, valid_pred, zero_division=0)
val_prec = precision_score(y_vl, valid_pred, zero_division=0)

tn_t,fp_t,fn_t,tp_t = confusion_matrix(y_te, test_pred).ravel()
test_s23r = tp_t/(tp_t+fn_t); test_s1r = tn_t/(tn_t+fp_t)
test_auroc= roc_auc_score(y_te, test_proba)
test_auprc= average_precision_score(y_te, test_proba)
test_f1   = f1_score(y_te, test_pred, zero_division=0)

print(f"\nBest Threshold : {best_t:.2f}")
print("=" * 70)
print("VALIDATION RESULT")
print("=" * 70)
print(f"AUROC          : {val_auroc:.4f}")
print(f"AUPRC          : {val_auprc:.4f}")
print(f"F1             : {val_f1:.4f}")
print(f"Precision      : {val_prec:.4f}")
print(f"Stage1 Recall  : {val_s1r:.4f}")
print(f"Stage2+3 Recall: {val_s23r:.4f}  ← 핵심")
print()
print(classification_report(y_vl, valid_pred,
      target_names=["Stage1","Stage2+3"]))

# Stage별 세부
print("Stage별 세부 탐지율:")
for sv, sn in [(1,"Stage1"),(2,"Stage2"),(3,"Stage3")]:
    mask = y_stage_vl == sv
    if mask.sum() > 0:
        if sv == 1:
            c = (valid_pred[mask]==0).sum()
        else:
            c = (valid_pred[mask]==1).sum()
        print(f"  {sn}: {c}/{mask.sum()} ({c/mask.sum()*100:.1f}%)")

print()
print("=" * 70)
print("TEST RESULT")
print("=" * 70)
print(f"AUROC          : {test_auroc:.4f}")
print(f"AUPRC          : {test_auprc:.4f}")
print(f"F1             : {test_f1:.4f}")
print(f"Stage1 Recall  : {test_s1r:.4f}")
print(f"Stage2+3 Recall: {test_s23r:.4f}  ← 핵심")

cm = confusion_matrix(y_vl, valid_pred)
print(f"\nConfusion Matrix:\n{cm}")

# Figures
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Stage1","Stage2+3"],
            yticklabels=["Stage1","Stage2+3"])
plt.title("Confusion Matrix\n(v13 full classweight)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=300)
plt.close()

cm_pct = cm.astype(float)/cm.sum(axis=1)[:,np.newaxis]
plt.figure(figsize=(6,5))
sns.heatmap(cm_pct, annot=True, fmt='.1%', cmap='Blues',
            xticklabels=["Stage1","Stage2+3"],
            yticklabels=["Stage1","Stage2+3"])
plt.title("Normalized CM (v13 full classweight)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix_percent.png", dpi=300)
plt.close()

fpr,tpr,_ = roc_curve(y_vl, valid_proba)
plt.figure(figsize=(6,6))
plt.plot(fpr,tpr,color="#185FA5",label=f"AUROC={val_auroc:.4f}")
plt.plot([0,1],[0,1],"--",color="gray")
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC Curve (v13 full classweight)")
plt.legend(); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/roc_curve.png", dpi=300); plt.close()

metrics = {"AUROC":val_auroc,"AUPRC":val_auprc,"F1":val_f1,
           "S1 Recall":val_s1r,"S2+3 Recall":val_s23r}
plt.figure(figsize=(9,5))
bars = plt.bar(metrics.keys(), metrics.values(),
               color=["#185FA5"]*3+["#4A90D9","#E24B4A"])
plt.ylim(0,1.1)
for bar,val in zip(bars,metrics.values()):
    plt.text(bar.get_x()+bar.get_width()/2,
             bar.get_height()+0.02,f"{val:.4f}",
             ha="center",fontsize=10)
plt.title("Performance Summary (v13 full classweight)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/performance_summary.png", dpi=300); plt.close()

importance_df = pd.DataFrame({
    "feature":FEATURE_COLS,
    "importance":model.feature_importances_
}).sort_values("importance",ascending=False)
importance_df.to_csv(f"{OUT_DIR}/feature_importance.csv",index=False)
top15 = importance_df.head(15)
plt.figure(figsize=(10,8))
plt.barh(top15["feature"],top15["importance"],color="#185FA5")
plt.gca().invert_yaxis()
plt.title("Top 15 Feature Importance (v13 full classweight)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/feature_importance.png",dpi=300); plt.close()

print(f"\nSaved → {OUT_DIR}")

pd.DataFrame({
    "Metric":["AUROC","AUPRC","F1","S1 Recall","S2+3 Recall"],
    "Valid":[val_auroc,val_auprc,val_f1,val_s1r,val_s23r],
    "Test":[test_auroc,test_auprc,test_f1,test_s1r,test_s23r],
}).to_csv(f"{OUT_DIR}/results.csv",index=False)

with open("models/stage2_LGBM_v13_full_classweight.pkl","wb") as f:
    pickle.dump({"model":model,"feature_cols":FEATURE_COLS,
                 "best_threshold":best_t,"version":"v13_full_classweight",
                 "metrics":{"valid_auroc":val_auroc,"valid_s23r":val_s23r,
                             "test_auroc":test_auroc,"test_s23r":test_s23r}},f)

print("MODEL SAVED : models/stage2_LGBM_v13_full_classweight.pkl")
print("FINISHED")