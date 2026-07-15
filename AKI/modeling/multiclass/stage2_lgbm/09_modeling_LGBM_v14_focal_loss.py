# ==========================================================
# 09_modeling_LGBM_v14_focal_loss.py
# Stage1 vs Stage2+3 | Full 35개
# Method: Focal Loss 효과를 sample_weight로 근사
#         어려운 샘플(경계 케이스)에 높은 가중치
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
    f1_score, classification_report,
    confusion_matrix, roc_auc_score,
    average_precision_score, roc_curve
)

OUT_DIR = "outputs/LGBM_v14_focal_loss"
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
print("LGBM v14 | Focal Loss (sample_weight 근사) | Stage1 vs Stage2+3")
print("=" * 70)

# ==========================================================
# 데이터 로딩
# ==========================================================

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

print(f"Train: Stage1={sum(y_tr==0)}, Stage2+3={sum(y_tr==1)}")
print()

# ==========================================================
# Focal Weight 계산 함수
# 핵심 아이디어:
#   1단계: 기본 모델로 각 샘플의 예측 확률 계산
#   2단계: 확률이 애매한 샘플(경계 케이스)에 높은 가중치
#          → Focal Loss의 핵심 효과와 동일
# ==========================================================

def compute_focal_weights(proba, y_true, gamma=2.0, alpha=0.25):
    """
    Focal Loss 가중치 근사:
      w_i = alpha * (1-p_i)^gamma  (Positive 샘플)
      w_i = (1-alpha) * p_i^gamma  (Negative 샘플)

    p_i가 0.5에 가까울수록 (애매할수록) 높은 가중치
    p_i가 0 또는 1에 가까울수록 (확실할수록) 낮은 가중치
    """
    p = np.clip(proba, 1e-7, 1-1e-7)
    weights = np.where(
        y_true == 1,
        alpha * (1-p)**gamma,          # Stage2+3 샘플
        (1-alpha) * p**gamma           # Stage1 샘플
    )
    # 정규화
    weights = weights / weights.mean()
    return weights

# ==========================================================
# Step 1: 기본 모델 학습 → 확률 추출
# ==========================================================

print("=" * 70)
print("Step 1: 기본 모델로 초기 확률 추출")
print("=" * 70)

classes = np.unique(y_tr)
cw_arr  = compute_class_weight("balanced", classes=classes, y=y_tr)
cw_dict = dict(zip(classes, cw_arr))
sw_base = np.array([cw_dict[y] for y in y_tr])

base_model = LGBMClassifier(
    objective="binary", n_estimators=200,
    max_depth=5, learning_rate=0.05,
    random_state=42, n_jobs=-1, verbose=-1
)
base_model.fit(X_tr, y_tr, sample_weight=sw_base)
base_proba_tr = base_model.predict_proba(X_tr)[:, 1]

print(f"기본 모델 학습 완료")
print(f"Train 확률 분포: "
      f"S1 평균={base_proba_tr[y_tr==0].mean():.4f}, "
      f"S2+3 평균={base_proba_tr[y_tr==1].mean():.4f}")
print()

# ==========================================================
# Step 2: gamma 파라미터 탐색
# ==========================================================

print("=" * 70)
print("Step 2: Focal gamma 파라미터 탐색")
print("=" * 70)
print(f"{'gamma':>6} {'alpha':>6} | {'t':>5} | "
      f"{'S1':>8} {'S2+3':>8} {'F1':>8}")
print("-" * 55)

gamma_results = []

for gamma in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
    for alpha in [0.25, 0.50, 0.75]:
        # Focal 가중치 계산
        focal_w = compute_focal_weights(
            base_proba_tr, y_tr, gamma=gamma, alpha=alpha
        )

        m = LGBMClassifier(
            objective="binary", n_estimators=200,
            max_depth=5, learning_rate=0.05,
            random_state=42, n_jobs=-1, verbose=-1
        )
        m.fit(X_tr, y_tr, sample_weight=focal_w)
        proba_vl = m.predict_proba(X_vl)[:, 1]

        # Threshold 탐색
        best_t, best_sc = 0.5, 0
        for t in np.arange(0.10, 0.91, 0.01):
            pred = (proba_vl >= t).astype(int)
            if len(np.unique(pred)) < 2:
                continue
            tn,fp,fn,tp_ = confusion_matrix(y_vl, pred).ravel()
            s1r_  = tn/(tn+fp) if (tn+fp)>0 else 0
            s23r_ = tp_/(tp_+fn) if (tp_+fn)>0 else 0
            sc    = s1r_*0.5 + s23r_*0.5
            if s23r_ >= 0.75 and sc > best_sc:
                best_sc = sc; best_t = t

        pred = (proba_vl >= best_t).astype(int)
        if len(np.unique(pred)) < 2:
            continue

        tn,fp,fn,tp_ = confusion_matrix(y_vl, pred).ravel()
        s1r  = tn/(tn+fp) if (tn+fp)>0 else 0
        s23r = tp_/(tp_+fn) if (tp_+fn)>0 else 0
        f1   = f1_score(y_vl, pred, zero_division=0)

        gamma_results.append([gamma, alpha, best_t, s1r, s23r, f1])
        print(f"{gamma:>6} {alpha:>6} | {best_t:>5.2f} | "
              f"{s1r:>8.4f} {s23r:>8.4f} {f1:>8.4f}")

gamma_df = pd.DataFrame(
    gamma_results,
    columns=["gamma","alpha","threshold","s1r","s23r","f1"]
)
gamma_df.to_csv(f"{OUT_DIR}/focal_search.csv", index=False)

# 최적 파라미터 선정
cond = gamma_df[gamma_df["s23r"] >= 0.75]
if len(cond) > 0:
    best_row   = cond.loc[cond["s1r"].idxmax()]
    print("\n[S2+3≥0.75 + S1 최대]")
else:
    best_row   = gamma_df.loc[gamma_df["s1r"].idxmax()]
    print("\n[S1 최대 (조건 완화)]")

best_gamma = best_row["gamma"]
best_alpha = best_row["alpha"]
print(f"Best gamma={best_gamma}, alpha={best_alpha}")
print(f"예상 S1={best_row['s1r']:.4f}, S2+3={best_row['s23r']:.4f}")
print()

# ==========================================================
# Step 3: 최적 gamma로 Optuna 튜닝
# ==========================================================

focal_w_best = compute_focal_weights(
    base_proba_tr, y_tr,
    gamma=best_gamma, alpha=best_alpha
)

trial_history = []

def objective(trial):
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
    m.fit(X_tr, y_tr, sample_weight=focal_w_best)
    proba_vl = m.predict_proba(X_vl)[:, 1]

    best_t, best_sc = 0.5, 0
    for t in np.arange(0.10, 0.91, 0.01):
        pred = (proba_vl >= t).astype(int)
        if len(np.unique(pred)) < 2:
            continue
        tn,fp,fn,tp_ = confusion_matrix(y_vl, pred).ravel()
        s1r_  = tn/(tn+fp) if (tn+fp)>0 else 0
        s23r_ = tp_/(tp_+fn) if (tp_+fn)>0 else 0
        sc    = s1r_*0.5 + s23r_*0.5
        if s23r_ >= 0.75 and sc > best_sc:
            best_sc = sc; best_t = t

    pred = (proba_vl >= best_t).astype(int)
    if len(np.unique(pred)) < 2:
        return 0.0

    tn,fp,fn,tp_ = confusion_matrix(y_vl, pred).ravel()
    s1r  = tn/(tn+fp) if (tn+fp)>0 else 0
    s23r = tp_/(tp_+fn) if (tp_+fn)>0 else 0
    score = s1r*0.5 + s23r*0.5

    trial_history.append([trial.number, s1r, s23r, score])
    print(f"Trial {trial.number:03d} | "
          f"S1={s1r:.4f} | S2+3={s23r:.4f} | Score={score:.4f}")
    return score

print("=" * 70)
print(f"OPTUNA (50 trials | gamma={best_gamma} alpha={best_alpha})")
print("=" * 70)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

trial_df = pd.DataFrame(
    trial_history, columns=["trial","s1r","s23r","score"]
)
trial_df.to_csv(f"{OUT_DIR}/optuna_history.csv", index=False)
print(f"\nBest Score: {study.best_value:.5f}")
for k, v in study.best_params.items():
    print(f"  {k:<25} : {v}")
print()

# ==========================================================
# 최종 모델
# ==========================================================

bp = study.best_params
bp.update({
    "objective":"binary","metric":"binary_logloss",
    "random_state":42,"n_jobs":-1,"verbose":-1
})
model = LGBMClassifier(**bp)
model.fit(X_tr, y_tr, sample_weight=focal_w_best)

valid_proba = model.predict_proba(X_vl)[:, 1]
test_proba  = model.predict_proba(X_te)[:, 1]

best_t, best_sc = 0.5, 0
for t in np.arange(0.10, 0.91, 0.01):
    pred = (valid_proba >= t).astype(int)
    if len(np.unique(pred)) < 2:
        continue
    tn,fp,fn,tp_ = confusion_matrix(y_vl, pred).ravel()
    s1r_  = tn/(tn+fp) if (tn+fp)>0 else 0
    s23r_ = tp_/(tp_+fn) if (tp_+fn)>0 else 0
    sc    = s1r_*0.5 + s23r_*0.5
    if s23r_ >= 0.75 and sc > best_sc:
        best_sc = sc; best_t = t

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
for sv, sn in [(1,"Stage1"),(2,"Stage2"),(3,"Stage3")]:
    mask = y_stage_vl == sv
    if mask.sum() > 0:
        c = (valid_pred[mask]==0).sum() if sv==1 \
            else (valid_pred[mask]==1).sum()
        print(f"  {sn}: {c}/{mask.sum()} ({c/mask.sum()*100:.1f}%)")
print()
print("=" * 70)
print("TEST RESULT")
print("=" * 70)
print(f"AUROC={test_auroc:.4f} | S1={test_s1r:.4f} | S2+3={test_s23r:.4f}")

cm = confusion_matrix(y_vl, valid_pred)
print(f"\nConfusion Matrix:\n{cm}")

# ==========================================================
# FIGURES
# ==========================================================

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Stage1","Stage2+3"],
            yticklabels=["Stage1","Stage2+3"])
plt.title(f"Confusion Matrix\n(v14 focal γ={best_gamma} α={best_alpha})")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=300); plt.close()

cm_pct = cm.astype(float)/cm.sum(axis=1)[:,np.newaxis]
plt.figure(figsize=(6,5))
sns.heatmap(cm_pct, annot=True, fmt='.1%', cmap='Blues',
            xticklabels=["Stage1","Stage2+3"],
            yticklabels=["Stage1","Stage2+3"])
plt.title("Normalized CM (v14 focal_loss)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix_percent.png", dpi=300); plt.close()

fpr,tpr,_ = roc_curve(y_vl, valid_proba)
plt.figure(figsize=(6,6))
plt.plot(fpr,tpr,color="#185FA5",label=f"AUROC={val_auroc:.4f}")
plt.plot([0,1],[0,1],"--",color="gray")
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC Curve (v14 focal_loss)")
plt.legend(); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/roc_curve.png", dpi=300); plt.close()

pivot_s1 = gamma_df.pivot_table(
    values="s1r", index="gamma", columns="alpha")
plt.figure(figsize=(8,5))
sns.heatmap(pivot_s1, annot=True, fmt='.3f', cmap='Blues')
plt.title("Focal 파라미터별 S1 Recall")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/focal_heatmap_s1.png", dpi=300); plt.close()

pivot_s23 = gamma_df.pivot_table(
    values="s23r", index="gamma", columns="alpha")
plt.figure(figsize=(8,5))
sns.heatmap(pivot_s23, annot=True, fmt='.3f', cmap='Reds')
plt.title("Focal 파라미터별 S2+3 Recall")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/focal_heatmap_s23.png", dpi=300); plt.close()

plt.figure(figsize=(10,5))
plt.plot(trial_df["trial"], trial_df["s1r"],
         label="S1 Recall", color="#185FA5", marker="o", markersize=3)
plt.plot(trial_df["trial"], trial_df["s23r"],
         label="S2+3 Recall", color="#E24B4A", marker="o", markersize=3)
plt.axhline(0.50, color="#185FA5", linestyle="--", alpha=0.4)
plt.axhline(0.75, color="#E24B4A", linestyle="--", alpha=0.4)
plt.xlabel("Trial"); plt.ylabel("Score")
plt.title("Optuna History (v14 focal_loss)")
plt.legend(); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/optuna_history.png", dpi=300); plt.close()

# Focal 가중치 분포 시각화
plt.figure(figsize=(8,5))
plt.hist(focal_w_best[y_tr==0], bins=30, alpha=0.5,
         label="Stage1 weights", color="#185FA5")
plt.hist(focal_w_best[y_tr==1], bins=30, alpha=0.5,
         label="Stage2+3 weights", color="#E24B4A")
plt.xlabel("Sample Weight"); plt.ylabel("Count")
plt.title(f"Focal 가중치 분포 (γ={best_gamma}, α={best_alpha})")
plt.legend(); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/focal_weight_distribution.png", dpi=300); plt.close()

importance_df = pd.DataFrame({
    "feature"   : FEATURE_COLS,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)
importance_df.to_csv(f"{OUT_DIR}/feature_importance.csv", index=False)
top15 = importance_df.head(15)
plt.figure(figsize=(10,8))
plt.barh(top15["feature"], top15["importance"], color="#185FA5")
plt.gca().invert_yaxis()
plt.title("Top 15 Feature Importance (v14 focal_loss)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=300); plt.close()

print(f"\nSaved → {OUT_DIR}")

pd.DataFrame({
    "Metric": ["AUROC","AUPRC","F1","S1 Recall","S2+3 Recall"],
    "Valid" : [val_auroc,val_auprc,val_f1,val_s1r,val_s23r],
    "Test"  : [test_auroc,
               average_precision_score(y_te,test_proba),
               test_f1,test_s1r,test_s23r],
}).to_csv(f"{OUT_DIR}/results.csv", index=False)

with open("models/stage2_LGBM_v14_focal_loss.pkl","wb") as f:
    pickle.dump({
        "model"         : model,
        "base_model"    : base_model,
        "feature_cols"  : FEATURE_COLS,
        "best_threshold": best_t,
        "gamma"         : best_gamma,
        "alpha"         : best_alpha,
        "version"       : "v14_focal_loss",
        "metrics"       : {
            "valid_s1r"  : val_s1r,
            "valid_s23r" : val_s23r,
            "valid_auroc": val_auroc,
            "test_s1r"   : test_s1r,
            "test_s23r"  : test_s23r,
        }
    }, f)

print("MODEL SAVED : models/stage2_LGBM_v14_focal_loss.pkl")
print("FINISHED")