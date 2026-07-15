# ==========================================================
# 09_modeling_LGBM_v14_calibration.py
# Stage1 vs Stage2+3 | Full 35개
# Method: Probability Calibration
#         Platt Scaling + Isotonic Regression
# ==========================================================

import os, warnings, pickle
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus'] = False

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    f1_score, recall_score, classification_report,
    confusion_matrix, roc_auc_score,
    average_precision_score, roc_curve
)
from sklearn.utils.class_weight import compute_class_weight

OUT_DIR = "outputs/LGBM_v14_calibration"
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
print("LGBM v14 | Calibration | Stage1 vs Stage2+3")
print("Platt Scaling + Isotonic Regression 비교")
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

classes = np.unique(y_tr)
weights = compute_class_weight("balanced", classes=classes, y=y_tr)
cw = dict(zip(classes, weights))
sw = np.array([cw[y] for y in y_tr])

# ==========================================================
# v13_full_classweight 베이스 모델 로딩
# ==========================================================

with open("models/stage2_LGBM_v13_full_classweight.pkl", "rb") as f:
    v13 = pickle.load(f)

base_model = v13["model"]
print("v13_full_classweight 베이스 모델 로딩 완료")
print()

# ==========================================================
# 기본 모델 확률 분포 확인
# ==========================================================

train_proba_raw = base_model.predict_proba(X_tr)[:, 1]
valid_proba_raw = base_model.predict_proba(X_vl)[:, 1]
test_proba_raw  = base_model.predict_proba(X_te)[:, 1]

print("기본 모델 확률 분포 (Valid):")
print(f"  전체 평균     : {valid_proba_raw.mean():.4f}")
print(f"  중앙값        : {np.median(valid_proba_raw):.4f}")
print(f"  Stage1 평균   : {valid_proba_raw[y_vl==0].mean():.4f}")
print(f"  Stage2+3 평균 : {valid_proba_raw[y_vl==1].mean():.4f}")
print()

# ==========================================================
# Calibration 적용
# Train set으로 보정 함수 학습
# Valid/Test에 적용
# ==========================================================

print("=" * 70)
print("Calibration 적용")
print("=" * 70)

results = {}

for method in ["sigmoid", "isotonic"]:
    print(f"\n[{method.upper()}]")

    if method == "sigmoid":
        # Platt Scaling — LogisticRegression으로 확률 보정
        cal_model = LogisticRegression(random_state=42)
        cal_model.fit(train_proba_raw.reshape(-1, 1), y_tr)
        cal_proba_vl = cal_model.predict_proba(
            valid_proba_raw.reshape(-1, 1))[:, 1]
        cal_proba_te = cal_model.predict_proba(
            test_proba_raw.reshape(-1, 1))[:, 1]

    else:
        # Isotonic Regression
        cal_model = IsotonicRegression(out_of_bounds="clip")
        cal_model.fit(train_proba_raw, y_tr)
        cal_proba_vl = cal_model.predict(valid_proba_raw)
        cal_proba_te = cal_model.predict(test_proba_raw)

    print(f"보정 후 확률 분포 (Valid):")
    print(f"  Stage1 평균   : {cal_proba_vl[y_vl==0].mean():.4f}")
    print(f"  Stage2+3 평균 : {cal_proba_vl[y_vl==1].mean():.4f}")

    # Threshold 탐색 (S2+3 ≥ 0.75 + S1 최대)
    best_t, best_score = 0.5, 0
    for t in np.arange(0.10, 0.91, 0.01):
        pred = (cal_proba_vl >= t).astype(int)
        if len(np.unique(pred)) < 2:
            continue
        tn, fp, fn, tp_ = confusion_matrix(y_vl, pred).ravel()
        s1r  = tn/(tn+fp) if (tn+fp)>0 else 0
        s23r = tp_/(tp_+fn) if (tp_+fn)>0 else 0
        score = s1r*0.5 + s23r*0.5
        if s23r >= 0.75 and score > best_score:
            best_score = score; best_t = t

    cal_pred_vl = (cal_proba_vl >= best_t).astype(int)
    cal_pred_te = (cal_proba_te >= best_t).astype(int)

    tn,fp,fn,tp_ = confusion_matrix(y_vl, cal_pred_vl).ravel()
    s1r  = tn/(tn+fp); s23r = tp_/(tp_+fn)
    f1   = f1_score(y_vl, cal_pred_vl, zero_division=0)
    auroc= roc_auc_score(y_vl, cal_proba_vl)
    auprc= average_precision_score(y_vl, cal_proba_vl)

    tn_t,fp_t,fn_t,tp_t = confusion_matrix(y_te, cal_pred_te).ravel()
    ts1r  = tn_t/(tn_t+fp_t); ts23r = tp_t/(tp_t+fn_t)
    test_auroc = roc_auc_score(y_te, cal_proba_te)
    test_f1    = f1_score(y_te, cal_pred_te, zero_division=0)

    print(f"Best Threshold : {best_t:.2f}")
    print(f"Valid: S1={s1r:.4f} | S2+3={s23r:.4f} | "
          f"F1={f1:.4f} | AUROC={auroc:.4f}")
    print(f"Test:  S1={ts1r:.4f} | S2+3={ts23r:.4f}")
    print()
    print(classification_report(y_vl, cal_pred_vl,
          target_names=["Stage1","Stage2+3"]))
    print("Stage별 세부:")
    for sv, sn in [(1,"Stage1"),(2,"Stage2"),(3,"Stage3")]:
        mask = y_stage_vl == sv
        if mask.sum() > 0:
            c = (cal_pred_vl[mask]==0).sum() if sv==1 \
                else (cal_pred_vl[mask]==1).sum()
            print(f"  {sn}: {c}/{mask.sum()} ({c/mask.sum()*100:.1f}%)")

    results[method] = {
        "cal_model"  : cal_model,
        "threshold"  : best_t,
        "proba_vl"   : cal_proba_vl,
        "proba_te"   : cal_proba_te,
        "pred_vl"    : cal_pred_vl,
        "pred_te"    : cal_pred_te,
        "s1r"        : s1r,
        "s23r"       : s23r,
        "f1"         : f1,
        "auroc"      : auroc,
        "auprc"      : auprc,
        "ts1r"       : ts1r,
        "ts23r"      : ts23r,
        "test_auroc" : test_auroc,
        "test_f1"    : test_f1,
    }

# ==========================================================
# FIGURES
# ==========================================================

print("=" * 70)
print("GENERATE FIGURES")
print("=" * 70)

# 1. Calibration Curve 비교
plt.figure(figsize=(8, 6))
plt.plot([0,1],[0,1],"k--",label="완벽한 보정")
for name, color, proba in [
    ("Raw",      "gray",      valid_proba_raw),
    ("Sigmoid",  "#185FA5",   results["sigmoid"]["proba_vl"]),
    ("Isotonic", "#E24B4A",   results["isotonic"]["proba_vl"]),
]:
    prob_true, prob_pred = calibration_curve(
        y_vl, proba, n_bins=10
    )
    plt.plot(prob_pred, prob_true, marker="o",
             color=color, label=name)
plt.xlabel("예측 확률")
plt.ylabel("실제 비율")
plt.title("Calibration Curve 비교")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/calibration_curve.png", dpi=300)
plt.close()

# 2. 확률 분포 비교 (Raw vs Sigmoid vs Isotonic)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (name, proba) in zip(axes, [
    ("Raw",      valid_proba_raw),
    ("Sigmoid",  results["sigmoid"]["proba_vl"]),
    ("Isotonic", results["isotonic"]["proba_vl"]),
]):
    ax.hist(proba[y_vl==0], bins=30, alpha=0.5,
            label="Stage1", color="#185FA5")
    ax.hist(proba[y_vl==1], bins=30, alpha=0.5,
            label="Stage2+3", color="#E24B4A")
    s1_mean  = proba[y_vl==0].mean()
    s23_mean = proba[y_vl==1].mean()
    ax.axvline(s1_mean,  color="#185FA5", linestyle="--",
               alpha=0.8, label=f"S1 평균={s1_mean:.3f}")
    ax.axvline(s23_mean, color="#E24B4A", linestyle="--",
               alpha=0.8, label=f"S2+3 평균={s23_mean:.3f}")
    ax.set_title(f"{name} 확률 분포")
    ax.set_xlabel("P(Stage2+3)")
    ax.legend(fontsize=8)
plt.suptitle("Calibration 전후 확률 분포 비교")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/probability_distribution.png", dpi=300)
plt.close()

# 3. 두 방법 성능 비교
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
methods = ["sigmoid", "isotonic"]
s1rs   = [results[m]["s1r"]  for m in methods]
s23rs  = [results[m]["s23r"] for m in methods]
aurocs = [results[m]["auroc"] for m in methods]

x = np.arange(len(methods))
w = 0.35
axes[0].bar(x-w/2, s1rs,  w, label="Stage1 Recall",   color="#185FA5")
axes[0].bar(x+w/2, s23rs, w, label="Stage2+3 Recall", color="#E24B4A")
axes[0].set_xticks(x)
axes[0].set_xticklabels(["Sigmoid\n(Platt)", "Isotonic"])
axes[0].set_ylim(0, 1.1)
axes[0].axhline(0.50, color="#185FA5", linestyle="--",
                alpha=0.5, label="S1 목표 0.50")
axes[0].axhline(0.75, color="#E24B4A", linestyle="--",
                alpha=0.5, label="S2+3 목표 0.75")
for i,(s1,s23) in enumerate(zip(s1rs,s23rs)):
    axes[0].text(i-w/2, s1+0.02, f"{s1:.3f}", ha='center', fontsize=10)
    axes[0].text(i+w/2, s23+0.02, f"{s23:.3f}", ha='center', fontsize=10)
axes[0].legend(); axes[0].set_title("Valid Recall 비교")

axes[1].bar(x, aurocs, color=["#185FA5","#E24B4A"])
axes[1].set_xticks(x)
axes[1].set_xticklabels(["Sigmoid\n(Platt)", "Isotonic"])
axes[1].set_ylim(0, 1.0)
for i, a in enumerate(aurocs):
    axes[1].text(i, a+0.01, f"{a:.4f}", ha='center', fontsize=10)
axes[1].set_title("AUROC 비교")

plt.suptitle("Calibration 방법별 성능 비교 (v14)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/method_comparison.png", dpi=300)
plt.close()

# 4. 최종 채택 모델 Confusion Matrix
best_method = max(results, key=lambda x: results[x]["s1r"])
best_result = results[best_method]
print(f"최종 채택: {best_method.upper()}")
print(f"S1 Recall  : {best_result['s1r']:.4f}")
print(f"S2+3 Recall: {best_result['s23r']:.4f}")

cm = confusion_matrix(y_vl, best_result["pred_vl"])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Stage1","Stage2+3"],
            yticklabels=["Stage1","Stage2+3"])
plt.title(f"Confusion Matrix (v14 calibration {best_method})")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=300)
plt.close()

cm_pct = cm.astype(float)/cm.sum(axis=1)[:,np.newaxis]
plt.figure(figsize=(6, 5))
sns.heatmap(cm_pct, annot=True, fmt='.1%', cmap='Blues',
            xticklabels=["Stage1","Stage2+3"],
            yticklabels=["Stage1","Stage2+3"])
plt.title(f"Normalized CM (v14 calibration {best_method})")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix_percent.png", dpi=300)
plt.close()

fpr, tpr, _ = roc_curve(y_vl, best_result["proba_vl"])
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color="#185FA5",
         label=f"AUROC={best_result['auroc']:.4f}")
plt.plot([0,1],[0,1],"--",color="gray")
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title(f"ROC Curve (v14 calibration {best_method})")
plt.legend(); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/roc_curve.png", dpi=300)
plt.close()

print(f"\nSaved → {OUT_DIR}")

# ==========================================================
# 결과 CSV
# ==========================================================

pd.DataFrame({
    "Metric" : ["AUROC","AUPRC","F1","S1 Recall","S2+3 Recall"],
    "Valid"  : [best_result["auroc"], best_result["auprc"],
                best_result["f1"], best_result["s1r"],
                best_result["s23r"]],
    "Test"   : [best_result["test_auroc"],
                average_precision_score(y_te, best_result["proba_te"]),
                best_result["test_f1"],
                best_result["ts1r"], best_result["ts23r"]],
}).to_csv(f"{OUT_DIR}/results.csv", index=False)

# 전체 방법 비교 CSV
pd.DataFrame([{
    "method"   : m,
    "threshold": results[m]["threshold"],
    "s1r"      : results[m]["s1r"],
    "s23r"     : results[m]["s23r"],
    "auroc"    : results[m]["auroc"],
    "ts1r"     : results[m]["ts1r"],
    "ts23r"    : results[m]["ts23r"],
} for m in results]).to_csv(f"{OUT_DIR}/calibration_comparison.csv", index=False)

# ==========================================================
# SAVE MODEL
# ==========================================================

with open("models/stage2_LGBM_v14_calibration.pkl", "wb") as f:
    pickle.dump({
        "base_model"   : base_model,
        "cal_model"    : best_result["cal_model"],
        "feature_cols" : FEATURE_COLS,
        "best_threshold": best_result["threshold"],
        "cal_method"   : best_method,
        "version"      : "v14_calibration",
        "metrics"      : {
            "valid_s1r"  : best_result["s1r"],
            "valid_s23r" : best_result["s23r"],
            "valid_auroc": best_result["auroc"],
            "test_s1r"   : best_result["ts1r"],
            "test_s23r"  : best_result["ts23r"],
        }
    }, f)

print("MODEL SAVED : models/stage2_LGBM_v14_calibration.pkl")
print("FINISHED")