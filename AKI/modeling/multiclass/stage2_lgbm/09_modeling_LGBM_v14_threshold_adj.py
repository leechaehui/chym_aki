# ==========================================================
# 09_modeling_LGBM_v14_threshold_adj.py
# Stage1 vs Stage2+3 | Full 35개
# Method: Threshold 재조정
#         full_classweight 모델 재활용
#         S1 Recall ≥ 0.50 + S2+3 Recall 최대화
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

from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, roc_curve
)

OUT_DIR = "outputs/LGBM_v14_threshold_adj"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("LGBM v14 | Threshold 재조정 | v13_full_classweight 재활용")
print("=" * 70)

# ==========================================================
# v13_full_classweight 모델 로딩
# ==========================================================

with open("models/stage2_LGBM_v13_full_classweight.pkl", "rb") as f:
    v13 = pickle.load(f)

model        = v13["model"]
FEATURE_COLS = v13["feature_cols"]
old_t        = v13["best_threshold"]
print(f"기존 Threshold : {old_t}")
print(f"피처 수        : {len(FEATURE_COLS)}")
print()

# ==========================================================
# 데이터 로딩
# ==========================================================

X_train_orig  = np.load("data/X_train.npy")
X_valid_orig  = np.load("data/X_valid.npy")
X_test_orig   = np.load("data/X_test.npy")
y_stage_train = np.load("data/y_stage_train.npy")
y_stage_valid = np.load("data/y_stage_valid.npy")
y_stage_test  = np.load("data/y_stage_test.npy")

valid_mask = y_stage_valid > 0
test_mask  = y_stage_test  > 0

X_vl = X_valid_orig[valid_mask]
X_te = X_test_orig[test_mask]
y_stage_vl = y_stage_valid[valid_mask]

y_vl = (y_stage_valid[valid_mask] >= 2).astype(int)
y_te = (y_stage_test[test_mask]   >= 2).astype(int)

valid_proba = model.predict_proba(X_vl)[:, 1]
test_proba  = model.predict_proba(X_te)[:, 1]

# ==========================================================
# Threshold 전체 탐색
# ==========================================================

print("=" * 70)
print("Threshold 전체 탐색")
print("=" * 70)
print(f"{'Threshold':>10} | {'S1 Recall':>10} | {'S2+3 Recall':>12} | {'F1':>8} | {'AUROC':>8}")
print("-" * 60)

results = []
for t in np.arange(0.10, 0.91, 0.01):
    pred = (valid_proba >= t).astype(int)
    tn, fp, fn, tp_ = confusion_matrix(y_vl, pred).ravel()
    s1r  = tn / (tn + fp) if (tn + fp) > 0 else 0
    s23r = tp_ / (tp_ + fn) if (tp_ + fn) > 0 else 0
    f1   = f1_score(y_vl, pred, zero_division=0)
    results.append([round(t, 2), s1r, s23r, f1])
    print(f"{t:>10.2f} | {s1r:>10.4f} | {s23r:>12.4f} | {f1:>8.4f}")

results_df = pd.DataFrame(
    results, columns=["threshold","s1r","s23r","f1"]
)
results_df.to_csv(f"{OUT_DIR}/threshold_search.csv", index=False)

# ==========================================================
# 목표별 최적 Threshold 선정
# ==========================================================

print()
print("=" * 70)
print("목표별 최적 Threshold 선정")
print("=" * 70)

# 목표 1: S2+3 ≥ 0.75 + S1 최대
cond1 = results_df[results_df["s23r"] >= 0.75]
t_balanced = cond1.loc[cond1["s1r"].idxmax(), "threshold"] if len(cond1) > 0 else 0.56
row1 = results_df[results_df["threshold"] == t_balanced].iloc[0]
print(f"[균형] S2+3 ≥ 0.75 + S1 최대")
print(f"  Threshold : {t_balanced}")
print(f"  S1 Recall : {row1['s1r']:.4f}")
print(f"  S2+3 Recall: {row1['s23r']:.4f}")
print()

# 목표 2: S1 ≥ 0.50 + S2+3 최대
cond2 = results_df[results_df["s1r"] >= 0.50]
t_s1focus = cond2.loc[cond2["s23r"].idxmax(), "threshold"] if len(cond2) > 0 else 0.70
row2 = results_df[results_df["threshold"] == t_s1focus].iloc[0]
print(f"[S1 중시] S1 ≥ 0.50 + S2+3 최대")
print(f"  Threshold : {t_s1focus}")
print(f"  S1 Recall : {row2['s1r']:.4f}")
print(f"  S2+3 Recall: {row2['s23r']:.4f}")
print()

# 목표 3: F1 최대
t_f1 = results_df.loc[results_df["f1"].idxmax(), "threshold"]
row3 = results_df[results_df["threshold"] == t_f1].iloc[0]
print(f"[F1 최대]")
print(f"  Threshold : {t_f1}")
print(f"  S1 Recall : {row3['s1r']:.4f}")
print(f"  S2+3 Recall: {row3['s23r']:.4f}")
print(f"  F1        : {row3['f1']:.4f}")
print()

# ==========================================================
# 세 Threshold 모두 평가
# ==========================================================

print("=" * 70)
print("세 Threshold 비교 평가")
print("=" * 70)

thresholds = {
    f"균형 (t={t_balanced})": t_balanced,
    f"S1중시 (t={t_s1focus})": t_s1focus,
    f"F1최대 (t={t_f1})": t_f1,
    f"기존 (t={old_t})": old_t,
}

best_t = t_balanced  # 최종 채택 기본값

summary = []
for name, t in thresholds.items():
    vp = (valid_proba >= t).astype(int)
    tp_ = (test_proba >= t).astype(int)
    tn, fp, fn, tpp = confusion_matrix(y_vl, vp).ravel()
    s1r  = tn / (tn + fp)
    s23r = tpp / (tpp + fn)
    f1   = f1_score(y_vl, vp, zero_division=0)
    auroc = roc_auc_score(y_vl, valid_proba)

    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_te, tp_).ravel()
    ts1r  = tn_t / (tn_t + fp_t)
    ts23r = tp_t / (tp_t + fn_t)

    summary.append([name, t, s1r, s23r, f1, auroc, ts1r, ts23r])
    print(f"\n{name}")
    print(f"  Valid S1={s1r:.4f} | S2+3={s23r:.4f} | F1={f1:.4f}")
    print(f"  Test  S1={ts1r:.4f} | S2+3={ts23r:.4f}")
    print(classification_report(y_vl, vp,
          target_names=["Stage1","Stage2+3"]))
    print("  Stage별 세부:")
    for sv, sn in [(1,"Stage1"),(2,"Stage2"),(3,"Stage3")]:
        mask = y_stage_vl == sv
        if mask.sum() > 0:
            c = (vp[mask]==0).sum() if sv==1 else (vp[mask]==1).sum()
            print(f"    {sn}: {c}/{mask.sum()} ({c/mask.sum()*100:.1f}%)")

summary_df = pd.DataFrame(summary, columns=[
    "name","threshold","s1r","s23r","f1","auroc","test_s1r","test_s23r"
])
summary_df.to_csv(f"{OUT_DIR}/threshold_comparison.csv", index=False)

# ==========================================================
# FIGURES
# ==========================================================

# 1. Threshold 탐색 그래프
plt.figure(figsize=(12, 6))
plt.plot(results_df["threshold"], results_df["s1r"],
         label="Stage1 Recall", color="#185FA5", marker="o", markersize=3)
plt.plot(results_df["threshold"], results_df["s23r"],
         label="Stage2+3 Recall", color="#E24B4A", marker="o", markersize=3)
plt.plot(results_df["threshold"], results_df["f1"],
         label="F1", color="#E67E22", marker="o", markersize=3)
plt.axvline(t_balanced, color="#185FA5", linestyle="--",
            label=f"균형 t={t_balanced}")
plt.axvline(t_s1focus, color="#27AE60", linestyle="--",
            label=f"S1중시 t={t_s1focus}")
plt.axvline(old_t, color="gray", linestyle=":",
            label=f"기존 t={old_t}")
plt.axhline(0.50, color="#185FA5", linestyle=":", alpha=0.4,
            label="S1 목표 0.50")
plt.axhline(0.75, color="#E24B4A", linestyle=":", alpha=0.4,
            label="S2+3 목표 0.75")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Threshold 탐색 — Stage1 vs Stage2+3 트레이드오프")
plt.legend(loc="center right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/threshold_tradeoff.png", dpi=300)
plt.close()

# 2. 균형 Threshold Confusion Matrix
vp_best = (valid_proba >= t_balanced).astype(int)
cm = confusion_matrix(y_vl, vp_best)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Stage1","Stage2+3"],
            yticklabels=["Stage1","Stage2+3"])
plt.title(f"Confusion Matrix (v14 threshold={t_balanced})")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=300)
plt.close()

cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(6, 5))
sns.heatmap(cm_pct, annot=True, fmt='.1%', cmap='Blues',
            xticklabels=["Stage1","Stage2+3"],
            yticklabels=["Stage1","Stage2+3"])
plt.title(f"Normalized CM (v14 threshold={t_balanced})")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix_percent.png", dpi=300)
plt.close()

# 3. ROC Curve
fpr, tpr, _ = roc_curve(y_vl, valid_proba)
auroc = roc_auc_score(y_vl, valid_proba)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color="#185FA5", label=f"AUROC={auroc:.4f}")
plt.plot([0,1],[0,1],"--",color="gray")
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC Curve (v14 threshold_adj)")
plt.legend(); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/roc_curve.png", dpi=300)
plt.close()

# 4. 비교 막대 그래프
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
names = [r["name"] for _, r in summary_df.iterrows()]
s1rs  = summary_df["s1r"].values
s23rs = summary_df["s23r"].values

x = np.arange(len(names))
w = 0.35
axes[0].bar(x - w/2, s1rs,  w, label="Stage1 Recall",   color="#185FA5")
axes[0].bar(x + w/2, s23rs, w, label="Stage2+3 Recall", color="#E24B4A")
axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=15, ha='right')
axes[0].set_ylim(0, 1.1); axes[0].set_ylabel("Recall")
axes[0].axhline(0.50, color="#185FA5", linestyle="--", alpha=0.5)
axes[0].axhline(0.75, color="#E24B4A", linestyle="--", alpha=0.5)
axes[0].legend(); axes[0].set_title("Valid Recall 비교")
for i, (s1, s23) in enumerate(zip(s1rs, s23rs)):
    axes[0].text(i-w/2, s1+0.02, f"{s1:.3f}", ha='center', fontsize=9)
    axes[0].text(i+w/2, s23+0.02, f"{s23:.3f}", ha='center', fontsize=9)

ts1rs  = summary_df["test_s1r"].values
ts23rs = summary_df["test_s23r"].values
axes[1].bar(x - w/2, ts1rs,  w, label="Stage1 Recall",   color="#185FA5")
axes[1].bar(x + w/2, ts23rs, w, label="Stage2+3 Recall", color="#E24B4A")
axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=15, ha='right')
axes[1].set_ylim(0, 1.1); axes[1].set_ylabel("Recall")
axes[1].legend(); axes[1].set_title("Test Recall 비교")
for i, (s1, s23) in enumerate(zip(ts1rs, ts23rs)):
    axes[1].text(i-w/2, s1+0.02, f"{s1:.3f}", ha='center', fontsize=9)
    axes[1].text(i+w/2, s23+0.02, f"{s23:.3f}", ha='center', fontsize=9)

plt.suptitle("Threshold 조정 비교 (v14)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/threshold_comparison.png", dpi=300)
plt.close()

print(f"\nSaved → {OUT_DIR}")

with open("models/stage2_LGBM_v14_threshold_adj.pkl", "wb") as f:
    pickle.dump({
        "model"            : model,
        "feature_cols"     : FEATURE_COLS,
        "threshold_balanced": t_balanced,
        "threshold_s1focus" : t_s1focus,
        "threshold_f1max"   : t_f1,
        "version"          : "v14_threshold_adj",
    }, f)

print("MODEL SAVED : models/stage2_LGBM_v14_threshold_adj.pkl")
print("FINISHED")