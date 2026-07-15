# ==========================================================
# 09_modeling_LGBM_v14_dual_threshold.py
# Stage1 vs Stage2+3 | Full 35개
# Method: 이중 Threshold
#         P < t_low  → Stage1 확정
#         P > t_high → Stage2+3 확정
#         중간 구간  → "경계 환자" 별도 처리
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
    f1_score, recall_score, classification_report,
    confusion_matrix, roc_auc_score,
    average_precision_score, roc_curve
)

OUT_DIR = "outputs/LGBM_v14_dual_threshold"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("LGBM v14 | Dual Threshold | Stage1 vs Stage2+3")
print("=" * 70)

# v13_full_classweight 재활용
with open("models/stage2_LGBM_v13_full_classweight.pkl", "rb") as f:
    v13 = pickle.load(f)

model        = v13["model"]
FEATURE_COLS = v13["feature_cols"]

X_valid_orig  = np.load("data/X_valid.npy")
X_test_orig   = np.load("data/X_test.npy")
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
# 이중 Threshold 탐색
# ==========================================================

print("=" * 70)
print("이중 Threshold 탐색")
print("=" * 70)

dual_results = []

for t_low in np.arange(0.20, 0.55, 0.05):
    for t_high in np.arange(t_low+0.10, 0.91, 0.05):

        # 확정 예측 (중간 구간 제외)
        certain_mask = (valid_proba < t_low) | (valid_proba >= t_high)
        uncertain_mask = ~certain_mask

        pred_certain = np.where(valid_proba < t_low, 0, 1)
        n_uncertain  = uncertain_mask.sum()
        n_certain    = certain_mask.sum()

        if n_certain == 0:
            continue

        # 확정 케이스만 평가
        y_certain  = y_vl[certain_mask]
        p_certain  = pred_certain[certain_mask]

        if len(np.unique(y_certain)) < 2:
            continue

        tn,fp,fn,tp_ = confusion_matrix(y_certain, p_certain).ravel()
        s1r  = tn/(tn+fp) if (tn+fp)>0 else 0
        s23r = tp_/(tp_+fn) if (tp_+fn)>0 else 0
        f1   = f1_score(y_certain, p_certain, zero_division=0)

        # 불확실 비율
        uncertain_ratio = n_uncertain / len(y_vl)

        dual_results.append([
            round(t_low,2), round(t_high,2),
            s1r, s23r, f1,
            n_uncertain, uncertain_ratio
        ])

dual_df = pd.DataFrame(dual_results,
    columns=["t_low","t_high","s1r","s23r","f1",
             "n_uncertain","uncertain_ratio"])
dual_df.to_csv(f"{OUT_DIR}/dual_threshold_search.csv", index=False)

# 최적 조합: S1 ≥ 0.50, S2+3 ≥ 0.75, 불확실 최소
cond = dual_df[(dual_df["s1r"] >= 0.50) & (dual_df["s23r"] >= 0.75)]
if len(cond) > 0:
    best_row = cond.loc[cond["uncertain_ratio"].idxmin()]
    print(f"[목표 달성] S1≥0.50, S2+3≥0.75 조합 발견")
else:
    cond2 = dual_df[dual_df["s23r"] >= 0.75]
    if len(cond2) > 0:
        best_row = cond2.loc[cond2["s1r"].idxmax()]
        print(f"[부분 달성] S2+3≥0.75 + S1 최대")
    else:
        best_row = dual_df.loc[dual_df["s1r"].idxmax()]
        print(f"[S1 최대] 조건 완화")

best_t_low  = best_row["t_low"]
best_t_high = best_row["t_high"]

print(f"Best t_low    : {best_t_low}")
print(f"Best t_high   : {best_t_high}")
print(f"S1 Recall     : {best_row['s1r']:.4f}")
print(f"S2+3 Recall   : {best_row['s23r']:.4f}")
print(f"불확실 환자    : {best_row['n_uncertain']}명 ({best_row['uncertain_ratio']*100:.1f}%)")
print()

# ==========================================================
# 최적 Threshold 적용
# ==========================================================

def dual_predict(proba, t_low, t_high, uncertain_as=1):
    pred = np.where(proba < t_low, 0, 1)
    uncertain = (proba >= t_low) & (proba < t_high)
    pred[uncertain] = uncertain_as
    return pred, uncertain

# 경계 환자를 Stage2+3으로 (보수적)
valid_pred_cons, valid_unc = dual_predict(valid_proba, best_t_low, best_t_high, uncertain_as=1)
test_pred_cons,  test_unc  = dual_predict(test_proba,  best_t_low, best_t_high, uncertain_as=1)

# 경계 환자를 Stage1으로 (관대)
valid_pred_lib, _ = dual_predict(valid_proba, best_t_low, best_t_high, uncertain_as=0)
test_pred_lib, _  = dual_predict(test_proba,  best_t_low, best_t_high, uncertain_as=0)

print("=" * 70)
print(f"VALIDATION RESULT (t_low={best_t_low}, t_high={best_t_high})")
print("=" * 70)

for pred_name, vp, tp_ in [
    ("보수적 (경계→Stage2+3)", valid_pred_cons, test_pred_cons),
    ("관대  (경계→Stage1   )", valid_pred_lib,  test_pred_lib),
]:
    tn,fp,fn,tpp = confusion_matrix(y_vl, vp).ravel()
    s1r  = tn/(tn+fp); s23r = tpp/(tpp+fn)
    f1   = f1_score(y_vl, vp, zero_division=0)
    tn_t,fp_t,fn_t,tp_t = confusion_matrix(y_te, tp_).ravel()
    ts1r  = tn_t/(tn_t+fp_t); ts23r = tp_t/(tp_t+fn_t)
    print(f"\n[{pred_name}]")
    print(f"Valid: S1={s1r:.4f} | S2+3={s23r:.4f} | F1={f1:.4f}")
    print(f"Test:  S1={ts1r:.4f} | S2+3={ts23r:.4f}")
    print("Stage별 세부:")
    for sv,sn in [(1,"Stage1"),(2,"Stage2"),(3,"Stage3")]:
        mask = y_stage_vl == sv
        if mask.sum() > 0:
            c = (vp[mask]==0).sum() if sv==1 else (vp[mask]==1).sum()
            print(f"  {sn}: {c}/{mask.sum()} ({c/mask.sum()*100:.1f}%)")

print()
print(f"불확실 환자: {valid_unc.sum()}명 ({valid_unc.sum()/len(y_vl)*100:.1f}%)")
print(f"  → 실제로는 Stage1: {y_vl[valid_unc].sum()==0} / Stage2+3: {y_vl[valid_unc].sum()}")
print(f"  Stage1 실제: {(y_vl[valid_unc]==0).sum()}명 / Stage2+3 실제: {(y_vl[valid_unc]==1).sum()}명")

# Probability Distribution 시각화
plt.figure(figsize=(10, 6))
plt.hist(valid_proba[y_vl==0], bins=50, alpha=0.5,
         label="Stage1 실제", color="#185FA5")
plt.hist(valid_proba[y_vl==1], bins=50, alpha=0.5,
         label="Stage2+3 실제", color="#E24B4A")
plt.axvline(best_t_low, color="#27AE60", linestyle="--", linewidth=2,
            label=f"t_low={best_t_low}")
plt.axvline(best_t_high, color="#E67E22", linestyle="--", linewidth=2,
            label=f"t_high={best_t_high}")
plt.axvspan(best_t_low, best_t_high, alpha=0.1, color="gray",
            label=f"불확실 구간 ({valid_unc.sum()}명)")
plt.xlabel("P(Stage2+3)"); plt.ylabel("Count")
plt.title("확률 분포 + 이중 Threshold")
plt.legend(); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/probability_distribution.png", dpi=300)
plt.close()

# 이중 Threshold 탐색 결과 시각화
plt.figure(figsize=(10, 6))
pivot = dual_df.pivot_table(
    values="s1r", index="t_low", columns="t_high"
)
sns.heatmap(pivot, annot=True, fmt='.2f', cmap='Blues')
plt.title("S1 Recall by (t_low, t_high)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/dual_threshold_heatmap_s1.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
pivot2 = dual_df.pivot_table(
    values="s23r", index="t_low", columns="t_high"
)
sns.heatmap(pivot2, annot=True, fmt='.2f', cmap='Reds')
plt.title("S2+3 Recall by (t_low, t_high)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/dual_threshold_heatmap_s23.png", dpi=300)
plt.close()

print(f"\nSaved → {OUT_DIR}")

with open("models/stage2_LGBM_v14_dual_threshold.pkl","wb") as f:
    pickle.dump({"model":model,"feature_cols":FEATURE_COLS,
                 "t_low":best_t_low,"t_high":best_t_high,
                 "version":"v14_dual_threshold"},f)
print("MODEL SAVED : models/stage2_LGBM_v14_dual_threshold.pkl")
print("FINISHED")