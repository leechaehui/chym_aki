"""
Descriptor 예측 — 5-head Task-Specific Attention MIL (최종설계 핵심 빌드)

AKI-only 83명, KPMP TIV Descriptor Score 라벨로 병리지표 정량예측:
  immune(BCE, AIN1/ATI0) · tubulitis(BCE) · wbc_pct(MSE) · fibrosis_pct(MSE) · atrophy(MSE)
질환명 분류가 아니라 '조직학적 상태' 예측. 각 task 독립 attention. 마스킹 멀티태스크
(각 손실은 라벨 있는 환자만). 환자단위 5-fold stratified, pooled-OOF + bootstrap 95%CI.

stain=HE/PAS/MT/Silver(IF 제외), 10x. silver_mode=prediction(Silver를 예측 stain으로 포함).
사용: python train_descriptor.py --encoder ctranspath --mags 10
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "Pathology_model")))
from mil.train import ROOT, SEED, load_bags, seed_all

OUT = ROOT / "Pathology_model/results/08_descriptor_prediction"
CLF = ["immune", "tubulitis"]
REG = ["wbc_pct", "fibrosis_pct", "atrophy"]
TASKS = CLF + REG
STAIN_KEEP = {"HE", "PAS", "MT", "SILVER"}     # IF 제외
REG_SCALE = {"wbc_pct": 100.0, "fibrosis_pct": 100.0, "atrophy": 100.0}  # [0,1] 정규화


def boot_ci(y, p, fn, n=2000, seed=SEED, need2cls=False):
    y, p = np.asarray(y, float), np.asarray(p, float)
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        ix = rng.integers(0, len(y), len(y))
        if need2cls and len(np.unique(y[ix])) < 2:
            continue
        try:
            vals.append(fn(y[ix], p[ix]))
        except Exception:
            pass
    pt = round(float(fn(y, p)), 3) if (not need2cls or len(np.unique(y)) > 1) else None
    if not vals:
        return {"point": pt, "ci95": [None, None]}
    return {"point": pt, "ci95": [round(float(np.percentile(vals, 2.5)), 3),
                                  round(float(np.percentile(vals, 97.5)), 3)]}


def main():
    import torch
    import torch.nn.functional as F
    from mil.model import TaskAttentionMIL
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                                 precision_score, recall_score, balanced_accuracy_score,
                                 confusion_matrix, mean_absolute_error, mean_squared_error,
                                 r2_score)
    from scipy.stats import pearsonr, spearmanr

    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="ctranspath")
    ap.add_argument("--mags", default="10")
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    seed_all(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 임베딩 (HE/PAS/MT/Silver, 10x)
    idx = pd.read_csv(ROOT / "data/embeddings" / args.encoder / "index.csv", dtype={"magnification": str})
    msel = [m.strip() for m in args.mags.split(",")]
    idx = idx[idx["magnification"].isin(msel) & idx["stain"].isin(STAIN_KEEP)].copy()
    embed_dim = int(idx["embed_dim"].iloc[0])

    # 라벨 결합: split_manifest(immune) + descriptor_labels(나머지)
    sm = pd.read_csv(ROOT / "Pathology_model/artifacts/split_manifest.csv").drop_duplicates("patient_id")
    sm["patient_id"] = sm["patient_id"].astype(str)
    desc = pd.read_csv(ROOT / "Pathology_model/artifacts/descriptor_labels.csv")
    desc["patient_id"] = desc["patient_id"].astype(str)
    df = sm[["patient_id", "task_immune"]].merge(desc, on="patient_id", how="outer")

    def col(name, src):
        v = pd.to_numeric(df[src], errors="coerce")
        v = v.where(v < 999)                      # 999=결측
        return v
    lab = pd.DataFrame({"patient_id": df["patient_id"]})
    lab["immune"] = pd.to_numeric(df["task_immune"], errors="coerce")
    lab["tubulitis"] = col("tubulitis", "tubulitis")
    lab["wbc_pct"] = col("wbc", "interstitial_mononuclear_wbc_pct")
    lab["fibrosis_pct"] = col("fibrosis", "interstitial_fibrosis_pct")
    lab["atrophy"] = col("atrophy", "tubular_atrophy")

    bags = load_bags(lab["patient_id"].tolist(), idx, keep=STAIN_KEEP)
    lab = lab[lab["patient_id"].isin(bags.keys())].reset_index(drop=True)
    # 코호트: 어떤 task든 라벨 1개 이상 보유
    has_any = lab[TASKS].notna().any(axis=1)
    lab = lab[has_any].reset_index(drop=True)
    n_lab = {t: int(lab[t].notna().sum()) for t in TASKS}
    print(f"cohort={len(lab)} | labels={n_lab} | stains=HE/PAS/MT/Silver | device={device}", flush=True)

    # 층화 키: immune(AIN/ATI/none) — RQ1 균형 보장
    strat = lab["immune"].map({1: "AIN", 0: "ATI"}).fillna("none").to_numpy()
    pids = lab["patient_id"].to_numpy()

    oof = {t: {"y": [], "p": [], "pid": []} for t in TASKS}
    sc_sum = {t: {} for t in TASKS}; sc_cnt = {t: {} for t in TASKS}
    ent_sum = {t: 0.0 for t in TASKS}; ent_cnt = {t: 0 for t in TASKS}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for fold, (tri, vai) in enumerate(skf.split(pids, strat)):
        tr = lab.iloc[tri]; va = lab.iloc[vai]
        torch.manual_seed(SEED); np.random.seed(SEED)
        model = TaskAttentionMIL(in_dim=embed_dim, silver_mode="prediction", tasks=TASKS).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

        def pw(t):
            yy = tr[t].dropna(); n1 = (yy == 1).sum(); n0 = (yy == 0).sum()
            return torch.tensor(n0 / max(n1, 1), device=device, dtype=torch.float32)
        PW = {t: pw(t) for t in CLF}

        for ep in range(args.epochs):
            model.train()
            for r in tr.sample(frac=1, random_state=SEED + ep).itertuples(index=False):
                pid = r.patient_id
                bag = {s: torch.from_numpy(v).to(device) for s, v in bags[pid].items()}
                out = model(bag); loss = 0.0; nt = 0
                rd = dict(zip(lab.columns, r))
                for t in CLF:
                    yv = rd[t]
                    if yv == yv:
                        loss = loss + F.binary_cross_entropy_with_logits(
                            out[t], torch.tensor(float(yv), device=device), pos_weight=PW[t]); nt += 1
                for t in REG:
                    yv = rd[t]
                    if yv == yv:
                        tgt = float(yv) / REG_SCALE[t]
                        loss = loss + F.mse_loss(torch.sigmoid(out[t]),
                                                 torch.tensor(tgt, device=device)); nt += 1
                if nt:
                    opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            for r in va.itertuples(index=False):
                rd = dict(zip(lab.columns, r)); pid = rd["patient_id"]
                bag = {s: torch.from_numpy(v).to(device) for s, v in bags[pid].items()}
                out = model(bag)
                for t in TASKS:
                    yv = rd[t]
                    if yv != yv:
                        continue
                    pr = float(torch.sigmoid(out[t]))
                    oof[t]["y"].append(float(yv) if t in CLF else float(yv))
                    oof[t]["p"].append(pr if t in CLF else pr * REG_SCALE[t])
                    oof[t]["pid"].append(pid)
                    for s, w in out["task_stain_contrib_norm"][t].items():
                        sc_sum[t][s] = sc_sum[t].get(s, 0.0) + w; sc_cnt[t][s] = sc_cnt[t].get(s, 0) + 1
                    e = out["task_entropy"].get(t)
                    if e is not None:
                        ent_sum[t] += float(e); ent_cnt[t] += 1
        print(f"  fold {fold} done", flush=True)

    # ===== 평가 =====
    report = {"config": {"encoder": args.encoder, "mags": args.mags, "stains": "HE/PAS/MT/Silver",
                         "model": "TaskAttentionMIL-5head", "silver_mode": "prediction",
                         "seed": SEED, "epochs": args.epochs, "cohort": int(len(lab)), "labels": n_lab},
              "tasks": {}}
    for t in CLF:
        y = np.array(oof[t]["y"]); p = np.array(oof[t]["p"]); pred = (p > 0.5).astype(int)
        tn, fp, fn_, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) else None
        npv = tn / (tn + fn_) if (tn + fn_) else None
        report["tasks"][t] = {"type": "clf", "n": len(y), "n_pos": int(y.sum()),
            "AUROC": boot_ci(y, p, roc_auc_score, need2cls=True),
            "AUPRC": boot_ci(y, p, average_precision_score, need2cls=True),
            "precision": round(float(precision_score(y, pred, zero_division=0)), 3),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 3),
            "f1": round(float(f1_score(y, pred, zero_division=0)), 3),
            "specificity": round(float(spec), 3) if spec is not None else None,
            "npv": round(float(npv), 3) if npv is not None else None,
            "balanced_acc": round(float(balanced_accuracy_score(y, pred)), 3),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn_), "tp": int(tp)}}
    for t in REG:
        y = np.array(oof[t]["y"]); p = np.array(oof[t]["p"])
        report["tasks"][t] = {"type": "reg", "n": len(y),
            "MAE": round(float(mean_absolute_error(y, p)), 2),
            "RMSE": round(float(np.sqrt(mean_squared_error(y, p))), 2),
            "R2": round(float(r2_score(y, p)), 3) if len(y) > 2 else None,
            "pearson": round(float(pearsonr(y, p)[0]), 3) if np.std(p) > 0 else None,
            "spearman_ci": boot_ci(y, p, lambda a, b: spearmanr(a, b).correlation),
            "y_mean": round(float(y.mean()), 1), "y_range": [float(y.min()), float(y.max())]}
    # stain contribution(normalized) + entropy
    report["task_stain_contribution_norm"] = {
        t: {s: round(sc_sum[t][s] / sc_cnt[t][s], 5) for s in sc_sum[t]} for t in TASKS if sc_sum[t]}
    report["task_attn_entropy"] = {t: round(ent_sum[t] / ent_cnt[t], 3) for t in TASKS if ent_cnt[t]}

    (OUT / "descriptor_metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = [{"task": t, "patient_id": oof[t]["pid"][i], "y": oof[t]["y"][i], "p": oof[t]["p"][i]}
            for t in TASKS for i in range(len(oof[t]["y"]))]
    pd.DataFrame(rows).to_csv(OUT / "descriptor_oof.csv", index=False, encoding="utf-8")

    print("\n=== Descriptor 5-head (pooled-OOF + bootstrap 95%CI) ===")
    for t in CLF:
        m = report["tasks"][t]
        print(f"[{t}] AUROC {m['AUROC']['point']} CI{m['AUROC']['ci95']} | AUPRC {m['AUPRC']['point']} "
              f"CI{m['AUPRC']['ci95']} | P {m['precision']} R {m['recall']} F1 {m['f1']} | n{m['n']}({m['n_pos']})")
    for t in REG:
        m = report["tasks"][t]
        print(f"[{t}] MAE {m['MAE']} RMSE {m['RMSE']} R2 {m['R2']} | Spearman {m['spearman_ci']['point']} "
              f"CI{m['spearman_ci']['ci95']} | n{m['n']} (y~{m['y_mean']})")
    print(f"\nstain contrib(norm): {report['task_stain_contribution_norm']}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
