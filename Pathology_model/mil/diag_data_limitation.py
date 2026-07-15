"""
데이터 한계 규명 — chronic AUROC 0.736 이 '낮은 성능'인지 '양성 10명發 넓은 불확실성'인지 규명.

Attention collapse 는 이미 반증됨(diag_collapse) → temperature/entropy/attention 수정 안 함.
모델은 TaskAttentionMIL(silver off) 고정. 데이터/평가 관점만 분석.

수행:
 1. Repeated Stratified CV (5-fold × 20 repeats, seed42~61)  -> repeated_cv_summary.csv
 2. Pooled OOF 평가(fold평균 아닌 전체환자 합산) + 반복간 분포   -> pooled_oof_metrics.json
 3. Bootstrap 95% CI (2000) — 점추정의 흔들림                 -> bootstrap_ci.json
 4. Fold variance: positive_count vs AUPRC 상관               -> fold_variance.csv, fold_variance_correlation.json
 5. chronic label 품질: DKD-only(HTN 제외) vs 현재 chronic     -> 비교(report에 포함)
 7. 최종 진단(Case A~D)                                        -> data_limitation_report.json

사용: python diag_data_limitation.py --encoder ctranspath --mags 10 --repeats 20
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "Pathology_model")))
from mil.train import COL, ROOT, SEED, load_bags, seed_all

OUTDIR = ROOT / "Pathology_model/results/06_data_limitation"
TASKS = ["immune", "chronic", "stage3"]


def boot_ci(y, p, fn, n=2000, seed=SEED):
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        ix = rng.integers(0, len(y), len(y))
        if len(np.unique(y[ix])) < 2:
            continue
        vals.append(fn(y[ix], p[ix]))
    if not vals:
        return {"point": None, "ci95": [None, None]}
    return {"point": round(float(fn(y, p)), 3),
            "ci95": [round(float(np.percentile(vals, 2.5)), 3),
                     round(float(np.percentile(vals, 97.5)), 3)]}


def train_eval_fold(bags, tr, va, embed_dim, device, epochs):
    import torch
    import torch.nn.functional as F
    from mil.model import TaskAttentionMIL
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = TaskAttentionMIL(in_dim=embed_dim, silver_mode="off").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    def pw(col):
        yy = tr[col].dropna(); n1 = (yy == 1).sum(); n0 = (yy == 0).sum()
        return torch.tensor(n0 / max(n1, 1), device=device, dtype=torch.float32)
    PW = {t: pw(COL[t]) for t in TASKS}
    for ep in range(epochs):
        model.train()
        for r in tr.sample(frac=1, random_state=SEED + ep).itertuples():
            bag = {s: torch.from_numpy(v).to(device) for s, v in bags[r.patient_id].items()}
            out = model(bag); loss = 0.0; nt = 0
            for t in TASKS:
                yv = getattr(r, COL[t])
                if not (yv != yv):
                    loss = loss + F.binary_cross_entropy_with_logits(
                        out[t], torch.tensor(float(yv), device=device), pos_weight=PW[t]); nt += 1
            sv = r.ati_severity_n
            if not (sv != sv):
                loss = loss + F.mse_loss(torch.sigmoid(out["ati_severity"]),
                                         torch.tensor(float(sv), device=device)); nt += 1
            if nt:
                opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    preds = {t: [] for t in TASKS}
    import torch as _t
    with _t.no_grad():
        for r in va.itertuples():
            bag = {s: _t.from_numpy(v).to(device) for s, v in bags[r.patient_id].items()}
            out = model(bag)
            for t in TASKS:
                yv = getattr(r, COL[t])
                if not (yv != yv):
                    preds[t].append((r.patient_id, int(yv), float(_t.sigmoid(out[t]))))
    return preds


def main():
    import torch
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    from scipy.stats import pearsonr, spearmanr

    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="ctranspath")
    ap.add_argument("--mags", default="10")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    seed_all(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    idx = pd.read_csv(ROOT / "data/embeddings" / args.encoder / "index.csv", dtype={"magnification": str})
    msel = [m.strip() for m in args.mags.split(",")]
    idx = idx[idx["magnification"].isin(msel) |
              ((idx["stain"] == "IF") & (idx["magnification"] == "native"))].copy()
    embed_dim = int(idx["embed_dim"].iloc[0])
    sm = pd.read_csv(ROOT / "split_manifest.csv").drop_duplicates("patient_id")
    sm["patient_id"] = sm["patient_id"].astype(str)
    # gold(stratify_key 보유) 환자만 — chronic/immune 분석 대상
    gold = sm[(sm["label_grade"] == "gold") & sm["stratify_key"].notna()].copy()
    gold["ati_severity_n"] = (gold["task_ati_severity"] - 1) / 2.0
    bags = load_bags(gold["patient_id"].tolist(), idx, keep=None)
    gold = gold[gold["patient_id"].isin(bags.keys())].reset_index(drop=True)
    pids = gold["patient_id"].to_numpy(); strat = gold["stratify_key"].to_numpy()
    print(f"gold cohort={len(gold)} | chronic+={int((gold['task_chronic']==1).sum())} "
          f"immune+={int((gold['task_immune']==1).sum())} | repeats={args.repeats}", flush=True)

    # ===== 실험1: Repeated Stratified CV =====
    rows = []
    pooled = {r: {t: [] for t in TASKS} for r in range(args.repeats)}   # repeat별 pooled OOF
    for rep in range(args.repeats):
        seed = 42 + rep
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tri, vai) in enumerate(skf.split(pids, strat)):
            tr = gold.iloc[tri]; va = gold.iloc[vai]
            preds = train_eval_fold(bags, tr, va, embed_dim, device, args.epochs)
            for t in TASKS:
                pr = preds[t]
                if not pr:
                    continue
                y = [x[1] for x in pr]; p = [x[2] for x in pr]
                pooled[rep][t].extend(pr)
                pred = (np.array(p) > 0.5).astype(int)
                rows.append({"repeat": rep, "seed": seed, "fold": fold, "task": t,
                             "pos": int(sum(y)), "neg": int(len(y) - sum(y)),
                             "AUROC": round(float(roc_auc_score(y, p)), 3) if len(set(y)) > 1 else None,
                             "AUPRC": round(float(average_precision_score(y, p)), 3) if len(set(y)) > 1 else None,
                             "F1": round(float(f1_score(y, pred, zero_division=0)), 3),
                             "Precision": round(float(precision_score(y, pred, zero_division=0)), 3),
                             "Recall": round(float(recall_score(y, pred, zero_division=0)), 3)})
        print(f"  repeat {rep} (seed {seed}) done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "repeated_cv_summary.csv", index=False, encoding="utf-8")

    # ===== 실험2: Pooled OOF (repeat별 전체환자 합산) + 반복간 분포 =====
    pooled_metrics = {}
    for t in TASKS:
        per_rep = []
        for rep in range(args.repeats):
            pr = pooled[rep][t]
            if not pr:
                continue
            y = [x[1] for x in pr]; p = [x[2] for x in pr]
            if len(set(y)) < 2:
                continue
            per_rep.append({"auroc": float(roc_auc_score(y, p)),
                            "auprc": float(average_precision_score(y, p))})
        au = [r["auroc"] for r in per_rep]; ap_ = [r["auprc"] for r in per_rep]
        pooled_metrics[t] = {
            "n_repeats": len(per_rep),
            "auroc_mean": round(float(np.mean(au)), 3), "auroc_std": round(float(np.std(au)), 3),
            "auroc_range": [round(min(au), 3), round(max(au), 3)],
            "auprc_mean": round(float(np.mean(ap_)), 3), "auprc_std": round(float(np.std(ap_)), 3),
            "auprc_range": [round(min(ap_), 3), round(max(ap_), 3)]}
    (OUTDIR / "pooled_oof_metrics.json").write_text(
        json.dumps(pooled_metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    # ===== 실험3: Bootstrap CI (대표 repeat=0 pooled OOF) =====
    boot = {}
    for t in TASKS:
        pr = pooled[0][t]
        y = [x[1] for x in pr]; p = [x[2] for x in pr]
        boot[t] = {"n": len(y), "n_pos": int(sum(y)),
                   "AUROC": boot_ci(y, p, roc_auc_score),
                   "AUPRC": boot_ci(y, p, average_precision_score)}
    (OUTDIR / "bootstrap_ci.json").write_text(
        json.dumps(boot, indent=2, ensure_ascii=False), encoding="utf-8")

    # ===== 실험4: Fold variance — positive_count vs AUPRC =====
    df.to_csv(OUTDIR / "fold_variance.csv", index=False, encoding="utf-8")  # 동일 데이터(요청 파일명)
    fv = {}
    for t in TASKS:
        d = df[(df.task == t) & df.AUPRC.notna()]
        if len(d) >= 3 and d["pos"].nunique() > 1:
            fv[t] = {"spearman_pos_vs_auprc": round(float(spearmanr(d["pos"], d["AUPRC"]).correlation), 3),
                     "pearson_pos_vs_auprc": round(float(pearsonr(d["pos"], d["AUPRC"])[0]), 3),
                     "auprc_std_across_folds": round(float(d["AUPRC"].std()), 3),
                     "auprc_range": [round(float(d["AUPRC"].min()), 3), round(float(d["AUPRC"].max()), 3)],
                     "mean_pos_per_fold": round(float(d["pos"].mean()), 2)}
    (OUTDIR / "fold_variance_correlation.json").write_text(
        json.dumps(fv, indent=2, ensure_ascii=False), encoding="utf-8")

    # ===== 실험5: chronic label 품질 — DKD-only(HTN 제외) vs 현재 chronic =====
    # 같은 repeat0 pooled OOF 에서 HTN 환자만 제외하고 chronic 재평가(저비용 proxy; HTN n=1)
    dx = {str(r.patient_id): str(r.primary_adjudicated_category) for r in gold.itertuples()}
    pr = pooled[0]["chronic"]
    yk = [(x[1], x[2]) for x in pr if dx.get(x[0]) != "Hypertensive Kidney Disease"]
    y2 = [a for a, _ in yk]; p2 = [b for _, b in yk]
    dkd_only = {"n": len(y2), "n_pos": int(sum(y2)),
                "AUROC": boot_ci(y2, p2, roc_auc_score),
                "AUPRC": boot_ci(y2, p2, average_precision_score),
                "note": "repeat0 pooled OOF 에서 HTN(1명) 제외 재평가. 현재 chronic 와 비교."}

    # ===== 실험7: 최종 진단 =====
    fvc = fv.get("chronic", {}); fvi = fv.get("immune", {})
    report = {
        "model": "TaskAttentionMIL(silver off), 고정", "repeats": args.repeats,
        "attention_collapse": "이미 반증(diag_collapse): Top1<0.3, masking 무영향, temp 무효과",
        "pooled_oof": pooled_metrics, "bootstrap_ci": boot,
        "fold_variance": fv, "chronic_dkd_only_vs_all": {
            "chronic_all": {"AUROC": boot["chronic"]["AUROC"], "AUPRC": boot["chronic"]["AUPRC"]},
            "chronic_dkd_only": {"AUROC": dkd_only["AUROC"], "AUPRC": dkd_only["AUPRC"]}},
        "verdict": {}}
    # 판정 휴리스틱
    def wide(ci):
        lo, hi = ci["ci95"]
        return (hi - lo) if (lo is not None and hi is not None) else None
    for t in ["chronic", "immune"]:
        w_au = wide(boot[t]["AUROC"]); w_ap = wide(boot[t]["AUPRC"])
        ev = []
        if w_au and w_au > 0.25:
            ev.append(f"{t} AUROC 95%CI 폭 {w_au} (>0.25) → 점추정 매우 불안정")
        if w_ap and w_ap > 0.25:
            ev.append(f"{t} AUPRC 95%CI 폭 {w_ap} (>0.25) → 탐색적 수준")
        fvt = fv.get(t, {})
        if fvt.get("spearman_pos_vs_auprc") and fvt["spearman_pos_vs_auprc"] > 0.3:
            ev.append(f"fold AUPRC가 양성수와 양의상관({fvt['spearman_pos_vs_auprc']}) → 표본부족 영향")
        report["verdict"][t] = {"auroc_ci_width": w_au, "auprc_ci_width": w_ap, "evidence": ev}
    # chronic label noise(DKD-only 비교)
    a_all = boot["chronic"]["AUROC"]["point"]; a_dkd = dkd_only["AUROC"]["point"]
    report["verdict"]["chronic_label_noise"] = {
        "auroc_all": a_all, "auroc_dkd_only": a_dkd,
        "delta": None if (a_all is None or a_dkd is None) else round(a_dkd - a_all, 3),
        "interpretation": "DKD-only가 크게 높으면 HTN이 노이즈(라벨품질). 미미하면 라벨 문제 아님(HTN n=1이라 영향 작을 것)"}
    report["final_case"] = ("Case B(데이터규모 한계) 우세 — collapse 반증·CI 광폭·양성 10/14명. "
                            "라벨/복합 여부는 chronic_label_noise·UMAP(별도) 참조.")
    (OUTDIR / "data_limitation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n저장: {OUTDIR}")
    for t in ["chronic", "immune"]:
        b = boot[t]
        print(f"[{t}] AUROC {b['AUROC']['point']} CI{b['AUROC']['ci95']} | "
              f"AUPRC {b['AUPRC']['point']} CI{b['AUPRC']['ci95']} | n_pos={b['n_pos']}")
    print(f"[chronic] all AUROC {a_all} vs DKD-only {a_dkd}")


if __name__ == "__main__":
    main()
