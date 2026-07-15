"""
Router 개선 결정실험 — EXP-1(CORAL margin) vs EXP-2(5-seed ensemble) vs baseline(MC-dropout)

질문: uncertainty score를 바꾸면 risk-coverage가 monotonic(=selection 작동)해지는가?
 - EXP-1 margin   : unc = mean_k(1-|2σ(logit_k)-1|)  (CORAL 누적확률 결정경계 근접도). attention entropy 미사용.
 - EXP-2 ensemble : unc = task별 등급의 seed(42..46) std 평균  (deep ensemble 불일치).
판정: coverage↓일 때 error 단조 감소 → score 문제(C), 해결가능. 둘 다 평탄 → 데이터 한계(D).

출력: results/11_router_calibration/exp_margin_ensemble.json (+png)
사용: python router_exp.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "Pathology_model")))
from mil.train import ROOT, SEED, load_bags, seed_all

OUT = ROOT / "Pathology_model/results/11_router_calibration"
ORD = ["fibrosis", "atrophy", "inflammation"]
KBINS = 4
BANFF = {"fibrosis": [5, 25, 50], "atrophy": [5, 25, 50], "inflammation": [10, 25, 50]}
STAIN_KEEP = {"HE", "PAS", "MT", "SILVER"}
SEEDS = [42, 43, 44, 45, 46]


def to_ord(v, e):
    return np.nan if v != v else float(sum(v > x for x in e))


def main():
    import torch
    import torch.nn.functional as F
    from mil.model import TaskAttentionMIL
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import cohen_kappa_score
    import matplotlib
    matplotlib.use("Agg"); matplotlib.rcParams["font.family"] = "Malgun Gothic"
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    seed_all(SEED); device = "cuda" if torch.cuda.is_available() else "cpu"
    idx = pd.read_csv(ROOT / "data/embeddings/ctranspath/index.csv", dtype={"magnification": str})
    idx = idx[idx["magnification"].isin(["10", "40"]) & idx["stain"].isin(STAIN_KEEP)]
    embed_dim = int(idx["embed_dim"].iloc[0])
    sm = pd.read_csv(ROOT / "Pathology_model/artifacts/split_manifest.csv").drop_duplicates("patient_id")
    sm["patient_id"] = sm["patient_id"].astype(str)
    desc = pd.read_csv(ROOT / "Pathology_model/artifacts/descriptor_labels.csv")
    desc["patient_id"] = desc["patient_id"].astype(str)
    src = {"fibrosis": "interstitial_fibrosis_pct", "atrophy": "tubular_atrophy",
           "inflammation": "interstitial_mononuclear_wbc_pct"}
    lab = pd.DataFrame({"patient_id": desc["patient_id"]}).merge(
        sm[["patient_id", "task_immune"]], on="patient_id", how="left")
    for t, c in src.items():
        v = pd.to_numeric(desc[c], errors="coerce").where(lambda x: x < 999)
        lab[t] = v.map(lambda x: to_ord(x, BANFF[t]))
    bags = load_bags(lab["patient_id"].tolist(), idx, keep=STAIN_KEEP)
    lab = lab[lab["patient_id"].isin(bags) & lab[ORD].notna().any(axis=1)].reset_index(drop=True)
    strat = lab["task_immune"].map({1: "AIN", 0: "ATI"}).fillna("none").to_numpy()
    pids = lab["patient_id"].to_numpy()
    out_dims = {t: KBINS - 1 for t in ORD}

    def train_one(tr, seed):
        torch.manual_seed(seed); np.random.seed(seed)
        m = TaskAttentionMIL(in_dim=embed_dim, silver_mode="prediction", tasks=ORD, out_dims=out_dims).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=1e-4, weight_decay=1e-4)
        for ep in range(40):
            m.train()
            for r in tr.sample(frac=1, random_state=seed + ep).itertuples(index=False):
                rd = dict(zip(lab.columns, r))
                bag = {s: torch.from_numpy(v).to(device) for s, v in bags[rd["patient_id"]].items()}
                out = m(bag); loss = 0.0; nt = 0
                for t in ORD:
                    yv = rd[t]
                    if yv == yv:
                        lv = torch.tensor([1.0 if yv > k else 0.0 for k in range(KBINS - 1)], device=device)
                        loss = loss + F.binary_cross_entropy_with_logits(out[t], lv); nt += 1
                if nt:
                    opt.zero_grad(); loss.backward(); opt.step()
        return m

    rec = []
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for fold, (tri, vai) in enumerate(skf.split(pids, strat)):
        tr = lab.iloc[tri]; va = lab.iloc[vai]
        models = [train_one(tr, s) for s in SEEDS]
        for m in models:
            m.eval()
        with torch.no_grad():
            for r in va.itertuples(index=False):
                rd = dict(zip(lab.columns, r))
                bag = {s: torch.from_numpy(v).to(device) for s, v in bags[rd["patient_id"]].items()}
                row = {"patient_id": rd["patient_id"]}
                margins, ens_stds, errs = [], [], []
                for t in ORD:
                    if rd[t] != rd[t]:
                        continue
                    # 각 seed: 등급 + logits
                    grades, logit0 = [], None
                    for i, m in enumerate(models):
                        lo = m(bag)[t]
                        grades.append(float((torch.sigmoid(lo) > 0.5).sum()))
                        if i == 0:
                            logit0 = torch.sigmoid(lo).cpu().numpy()  # seed0 누적확률
                    ens_mean = float(np.mean(grades)); ens_std = float(np.std(grades))
                    pred = round(ens_mean)
                    margin = float(np.mean(1.0 - np.abs(2 * logit0 - 1.0)))  # 0.5 근접도(불확실)
                    row[f"{t}_y"] = rd[t]; row[f"{t}_pred"] = pred
                    margins.append(margin); ens_stds.append(min(1.0, ens_std / 1.5))
                    errs.append(abs(rd[t] - pred))
                row["unc_margin"] = float(np.mean(margins)) if margins else 1.0
                row["unc_ensemble"] = float(np.mean(ens_stds)) if ens_stds else 1.0
                row["mean_err"] = float(np.mean(errs)) if errs else np.nan
                rec.append(row)
        print(f"  fold {fold} done (5 seeds)", flush=True)

    df = pd.DataFrame(rec)
    df.to_csv(OUT / "exp_margin_ensemble_oof.csv", index=False, encoding="utf-8")

    def qwk_on(sub):
        vs = []
        for t in ORD:
            s = sub[sub[f"{t}_y"].notna()]
            if len(s) >= 5 and s[f"{t}_y"].nunique() > 1 and s[f"{t}_pred"].nunique() > 1:
                vs.append(cohen_kappa_score(s[f"{t}_y"], s[f"{t}_pred"], weights="quadratic", labels=[0, 1, 2, 3]))
        return float(np.mean(vs)) if vs else None

    def risk_coverage(score_col):
        d = df.sort_values(score_col)  # 낮을수록 confident
        rows = []
        for cov in np.round(np.arange(0.3, 1.01, 0.1), 2):
            k = max(5, int(round(cov * len(d))))
            sub = d.iloc[:k]
            q = qwk_on(sub)
            rows.append({"coverage": round(k / len(d), 3), "n": k,
                         "mean_err": round(float(sub["mean_err"].mean()), 3),
                         "qwk": round(q, 3) if q is not None else None})
        return rows

    full_err = round(float(df["mean_err"].mean()), 3); full_q = qwk_on(df)
    res = {"n": len(df), "full": {"mean_err": full_err, "qwk": round(full_q, 3) if full_q else None},
           "EXP1_margin": risk_coverage("unc_margin"),
           "EXP2_ensemble": risk_coverage("unc_ensemble")}

    def monotone(rows):  # coverage 줄일수록(앞쪽) error 감소?
        errs = [r["mean_err"] for r in rows]   # rows: coverage 0.3..1.0 오름차순
        return errs[0] < errs[-1] - 0.05       # 최저 coverage error가 full보다 의미있게 낮으면 selection 작동
    res["verdict"] = {
        "EXP1_selection_works": monotone(res["EXP1_margin"]),
        "EXP2_selection_works": monotone(res["EXP2_ensemble"]),
        "note": "둘 다 False면 (D) 데이터 한계 → router QC-only 고정. True면 해당 score 채택."}
    (OUT / "exp_margin_ensemble.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, key in [("baseline(MC) full", None), ("EXP1 margin", "EXP1_margin"), ("EXP2 ensemble", "EXP2_ensemble")]:
        if key:
            r = res[key]; ax.plot([x["coverage"] for x in r], [x["mean_err"] for x in r], "o-", label=name)
    ax.axhline(full_err, ls="--", c="r", label=f"full err={full_err}")
    ax.set_xlabel("coverage(ALLOW)"); ax.set_ylabel("ALLOW셋 ordinal err"); ax.legend()
    ax.set_title("Router score 비교 — error가 coverage↓에서 내려가야 selection 작동")
    fig.tight_layout(); fig.savefig(OUT / "exp_margin_ensemble.png", dpi=110); plt.close(fig)

    print(f"\nfull: err={full_err} QWK={res['full']['qwk']} (n={len(df)})")
    for k in ("EXP1_margin", "EXP2_ensemble"):
        print(f"\n[{k}]")
        for r in res[k]:
            print(f"  cov={r['coverage']} n={r['n']} err={r['mean_err']} qwk={r['qwk']}")
    print(f"\nverdict: {res['verdict']}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
