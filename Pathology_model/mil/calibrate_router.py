"""
#1 Router 임계 Calibration — selective prediction / risk-coverage

목적: cdss_engine의 ABSTAIN 임계(uncertainty)를 임의값(0.7)이 아니라 데이터로 정한다.
핵심 검증: "불확실성↑ 환자를 abstain하면 남은(ALLOW) 환자의 ordinal 오차가 실제로 줄어드는가?"
→ 그렇다면 uncertainty는 유효한 게이트. risk-coverage 곡선에서 운영점(τ)을 선택.

방법: 멀티스케일 ordinal CV(누수0) OOF에서 환자별 (true grade, pred grade, MC-dropout uncertainty)
수집 → 환자단위 overall_uncertainty로 정렬 → 임계 τ별 coverage / 남은셋 QWK·MAE 계산.

출력: results/11_router_calibration/{risk_coverage.csv, risk_coverage.png, recommended.json}
       + config.json 의 router 임계 갱신(단일 소스).
사용: python calibrate_router.py
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
MC = 20


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
    lab = pd.DataFrame({"patient_id": desc["patient_id"]})
    lab = lab.merge(sm[["patient_id", "task_immune"]], on="patient_id", how="left")
    for t, c in src.items():
        v = pd.to_numeric(desc[c], errors="coerce").where(lambda x: x < 999)
        lab[t] = v.map(lambda x: to_ord(x, BANFF[t]))
    bags = load_bags(lab["patient_id"].tolist(), idx, keep=STAIN_KEEP)
    lab = lab[lab["patient_id"].isin(bags) & lab[ORD].notna().any(axis=1)].reset_index(drop=True)
    strat = lab["task_immune"].map({1: "AIN", 0: "ATI"}).fillna("none").to_numpy()
    pids = lab["patient_id"].to_numpy()
    out_dims = {t: KBINS - 1 for t in ORD}

    rec = []  # 환자별 OOF: overall_unc, per-task err
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for fold, (tri, vai) in enumerate(skf.split(pids, strat)):
        tr = lab.iloc[tri]; va = lab.iloc[vai]
        torch.manual_seed(SEED); np.random.seed(SEED)
        model = TaskAttentionMIL(in_dim=embed_dim, silver_mode="prediction",
                                 tasks=ORD, out_dims=out_dims).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        for ep in range(40):
            model.train()
            for r in tr.sample(frac=1, random_state=SEED + ep).itertuples(index=False):
                rd = dict(zip(lab.columns, r))
                bag = {s: torch.from_numpy(v).to(device) for s, v in bags[rd["patient_id"]].items()}
                out = model(bag); loss = 0.0; nt = 0
                for t in ORD:
                    yv = rd[t]
                    if yv == yv:
                        lv = torch.tensor([1.0 if yv > k else 0.0 for k in range(KBINS - 1)], device=device)
                        loss = loss + F.binary_cross_entropy_with_logits(out[t], lv); nt += 1
                if nt:
                    opt.zero_grad(); loss.backward(); opt.step()
        # OOF: MC-dropout uncertainty (engine 동일 공식)
        model.eval()
        for m in model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()
        with torch.no_grad():
            for r in va.itertuples(index=False):
                rd = dict(zip(lab.columns, r))
                bag = {s: torch.from_numpy(v).to(device) for s, v in bags[rd["patient_id"]].items()}
                g = {t: [] for t in ORD}; ent = {t: [] for t in ORD}
                for _ in range(MC):
                    out = model(bag)
                    for t in ORD:
                        g[t].append(float((torch.sigmoid(out[t]) > 0.5).sum()))
                        e = out["task_entropy"].get(t)
                        if e is not None:
                            ent[t].append(float(e))
                row = {"patient_id": rd["patient_id"]}
                u_parts, errs = [], []
                for t in ORD:
                    if rd[t] != rd[t]:
                        continue
                    pred = float(np.mean(g[t])); gstd = float(np.std(g[t]))
                    e_m = float(np.mean(ent[t])) if ent[t] else 1.0
                    u = 0.5 * min(1.0, gstd / 1.5) + 0.5 * e_m
                    row[f"{t}_y"] = rd[t]; row[f"{t}_pred"] = round(pred)
                    row[f"{t}_err"] = abs(rd[t] - round(pred)); row[f"{t}_u"] = u
                    u_parts.append(u); errs.append(abs(rd[t] - round(pred)))
                row["overall_unc"] = float(np.mean(u_parts)) if u_parts else 1.0
                row["mean_err"] = float(np.mean(errs)) if errs else np.nan
                rec.append(row)
        print(f"  fold {fold} done", flush=True)

    df = pd.DataFrame(rec)
    df.to_csv(OUT / "oof_uncertainty.csv", index=False, encoding="utf-8")

    # risk-coverage: τ 낮출수록 abstain↑(coverage↓), 남은셋 오차/QWK
    def qwk_retained(mask):
        out = {}
        for t in ORD:
            sub = df[mask & df[f"{t}_y"].notna()]
            if len(sub) >= 5 and sub[f"{t}_y"].nunique() > 1 and sub[f"{t}_pred"].nunique() > 1:
                out[t] = round(float(cohen_kappa_score(sub[f"{t}_y"], sub[f"{t}_pred"],
                                weights="quadratic", labels=[0, 1, 2, 3])), 3)
            else:
                out[t] = None
        return out

    rows = []
    for tau in np.round(np.arange(0.30, 0.86, 0.05), 2):
        mask = df["overall_unc"] < tau
        cov = float(mask.mean())
        if mask.sum() < 5:
            continue
        q = qwk_retained(mask)
        qv = [v for v in q.values() if v is not None]
        rows.append({"tau": tau, "coverage": round(cov, 3), "n_allow": int(mask.sum()),
                     "mean_err_allow": round(float(df[mask]["mean_err"].mean()), 3),
                     "qwk_mean": round(float(np.mean(qv)), 3) if qv else None, **q})
    rc = pd.DataFrame(rows)
    rc.to_csv(OUT / "risk_coverage.csv", index=False, encoding="utf-8")

    full_err = round(float(df["mean_err"].mean()), 3)
    full_q = qwk_retained(pd.Series([True] * len(df)))
    full_qmean = round(float(np.mean([v for v in full_q.values() if v is not None])), 3)

    # 운영점: coverage>=0.6 중 mean_err 최소(동률이면 coverage 큰 쪽)
    cand = rc[rc["coverage"] >= 0.6]
    pick = (cand.sort_values(["mean_err_allow", "coverage"], ascending=[True, False]).iloc[0]
            if len(cand) else rc.sort_values("coverage", ascending=False).iloc[0])
    tau_star = float(pick["tau"])

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(rc["coverage"], rc["mean_err_allow"], "o-"); ax[0].axhline(full_err, ls="--", c="r", label=f"전체 err={full_err}")
    ax[0].set_xlabel("coverage(ALLOW 비율)"); ax[0].set_ylabel("ALLOW셋 평균 ordinal err"); ax[0].legend(); ax[0].set_title("Risk-Coverage")
    ax[1].plot(rc["tau"], rc["qwk_mean"], "s-"); ax[1].axhline(full_qmean, ls="--", c="r", label=f"전체 QWK={full_qmean}")
    ax[1].axvline(tau_star, ls=":", c="g", label=f"τ*={tau_star}"); ax[1].set_xlabel("uncertainty τ"); ax[1].set_ylabel("ALLOW셋 QWK_mean"); ax[1].legend(); ax[1].set_title("τ vs QWK")
    fig.tight_layout(); fig.savefig(OUT / "risk_coverage.png", dpi=110); plt.close(fig)

    rec_json = {"recommended_uncertainty_tau": tau_star,
                "operating_point": {k: (None if pd.isna(pick[k]) else float(pick[k]) if k != "tau" else float(pick[k]))
                                    for k in ["tau", "coverage", "n_allow", "mean_err_allow", "qwk_mean"]},
                "full_cohort": {"mean_err": full_err, "qwk_mean": full_qmean, "n": int(len(df))},
                "note": "coverage>=0.6 중 ALLOW셋 ordinal err 최소 운영점. abstain이 오차를 줄이면 uncertainty 게이트 유효.",
                "qc_tau": 0.40, "ord_conf_min": 0.40}
    (OUT / "recommended.json").write_text(json.dumps(rec_json, indent=2, ensure_ascii=False), encoding="utf-8")

    # config.json router 섹션 갱신(단일 소스)
    cfgp = ROOT / "Pathology_model/config.json"
    cfg = json.loads(cfgp.read_text(encoding="utf-8"))
    cfg["router"] = {"uncertainty_abstain": tau_star, "qc_abstain": 0.40, "ord_conf_min": 0.40,
                     "calibrated": True, "source": "calibrate_router.py risk-coverage"}
    cfgp.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n전체: err={full_err} QWK={full_qmean} (n={len(df)})")
    print(rc.to_string(index=False))
    print(f"\n추천 uncertainty τ* = {tau_star} (coverage {pick['coverage']}, "
          f"ALLOW err {pick['mean_err_allow']} vs 전체 {full_err}, QWK {pick['qwk_mean']} vs {full_qmean})")
    print(f"-> {OUT} , config.json[router] 갱신")


if __name__ == "__main__":
    main()
