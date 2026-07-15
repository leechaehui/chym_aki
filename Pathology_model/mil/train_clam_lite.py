"""
Exp8 — CLAM-lite baseline 환자단위 CV (설계 §8)

StainAwareMIL/MultiScaleMIL 와 '동일한 데이터·split·태스크·지표'로 CLAM-lite(Top-K=16)를
돌려 MIL 복잡도(stain-aware fusion·multi-scale·SILVER 제약)의 실효를 가른다.

train.py 와 거버넌스 동일(누수0 gold split·seed42·클래스가중·bootstrap95%CI).
SILVER 미사용·multi-scale 미사용(=baseline). 인스턴스 풀은 비-SILVER stain 패치 합본.

사용: python train_clam_lite.py --encoder ctranspath --mags 10 --tag exp8 --topk 16
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("c:/team/chym_aki/Pathology_model")))

from mil.train import (CLF_TASKS, COL, ROOT, SEED, clf_metrics, load_bags,
                       seed_all)


def main():
    import torch
    import torch.nn.functional as F
    from mil.clam_lite import TOPK, ClamLite
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr

    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="ctranspath")
    ap.add_argument("--mags", default="10")
    ap.add_argument("--tag", default="exp8")
    ap.add_argument("--stains", default="", help="콤마구분 stain 제한(비우면 전체, SILVER는 항상 미사용)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--topk", type=int, default=TOPK)
    args = ap.parse_args()

    seed_all(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from mil.cdss_paths import p as _p
    emb_dir = _p("embeddings") / args.encoder
    idx = pd.read_csv(emb_dir / "index.csv", dtype={"magnification": str})
    msel = [m.strip() for m in args.mags.split(",")]
    idx = idx[idx["magnification"].isin(msel) |
              ((idx["stain"] == "IF") & (idx["magnification"] == "native"))].copy()
    embed_dim = int(idx["embed_dim"].iloc[0])

    sm = pd.read_csv(ROOT / "split_manifest.csv").drop_duplicates("patient_id")
    coh = sm[sm["fold"] >= 0].copy()
    coh["patient_id"] = coh["patient_id"].astype(str)
    coh["ati_severity_n"] = (coh["task_ati_severity"] - 1) / 2.0

    keep = set(args.stains.split(",")) if args.stains else None
    bags = load_bags(coh["patient_id"].tolist(), idx, keep=keep)
    coh = coh[coh["patient_id"].isin(bags.keys())]
    n_lab = {t: int(coh[COL[t]].notna().sum()) for t in CLF_TASKS + ["ati_severity"]}
    print(f"[{args.tag}] CLAM-lite(K={args.topk}) encoder={args.encoder}({embed_dim}d) "
          f"device={device} cohort={len(coh)} labels={n_lab}", flush=True)

    folds = sorted(coh["fold"].unique())
    oof = {t: {"y": [], "p": [], "pid": [], "fold": []} for t in CLF_TASKS + ["ati_severity"]}
    per_fold = {t: [] for t in CLF_TASKS + ["ati_severity"]}

    for fold in folds:
        tr = coh[coh["fold"] != fold]; va = coh[coh["fold"] == fold]
        model = ClamLite(in_dim=embed_dim, topk=args.topk).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

        def pw(col):
            yy = tr[col].dropna(); n1 = (yy == 1).sum(); n0 = (yy == 0).sum()
            return torch.tensor(n0 / max(n1, 1), device=device, dtype=torch.float32)
        PW = {t: pw(COL[t]) for t in CLF_TASKS}

        for ep in range(args.epochs):
            model.train()
            for r in tr.sample(frac=1, random_state=SEED + ep).itertuples():
                bag = {s: torch.from_numpy(v).to(device) for s, v in bags[r.patient_id].items()}
                out = model(bag); loss = 0.0; nt = 0
                for t in CLF_TASKS:
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
        fp = {t: {"y": [], "p": []} for t in CLF_TASKS + ["ati_severity"]}
        with torch.no_grad():
            for r in va.itertuples():
                bag = {s: torch.from_numpy(v).to(device) for s, v in bags[r.patient_id].items()}
                out = model(bag)

                def rec(t, yv, pv):
                    oof[t]["y"].append(yv); oof[t]["p"].append(pv)
                    oof[t]["pid"].append(r.patient_id); oof[t]["fold"].append(int(fold))
                    fp[t]["y"].append(yv); fp[t]["p"].append(pv)
                for t in CLF_TASKS:
                    yv = getattr(r, COL[t])
                    if not (yv != yv):
                        rec(t, int(yv), float(torch.sigmoid(out[t])))
                sv = r.ati_severity_n
                if not (sv != sv):
                    rec("ati_severity", float(sv), float(torch.sigmoid(out["ati_severity"])))
        for t in CLF_TASKS:
            yy = fp[t]["y"]
            per_fold[t].append(round(float(roc_auc_score(yy, fp[t]["p"])), 3) if len(set(yy)) > 1 else None)
        ya = fp["ati_severity"]["y"]
        per_fold["ati_severity"].append(
            round(float(spearmanr(ya, fp["ati_severity"]["p"]).correlation), 3) if len(set(ya)) > 1 else None)
        print(f"  fold {fold} done", flush=True)

    report = {"config": {"tag": args.tag, "model": "CLAM-lite", "topk": args.topk,
                         "encoder": args.encoder, "embed_dim": embed_dim, "mags": args.mags,
                         "patch": 512, "seed": SEED, "epochs": args.epochs, "lr": args.lr,
                         "wd": args.wd, "device": device, "cohort": int(len(coh)),
                         "labels": n_lab, "silver": "unused", "multiscale": "unused"},
              "tasks": {}}
    print(f"\n=== [{args.tag}] CLAM-lite 환자단위 CV (OOF) ===")
    for t in CLF_TASKS:
        m = clf_metrics(oof[t]["y"], oof[t]["p"]); m["per_fold_auroc"] = per_fold[t]
        report["tasks"][t] = m
        print(f"[{t}] AUROC {m.get('auroc')} CI{m.get('auroc_ci95')} | PR-AUC {m.get('pr_auc')} | "
              f"F1 {m.get('f1')} | BalAcc {m.get('balanced_acc')} | n{m.get('n')}({m.get('n_pos')})")
    if oof["ati_severity"]["y"]:
        y = np.array(oof["ati_severity"]["y"]); p = np.array(oof["ati_severity"]["p"])
        rho = float(spearmanr(y, p).correlation) if len(np.unique(y)) > 1 else None
        report["tasks"]["ati_severity"] = {"spearman": None if rho is None else round(rho, 3),
                                           "mae": round(float(np.abs(y - p).mean()), 3),
                                           "n": int(len(y)), "per_fold_spearman": per_fold["ati_severity"]}
        print(f"[ati_severity] Spearman {report['tasks']['ati_severity']['spearman']} | "
              f"MAE {report['tasks']['ati_severity']['mae']} | n{len(y)}")
    (ROOT / f"mil_cv_{args.tag}_{args.encoder}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = [{"task": t, "patient_id": oof[t]["pid"][i], "fold": oof[t]["fold"][i],
             "y": oof[t]["y"][i], "p": oof[t]["p"][i]}
            for t in oof for i in range(len(oof[t]["y"]))]
    pd.DataFrame(rows).to_csv(ROOT / f"oof_{args.tag}_{args.encoder}.csv", index=False, encoding="utf-8")
    print(f"\n-> mil_cv_{args.tag}_{args.encoder}.json , oof_{args.tag}_{args.encoder}.csv")


if __name__ == "__main__":
    main()
