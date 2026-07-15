"""
Exp6 — 다중스케일(10x+40x) late-fusion 학습/평가 (MultiScaleMIL)

bag = {(stain, scale): emb}  scale in {10,40}, stain in MAIN(HE/PAS/MT).
환자단위 CV, 4 task(immune/chronic/stage3/ati_severity), bootstrap95%CI, scale contribution.
train.py와 동일 거버넌스(누수0·gold·seed). encoder=ctranspath 기본.

사용: python train_multiscale.py --encoder ctranspath --scales 10,40 --tag exp6
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "Pathology_model")))

ROOT = Path(__file__).resolve().parents[2]
SEED = 42
MAIN_STAINS = ["HE", "PAS", "MT"]
CLF = ["immune", "chronic", "stage3"]
COL = {"immune": "task_immune", "chronic": "task_chronic", "stage3": "task_stage3"}


def seed_all(s):
    import torch
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def load_scale_bags(pids, idx, scales):
    """patient -> {(stain,scale): array}. MAIN stain·지정 scale만."""
    bags = {}
    for pid in pids:
        sub = idx[idx["patient_id"].astype(str) == str(pid)]
        d = {}
        for r in sub.itertuples():
            if r.stain not in MAIN_STAINS or str(r.magnification) not in scales:
                continue
            arr = np.load(ROOT / r.npy_path)
            if arr.shape[0] > 0:
                d[(r.stain, str(r.magnification))] = arr.astype(np.float32)
        if d:
            bags[str(pid)] = d
    return bags


def _boot_ci(y, p, fn, n=2000, seed=SEED):
    y, p = np.array(y), np.array(p); rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        ix = rng.integers(0, len(y), len(y))
        if len(np.unique(y[ix])) > 1:
            v.append(fn(y[ix], p[ix]))
    return [None, None] if not v else [round(float(np.percentile(v, 2.5)), 3),
                                       round(float(np.percentile(v, 97.5)), 3)]


def clf_metrics(y, p):
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 f1_score, balanced_accuracy_score)
    y, p = np.array(y), np.array(p)
    if len(np.unique(y)) < 2:
        return {"auroc": None, "n": int(len(y)), "n_pos": int(y.sum())}
    pred = (p > 0.5).astype(int)
    return {"auroc": round(float(roc_auc_score(y, p)), 3), "auroc_ci95": _boot_ci(y, p, roc_auc_score),
            "pr_auc": round(float(average_precision_score(y, p)), 3),
            "f1": round(float(f1_score(y, pred, zero_division=0)), 3),
            "balanced_acc": round(float(balanced_accuracy_score(y, pred)), 3),
            "n": int(len(y)), "n_pos": int(y.sum())}


def main():
    import torch
    import torch.nn.functional as F
    from mil.model import MultiScaleMIL
    from scipy.stats import spearmanr

    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="ctranspath")
    ap.add_argument("--scales", default="10,40")
    ap.add_argument("--tag", default="exp6")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    args = ap.parse_args()

    seed_all(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scales = [s.strip() for s in args.scales.split(",")]
    idx = pd.read_csv(ROOT / "data/embeddings" / args.encoder / "index.csv",
                      dtype={"magnification": str})
    embed_dim = int(idx["embed_dim"].iloc[0])

    sm = pd.read_csv(ROOT / "split_manifest.csv").drop_duplicates("patient_id")
    coh = sm[sm["fold"] >= 0].copy(); coh["patient_id"] = coh["patient_id"].astype(str)
    coh["ati_severity_n"] = (coh["task_ati_severity"] - 1) / 2.0

    bags = load_scale_bags(coh["patient_id"].tolist(), idx, scales)
    coh = coh[coh["patient_id"].isin(bags.keys())]
    print(f"[{args.tag}] {args.encoder}({embed_dim}d) scales={scales} stains={MAIN_STAINS} "
          f"cohort={len(coh)}", flush=True)

    folds = sorted(coh["fold"].unique())
    oof = {t: {"y": [], "p": [], "pid": [], "fold": []} for t in CLF + ["ati_severity"]}
    sc_sum, sc_cnt = {}, {}
    for fold in folds:
        tr = coh[coh["fold"] != fold]; va = coh[coh["fold"] == fold]
        model = MultiScaleMIL(in_dim=embed_dim, scales=tuple(scales),
                              stains=tuple(MAIN_STAINS)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

        def pw(c):
            yy = tr[c].dropna(); return torch.tensor((yy == 0).sum() / max((yy == 1).sum(), 1),
                                                     device=device, dtype=torch.float32)
        PW = {t: pw(COL[t]) for t in CLF}
        for ep in range(args.epochs):
            model.train()
            for r in tr.sample(frac=1, random_state=SEED + ep).itertuples():
                if r.patient_id not in bags:
                    continue
                bag = {k: torch.from_numpy(v).to(device) for k, v in bags[r.patient_id].items()}
                out = model(bag); loss = 0.0; nt = 0
                for t in CLF:
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
        with torch.no_grad():
            for r in va.itertuples():
                if r.patient_id not in bags:
                    continue
                bag = {k: torch.from_numpy(v).to(device) for k, v in bags[r.patient_id].items()}
                out = model(bag)
                for s, w in out["scale_contrib"].items():
                    sc_sum[s] = sc_sum.get(s, 0.0) + w; sc_cnt[s] = sc_cnt.get(s, 0) + 1
                for t in CLF:
                    yv = getattr(r, COL[t])
                    if not (yv != yv):
                        oof[t]["y"].append(int(yv)); oof[t]["p"].append(float(torch.sigmoid(out[t])))
                        oof[t]["pid"].append(r.patient_id); oof[t]["fold"].append(int(fold))
                sv = r.ati_severity_n
                if not (sv != sv):
                    oof["ati_severity"]["y"].append(float(sv))
                    oof["ati_severity"]["p"].append(float(torch.sigmoid(out["ati_severity"])))
                    oof["ati_severity"]["pid"].append(r.patient_id); oof["ati_severity"]["fold"].append(int(fold))
        print(f"  fold {fold} done", flush=True)

    report = {"config": {"tag": args.tag, "encoder": args.encoder, "scales": scales,
                         "stains": MAIN_STAINS, "embed_dim": embed_dim, "seed": SEED,
                         "epochs": args.epochs, "cohort": int(len(coh))}, "tasks": {}}
    print(f"\n=== [{args.tag}] 멀티스케일 CV (OOF) ===")
    for t in CLF:
        m = clf_metrics(oof[t]["y"], oof[t]["p"]); report["tasks"][t] = m
        print(f"[{t}] AUROC {m.get('auroc')} CI{m.get('auroc_ci95')} | PR-AUC {m.get('pr_auc')} | "
              f"BalAcc {m.get('balanced_acc')} | n{m.get('n')}({m.get('n_pos')})")
    if oof["ati_severity"]["y"]:
        y = np.array(oof["ati_severity"]["y"]); p = np.array(oof["ati_severity"]["p"])
        rho = float(spearmanr(y, p).correlation) if len(np.unique(y)) > 1 else None
        report["tasks"]["ati_severity"] = {"spearman": None if rho is None else round(rho, 3),
                                           "mae": round(float(np.abs(y - p).mean()), 3), "n": int(len(y))}
        print(f"[ati_severity] Spearman {report['tasks']['ati_severity']['spearman']} | n{len(y)}")
    report["scale_contribution"] = {s: round(sc_sum[s] / sc_cnt[s], 3) for s in sc_sum}
    print(f"[scale contribution] {report['scale_contribution']}")
    (ROOT / f"mil_cv_{args.tag}_{args.encoder}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = [{"task": t, "patient_id": oof[t]["pid"][i], "fold": oof[t]["fold"][i],
             "y": oof[t]["y"][i], "p": oof[t]["p"][i]}
            for t in oof for i in range(len(oof[t]["y"]))]
    pd.DataFrame(rows).to_csv(ROOT / f"oof_{args.tag}_{args.encoder}.csv", index=False, encoding="utf-8")
    print(f"\n-> mil_cv_{args.tag}_{args.encoder}.json , oof_{args.tag}_{args.encoder}.csv")


if __name__ == "__main__":
    main()
