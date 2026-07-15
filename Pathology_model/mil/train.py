"""
Phase D — 환자단위 CV 멀티태스크 학습/평가 (encoder/experiment 인식, bootstrap CI)

태스크(라벨 가용 환자만 각 손실/평가에 기여):
  immune   (BCE, gold ATI/AIN)         - AIN vs ATI
  chronic  (BCE, gold)                  - DKD/HTN vs acute
  ati_severity (MSE, 전체 단일 kdigo)   - KDIGO 1/2/3 회귀 proxy
  stage3   (BCE, 전체 단일 kdigo)       - Stage3 vs Non-Stage3 (보조)

거버넌스: #1 환자단위 fold(누수0, fold는 환자에서 상속) · #5 seed고정 ·
#8 클래스가중 + AUROC/PR-AUC/F1/BalAcc + bootstrap95%CI (Accuracy 미사용).
멀티태스크 1모델/ fold, 각 손실은 라벨 있는 환자만(마스킹). 탐색적 성격.

사용: python train.py --encoder resnet50 --mags 10 --tag exp1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 코드 루트=패키지(이동 따라감)

try:                                        # 데이터 루트 = config.json(머신별 1곳) — 하드코딩 제거(이식성)
    from mil.cdss_paths import data_root as _data_root
    ROOT = _data_root()
except Exception:
    ROOT = Path("c:/team/chym_aki")          # 폴백(config.json 부재 시)
SEED = 42
CLF_TASKS = ["immune", "chronic", "stage3"]
COL = {"immune": "task_immune", "chronic": "task_chronic",
       "stage3": "task_stage3", "ati_severity": "ati_severity_n"}


def seed_all(s):
    import torch
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def load_bags(pids, idx, keep=None):
    """keep=stain 집합(None=전체). 한 stain에 여러 배율 행이 있으면 연결(멀티스케일)."""
    bags = {}
    for pid in pids:
        sub = idx[idx["patient_id"].astype(str) == str(pid)]
        d = {}
        for r in sub.itertuples():
            if keep is not None and r.stain not in keep:
                continue
            arr = np.load(ROOT / r.npy_path)
            if arr.shape[0] > 0:
                d[r.stain] = (np.vstack([d[r.stain], arr]) if r.stain in d
                              else arr.astype(np.float32))
        if d:
            bags[str(pid)] = d
    return bags


def _boot_ci(y, p, fn, n=2000, seed=SEED):
    y, p = np.array(y), np.array(p)
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        ix = rng.integers(0, len(y), len(y))
        if len(np.unique(y[ix])) < 2:
            continue
        vals.append(fn(y[ix], p[ix]))
    return [None, None] if not vals else [round(float(np.percentile(vals, 2.5)), 3),
                                          round(float(np.percentile(vals, 97.5)), 3)]


def clf_metrics(y, p):
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 f1_score, balanced_accuracy_score)
    y, p = np.array(y), np.array(p)
    if len(np.unique(y)) < 2:
        return {"auroc": None, "n": int(len(y)), "n_pos": int(y.sum())}
    pred = (p > 0.5).astype(int)
    return {"auroc": round(float(roc_auc_score(y, p)), 3),
            "auroc_ci95": _boot_ci(y, p, roc_auc_score),
            "pr_auc": round(float(average_precision_score(y, p)), 3),
            "pr_auc_ci95": _boot_ci(y, p, average_precision_score),
            "f1": round(float(f1_score(y, pred, zero_division=0)), 3),
            "balanced_acc": round(float(balanced_accuracy_score(y, pred)), 3),
            "n": int(len(y)), "n_pos": int(y.sum())}


def main():
    import torch
    import torch.nn.functional as F
    from mil.model import StainAwareMIL
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr

    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="resnet50")
    ap.add_argument("--mags", default="10")
    ap.add_argument("--tag", default="exp")
    ap.add_argument("--stains", default="", help="콤마구분 단독 stain ablation(비우면 전체 multi-stain)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--silver-mode", default="prediction",
                    choices=["prediction", "off", "consistency", "consistency_attn"],
                    help="SILVER 역할(설계 §7). exp7 ablation: off/consistency/consistency_attn")
    ap.add_argument("--silver-lambda", type=float, default=0.3,
                    help="L_silver_consistency 가중 λ (consistency 모드)")
    ap.add_argument("--attn-lambda", type=float, default=0.1,
                    help="attention 안정화 정규화 가중 (consistency_attn 모드)")
    args = ap.parse_args()

    import wandb
    wandb.init(
        project="chym_aki_pathology",
        name=f"{args.tag}_{args.encoder}",
        config=vars(args)
    )

    seed_all(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from mil.cdss_paths import p as _p
    emb_dir = _p("embeddings") / args.encoder
    idx = pd.read_csv(emb_dir / "index.csv", dtype={"magnification": str})
    # 선택 배율만(+ IF native 항상). 멀티스케일은 --mags 10,40
    msel = [m.strip() for m in args.mags.split(",")]
    idx = idx[idx["magnification"].isin(msel) |
              ((idx["stain"] == "IF") & (idx["magnification"] == "native"))].copy()
    embed_dim = int(idx["embed_dim"].iloc[0])

    sm = pd.read_csv(ROOT / "split_manifest.csv").drop_duplicates("patient_id")
    coh = sm[sm["fold"] >= 0].copy()
    coh["patient_id"] = coh["patient_id"].astype(str)
    coh["ati_severity_n"] = (coh["task_ati_severity"] - 1) / 2.0  # 0/0.5/1.0

    keep = set(args.stains.split(",")) if args.stains else None
    bags = load_bags(coh["patient_id"].tolist(), idx, keep=keep)
    coh = coh[coh["patient_id"].isin(bags.keys())]
    n_lab = {t: int(coh[COL[t]].notna().sum()) for t in CLF_TASKS + ["ati_severity"]}
    print(f"[{args.tag}] encoder={args.encoder}({embed_dim}d) device={device} "
          f"cohort={len(coh)} labels={n_lab}", flush=True)

    folds = sorted(coh["fold"].unique())
    oof = {t: {"y": [], "p": [], "pid": [], "fold": []} for t in CLF_TASKS + ["ati_severity"]}
    per_fold = {t: [] for t in CLF_TASKS + ["ati_severity"]}
    sc_sum, sc_cnt = {}, {}   # stain contribution(fusion attention) 집계

    for fold in folds:
        tr = coh[coh["fold"] != fold]; va = coh[coh["fold"] == fold]
        model = StainAwareMIL(in_dim=embed_dim, silver_mode=args.silver_mode).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

        def pw(col):
            yy = tr[col].dropna(); n1 = (yy == 1).sum(); n0 = (yy == 0).sum()
            return torch.tensor(n0 / max(n1, 1), device=device, dtype=torch.float32)
        PW = {t: pw(COL[t]) for t in CLF_TASKS}

        for ep in range(args.epochs):
            model.train()
            train_loss = 0.0
            epoch_stain_contrib = {}
            for r in tr.sample(frac=1, random_state=SEED + ep).itertuples():
                bag = {s: torch.from_numpy(v).to(device) for s, v in bags[r.patient_id].items()}
                out = model(bag); loss = 0.0; nt = 0
                for s, w in out["stain_contrib"].items():
                    if s not in epoch_stain_contrib:
                        epoch_stain_contrib[s] = []
                    epoch_stain_contrib[s].append(w)
                for t in CLF_TASKS:
                    yv = getattr(r, COL[t])
                    if not (yv != yv):  # not NaN
                        loss = loss + F.binary_cross_entropy_with_logits(
                            out[t], torch.tensor(float(yv), device=device), pos_weight=PW[t]); nt += 1
                sv = r.ati_severity_n
                if not (sv != sv):
                    loss = loss + F.mse_loss(torch.sigmoid(out["ati_severity"]),
                                             torch.tensor(float(sv), device=device)); nt += 1
                # SILVER 일관성/attention 정규화 (라벨 있는 환자에만 부가 — gradient만 가산)
                if nt:
                    if out.get("silver_align") is not None:
                        cons = 1.0 - F.cosine_similarity(
                            out["silver_align"].unsqueeze(0),
                            out["silver_target"].unsqueeze(0)).squeeze()
                        loss = loss + args.silver_lambda * cons
                    if out.get("main_attn_entropy") is not None:
                        # 엔트로피(분산)가 클수록 안정 → (1-엔트로피) 패널티 최소화
                        loss = loss + args.attn_lambda * (1.0 - out["main_attn_entropy"])
                    opt.zero_grad(); loss.backward(); opt.step()
                    train_loss += loss.item()
            
            mean_fw = {s: float(np.mean(epoch_stain_contrib[s])) for s in epoch_stain_contrib}
            w_vals = np.array(list(mean_fw.values()))
            w_vals = w_vals / (w_vals.sum() + 1e-8)
            entropy = float(-np.sum(w_vals * np.log(w_vals + 1e-8)))
            
            print(f"Epoch {ep} | Loss: {train_loss / max(len(tr), 1):.4f} | Fusion: ", end="")
            for s, w in mean_fw.items():
                print(f"{s} {w:.2f} ", end="")
            print(f"| Entropy: {entropy:.2f}", flush=True)
            
            wandb.log({f"fold_{fold}/train_loss": train_loss / max(len(tr), 1), "epoch": ep, f"fold_{fold}/fusion_entropy": entropy})

        model.eval()
        fp = {t: {"y": [], "p": []} for t in CLF_TASKS + ["ati_severity"]}
        with torch.no_grad():
            for r in va.itertuples():
                bag = {s: torch.from_numpy(v).to(device) for s, v in bags[r.patient_id].items()}
                out = model(bag)
                for s, w in out["stain_contrib"].items():
                    sc_sum[s] = sc_sum.get(s, 0.0) + w; sc_cnt[s] = sc_cnt.get(s, 0) + 1
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
            auc = round(float(roc_auc_score(yy, fp[t]["p"])), 3) if len(set(yy)) > 1 else None
            per_fold[t].append(auc)
            if auc is not None:
                wandb.log({f"fold_{fold}/val_auroc_{t}": auc})
        ya = fp["ati_severity"]["y"]
        spear = round(float(spearmanr(ya, fp["ati_severity"]["p"]).correlation), 3) if len(set(ya)) > 1 else None
        per_fold["ati_severity"].append(spear)
        if spear is not None:
            wandb.log({f"fold_{fold}/val_spearman_ati_severity": spear})
        print(f"  fold {fold} done", flush=True)

    report = {"config": {"tag": args.tag, "encoder": args.encoder, "embed_dim": embed_dim,
                         "mags": args.mags, "patch": 512, "seed": SEED, "epochs": args.epochs,
                         "lr": args.lr, "wd": args.wd, "device": device, "cohort": int(len(coh)),
                         "labels": n_lab, "norm": "HE=macenko, others=reinhard",
                         "silver_mode": args.silver_mode,
                         "silver_lambda": args.silver_lambda, "attn_lambda": args.attn_lambda},
              "tasks": {}}
    
    summary_metrics = {}
    print(f"\n=== [{args.tag}] 환자단위 CV (OOF) ===")
    for t in CLF_TASKS:
        m = clf_metrics(oof[t]["y"], oof[t]["p"]); m["per_fold_auroc"] = per_fold[t]
        report["tasks"][t] = m
        print(f"[{t}] AUROC {m.get('auroc')} CI{m.get('auroc_ci95')} | PR-AUC {m.get('pr_auc')} | "
              f"F1 {m.get('f1')} | BalAcc {m.get('balanced_acc')} | n{m.get('n')}({m.get('n_pos')})")
        if m.get('auroc') is not None:
            summary_metrics[f"oof_{t}_auroc"] = m.get('auroc')
            summary_metrics[f"oof_{t}_f1"] = m.get('f1')
    if oof["ati_severity"]["y"]:
        y = np.array(oof["ati_severity"]["y"]); p = np.array(oof["ati_severity"]["p"])
        rho = float(spearmanr(y, p).correlation) if len(np.unique(y)) > 1 else None
        report["tasks"]["ati_severity"] = {"spearman": None if rho is None else round(rho, 3),
                                           "mae": round(float(np.abs(y - p).mean()), 3),
                                           "n": int(len(y)), "per_fold_spearman": per_fold["ati_severity"]}
        print(f"[ati_severity] Spearman {report['tasks']['ati_severity']['spearman']} | "
              f"MAE {report['tasks']['ati_severity']['mae']} | n{len(y)} | per-fold {per_fold['ati_severity']}")
        if rho is not None:
            summary_metrics["oof_ati_severity_spearman"] = report['tasks']['ati_severity']['spearman']
        summary_metrics["oof_ati_severity_mae"] = report['tasks']['ati_severity']['mae']
        
    report["stain_contribution"] = {s: round(sc_sum[s] / sc_cnt[s], 3) for s in sc_sum}
    report["config"]["stains"] = args.stains or "all"
    print(f"[stain contribution(mean fusion attn)] {report['stain_contribution']}")
    
    # Log overall metrics to wandb
    wandb.log(summary_metrics)
    wandb.finish()

    (ROOT / f"mil_cv_{args.tag}_{args.encoder}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = [{"task": t, "patient_id": oof[t]["pid"][i], "fold": oof[t]["fold"][i],
             "y": oof[t]["y"][i], "p": oof[t]["p"][i]}
            for t in oof for i in range(len(oof[t]["y"]))]
    pd.DataFrame(rows).to_csv(ROOT / f"oof_{args.tag}_{args.encoder}.csv", index=False, encoding="utf-8")
    print(f"\n-> mil_cv_{args.tag}_{args.encoder}.json , oof_{args.tag}_{args.encoder}.csv")


if __name__ == "__main__":
    main()
