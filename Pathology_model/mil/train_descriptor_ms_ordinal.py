"""
1층 검증 — Multi-Scale(10x+40x) + Ordinal(Banff) descriptor 예측

목적(무비용): "40x 추가 + ordinal 전환 시 fibrosis/atrophy에 신호가 생기는가"를 판정.
- 멀티스케일: stain별 bag = 10x∪40x 패치 풀(load_bags가 배율 자동 vstack). Silver는 40x 부재→10x.
- Ordinal: fibrosis(ci)/atrophy(ct)/inflammation(i)을 Banff 4-bin(0–3)으로, CORAL(rank-monotone) 학습.
- immune/tubulitis는 이진 유지. 평가: QWK + Spearman + ordinal-AUROC(임계별) + bootstrap CI. pooled-OOF.
- 직전 10x 회귀 결과(results/08_descriptor_prediction)와 대조.

사용: python train_descriptor_ms_ordinal.py --encoder ctranspath
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "Pathology_model")))
from mil.train import ROOT, SEED, load_bags, seed_all
from mil.cdss_paths import p as _p

_ART = Path(__file__).resolve().parents[1] / "artifacts"   # Pathology_model/artifacts (코드 상대 — 데이터 ROOT 아님)
OUT = ROOT / "Pathology_model/results/10_multiscale_ordinal"
STAIN_KEEP = {"HE", "PAS", "MT", "SILVER"}
BIN_CLF = ["immune", "tubulitis"]                  # 이진
ORD = ["fibrosis", "atrophy", "inflammation"]      # ordinal 0–3 (ci/ct/i)
KBINS = 4
# Banff 경계(%): ci/ct = ≤5/6–25/26–50/>50 ; i(WBC%) = <10/10–25/26–50/>50
BANFF = {"fibrosis": [5, 25, 50], "atrophy": [5, 25, 50], "inflammation": [10, 25, 50]}


def to_ord(v, edges):
    if v != v:
        return np.nan
    return float(sum(v > e for e in edges))   # 0..3


def boot_ci(y, p, fn, n=2000, seed=SEED, need2=False):
    y, p = np.asarray(y, float), np.asarray(p, float)
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        ix = rng.integers(0, len(y), len(y))
        if need2 and len(np.unique(y[ix])) < 2:
            continue
        try:
            vals.append(fn(y[ix], p[ix]))
        except Exception:
            pass
    pt = round(float(fn(y, p)), 3) if (not need2 or len(np.unique(y)) > 1) else None
    if not vals:
        return {"point": pt, "ci95": [None, None]}
    return {"point": pt, "ci95": [round(float(np.percentile(vals, 2.5)), 3),
                                  round(float(np.percentile(vals, 97.5)), 3)]}


def main():
    import torch
    import torch.nn.functional as F
    from mil.model import TaskAttentionMIL
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (cohen_kappa_score, roc_auc_score, average_precision_score,
                                 f1_score, precision_score, recall_score)
    from scipy.stats import spearmanr

    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="ctranspath")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--stains", default="HE,PAS,MT,SILVER",
                    help="학습에 쓸 stain(콤마구분). 단일 stain 실험은 예: --stains PAS")
    ap.add_argument("--tag", default="", help="출력 폴더 접미사(baseline 미덮어씀). 예: --tag _pas_single")
    ap.add_argument("--seeds", type=int, default=1,
                    help="앙상블 시드 수(배포는 5). >1 이면 fold별로 시드마다 학습→예측 평균(pooled-OOF).")
    args = ap.parse_args()
    keep_stains = {s.strip().upper() for s in args.stains.split(",") if s.strip()}
    out_dir = ROOT / "Pathology_model/results" / f"10_multiscale_ordinal{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_all(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 10x + 40x 동시 포함 → load_bags가 stain별로 배율 연결(멀티스케일 풀)
    idx = pd.read_csv(_p("embeddings") / args.encoder / "index.csv", dtype={"magnification": str})
    idx = idx[idx["magnification"].isin(["10", "40"]) & idx["stain"].isin(keep_stains)].copy()
    embed_dim = int(idx["embed_dim"].iloc[0])

    sm = pd.read_csv(_ART / "split_manifest.csv").drop_duplicates("patient_id")
    sm["patient_id"] = sm["patient_id"].astype(str)
    desc = pd.read_csv(_ART / "descriptor_labels.csv")
    desc["patient_id"] = desc["patient_id"].astype(str)
    df = sm[["patient_id", "task_immune"]].merge(desc, on="patient_id", how="outer")

    def num(src):
        v = pd.to_numeric(df[src], errors="coerce"); return v.where(v < 999)
    lab = pd.DataFrame({"patient_id": df["patient_id"]})
    lab["immune"] = pd.to_numeric(df["task_immune"], errors="coerce")
    lab["tubulitis"] = num("tubulitis")
    lab["fibrosis"] = num("interstitial_fibrosis_pct").map(lambda v: to_ord(v, BANFF["fibrosis"]))
    lab["atrophy"] = num("tubular_atrophy").map(lambda v: to_ord(v, BANFF["atrophy"]))
    lab["inflammation"] = num("interstitial_mononuclear_wbc_pct").map(lambda v: to_ord(v, BANFF["inflammation"]))

    bags = load_bags(lab["patient_id"].tolist(), idx, keep=keep_stains)
    lab = lab[lab["patient_id"].isin(bags.keys())].reset_index(drop=True)
    tasks = BIN_CLF + ORD
    lab = lab[lab[tasks].notna().any(axis=1)].reset_index(drop=True)
    out_dims = {t: KBINS - 1 for t in ORD}        # CORAL: K-1 logits
    n_lab = {t: int(lab[t].notna().sum()) for t in tasks}
    # ordinal 분포(고등급 존재 확인)
    ord_dist = {t: lab[t].value_counts().sort_index().to_dict() for t in ORD}
    print(f"cohort={len(lab)} | labels={n_lab}", flush=True)
    print(f"ordinal 분포(0-3): {ord_dist}", flush=True)

    strat = lab["immune"].map({1: "AIN", 0: "ATI"}).fillna("none").to_numpy()
    pids = lab["patient_id"].to_numpy()
    oof = {t: {"y": [], "p": [], "pid": []} for t in tasks}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for fold, (tri, vai) in enumerate(skf.split(pids, strat)):
        tr = lab.iloc[tri]; va = lab.iloc[vai]

        def pw(t):
            yy = tr[t].dropna(); n1 = (yy == 1).sum(); n0 = (yy == 0).sum()
            return torch.tensor(n0 / max(n1, 1), device=device, dtype=torch.float32)
        PW = {t: pw(t) for t in BIN_CLF}

        # 시드별 학습→val 예측 누적(배포 5-seed 앙상블과 동일: 확률/등급 합산 후 /seeds).
        # seeds=1 이면 기존 1-seed 동작과 동일(baseline 불변).
        va_rows = list(va.itertuples(index=False))
        psum = {t: [0.0] * len(va_rows) for t in tasks}
        for si in range(args.seeds):
            sd = SEED + si
            torch.manual_seed(sd); np.random.seed(sd)
            model = TaskAttentionMIL(in_dim=embed_dim, silver_mode="prediction",
                                     tasks=tasks, out_dims=out_dims).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
            for ep in range(args.epochs):
                model.train()
                for r in tr.sample(frac=1, random_state=sd + ep).itertuples(index=False):
                    rd = dict(zip(lab.columns, r)); pid = rd["patient_id"]
                    bag = {s: torch.from_numpy(v).to(device) for s, v in bags[pid].items()}
                    out = model(bag); loss = 0.0; nt = 0
                    for t in BIN_CLF:
                        yv = rd[t]
                        if yv == yv:
                            loss = loss + F.binary_cross_entropy_with_logits(
                                out[t], torch.tensor(float(yv), device=device), pos_weight=PW[t]); nt += 1
                    for t in ORD:
                        yv = rd[t]
                        if yv == yv:
                            # CORAL: level k 이진라벨 = [y > k], k=0..K-2
                            lv = torch.tensor([1.0 if yv > k else 0.0 for k in range(KBINS - 1)], device=device)
                            loss = loss + F.binary_cross_entropy_with_logits(out[t], lv); nt += 1
                    if nt:
                        opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                for j, r in enumerate(va_rows):
                    rd = dict(zip(lab.columns, r)); pid = rd["patient_id"]
                    bag = {s: torch.from_numpy(v).to(device) for s, v in bags[pid].items()}
                    out = model(bag)
                    for t in BIN_CLF:
                        if rd[t] == rd[t]:
                            psum[t][j] += float(torch.sigmoid(out[t]))
                    for t in ORD:
                        if rd[t] == rd[t]:
                            probs = torch.sigmoid(out[t])
                            psum[t][j] += float((probs > 0.5).sum())   # CORAL 예측 등급
        # 시드 평균 → OOF (ORD 는 등급 평균을 반올림해 이산 등급 유지)
        for j, r in enumerate(va_rows):
            rd = dict(zip(lab.columns, r)); pid = rd["patient_id"]
            for t in BIN_CLF:
                if rd[t] == rd[t]:
                    oof[t]["y"].append(int(rd[t])); oof[t]["p"].append(psum[t][j] / args.seeds)
                    oof[t]["pid"].append(pid)
            for t in ORD:
                if rd[t] == rd[t]:
                    oof[t]["y"].append(int(rd[t])); oof[t]["p"].append(float(round(psum[t][j] / args.seeds)))
                    oof[t]["pid"].append(pid)
        print(f"  fold {fold} done ({args.seeds}-seed)", flush=True)

    report = {"config": {"encoder": args.encoder, "scales": "10x+40x", "model": "TaskAttn-MS-CORAL",
                         "stains": "/".join(sorted(keep_stains)), "seed": SEED, "epochs": args.epochs,
                         "cohort": int(len(lab)), "labels": n_lab, "ordinal_dist": ord_dist,
                         "banff_edges": BANFF}, "tasks": {}}
    print("\n=== Multi-Scale + Ordinal (pooled-OOF + bootstrap CI) ===")
    for t in BIN_CLF:
        y = np.array(oof[t]["y"]); p = np.array(oof[t]["p"]); pred = (p > 0.5).astype(int)
        report["tasks"][t] = {"type": "clf", "n": len(y), "n_pos": int(y.sum()),
            "AUROC": boot_ci(y, p, roc_auc_score, need2=True),
            "AUPRC": boot_ci(y, p, average_precision_score, need2=True),
            "precision": round(float(precision_score(y, pred, zero_division=0)), 3),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 3),
            "f1": round(float(f1_score(y, pred, zero_division=0)), 3)}
        m = report["tasks"][t]
        print(f"[{t}] AUROC {m['AUROC']['point']} CI{m['AUROC']['ci95']} | AUPRC {m['AUPRC']['point']} | "
              f"P{m['precision']} R{m['recall']} | n{m['n']}({m['n_pos']})")
    for t in ORD:
        y = np.array(oof[t]["y"]); p = np.array(oof[t]["p"])
        qwk = boot_ci(y, p, lambda a, b: cohen_kappa_score(a, b, weights="quadratic",
                                                           labels=[0, 1, 2, 3]))
        sp = boot_ci(y, p, lambda a, b: spearmanr(a, b).correlation)
        report["tasks"][t] = {"type": "ordinal", "n": len(y),
            "QWK": qwk, "spearman": sp,
            "mae_ord": round(float(np.abs(y - p).mean()), 3),
            "y_dist": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}}
        m = report["tasks"][t]
        print(f"[{t}] QWK {m['QWK']['point']} CI{m['QWK']['ci95']} | Spearman {m['spearman']['point']} "
              f"CI{m['spearman']['ci95']} | MAE_ord {m['mae_ord']} | n{m['n']} dist{m['y_dist']}")
    (out_dir / "ms_ordinal_metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = [{"task": t, "patient_id": oof[t]["pid"][i], "y": oof[t]["y"][i], "p": oof[t]["p"][i]}
            for t in tasks for i in range(len(oof[t]["y"]))]
    pd.DataFrame(rows).to_csv(out_dir / "ms_ordinal_oof.csv", index=False, encoding="utf-8")
    print(f"\n-> {out_dir}")


if __name__ == "__main__":
    main()
