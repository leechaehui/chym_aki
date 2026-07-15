"""
AUPRC 저하 원인 규명 — Attention Collapse 디버깅 프로토콜 (성능개선 아님, 원인규명)

TaskAttentionMIL(chronic/immune)의 낮은 AUPRC 가 ① attention collapse ② 데이터규모
③ 클래스불균형 ④ 과적합 중 무엇 때문인지 분리 검증한다.

수행(실험1~7):
 1. Fold별 AUROC/AUPRC/P/R/F1/pos/neg          -> diag_auprc_fold.csv
 2. 환자별 task entropy/pred/label             -> diag_entropy.csv
 3. Fold mean-entropy vs fold AUPRC 상관        -> entropy_vs_auprc.png (Spearman·Pearson)
 6. Top1/5/10 attention 의존도                  -> diag_top_patch.csv  (Top1>0.7=collapse 의심)
 7. Top1/5/10 patch masking permutation test    -> collapse_diagnosis.json
 4. Attention temperature sweep T=1/1.5/2/3     -> temperature_sweep.json
 5. Entropy regularization sweep λ=0~0.05       -> entropy_sweep.json
 최종 판정(Case A~E)                            -> collapse_diagnosis.json

사용: python diag_collapse.py --encoder ctranspath --mags 10
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "Pathology_model")))
from mil.train import CLF_TASKS, COL, ROOT, SEED, load_bags, seed_all

OUTDIR = ROOT / "Pathology_model/results/05_diag_collapse"
BIN_TASKS = ["immune", "chronic", "stage3"]   # AUPRC 대상(이진). 관심: immune/chronic


def _mask_topk(bag_np, attn, stain_ids, k):
    """attn 상위 k개 (비-SILVER) 패치를 제거한 새 bag(np) 반환."""
    from mil.model import SILVER
    if k <= 0 or k >= len(attn):
        top = set(range(len(attn))) if k >= len(attn) else set()
    else:
        top = set(np.argsort(attn)[::-1][:k].tolist())
    # 전역 행 -> (stain, local idx) 매핑(stain_ids 순서 = 비-SILVER 행 순서)
    drop_by_stain = {}
    for gi, s in enumerate(stain_ids):
        if gi in top:
            drop_by_stain.setdefault(s, set()).add(
                sum(1 for j in range(gi) if stain_ids[j] == s))  # local idx
    new = {}
    for s, arr in bag_np.items():
        if s == SILVER:
            new[s] = arr; continue
        drop = drop_by_stain.get(s, set())
        if drop:
            keep = [i for i in range(arr.shape[0]) if i not in drop]
            new[s] = arr[keep] if keep else arr[:0]
        else:
            new[s] = arr
    # 비어버린 stain 제거
    return {s: a for s, a in new.items() if a.shape[0] > 0}


def run_cv(bags, coh, embed_dim, device, epochs, temperature=1.0, lambda_entropy=0.0,
           capture=False):
    """1회 환자단위 CV. metrics + (capture 시) per-patient/permutation 캡처 반환."""
    import torch
    import torch.nn.functional as F
    from mil.model import TaskAttentionMIL
    from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)

    folds = sorted(coh["fold"].unique())
    oof = {t: {"y": [], "p": [], "fold": [], "pid": [], "ent": [],
               "t1": [], "t5": [], "t10": []} for t in BIN_TASKS}
    perm = {t: {"y": [], "p0": [], "p1": [], "p5": [], "p10": []} for t in BIN_TASKS}

    for fold in folds:
        tr = coh[coh["fold"] != fold]; va = coh[coh["fold"] == fold]
        torch.manual_seed(SEED); np.random.seed(SEED)
        model = TaskAttentionMIL(in_dim=embed_dim, silver_mode="off").to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

        def pw(col):
            yy = tr[col].dropna(); n1 = (yy == 1).sum(); n0 = (yy == 0).sum()
            return torch.tensor(n0 / max(n1, 1), device=device, dtype=torch.float32)
        PW = {t: pw(COL[t]) for t in CLF_TASKS}

        for ep in range(epochs):
            model.train()
            for r in tr.sample(frac=1, random_state=SEED + ep).itertuples():
                bag = {s: torch.from_numpy(v).to(device) for s, v in bags[r.patient_id].items()}
                out = model(bag, temperature=temperature); loss = 0.0; nt = 0
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
                    if lambda_entropy > 0 and out["task_entropy"]:
                        ent = torch.stack(list(out["task_entropy"].values())).mean()
                        loss = loss + lambda_entropy * (1.0 - ent)   # entropy↑ 유도(collapse 완화)
                    opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            for r in va.itertuples():
                bag_np = {s: v for s, v in bags[r.patient_id].items()}
                bag = {s: torch.from_numpy(v).to(device) for s, v in bag_np.items()}
                out = model(bag, temperature=temperature)
                for t in BIN_TASKS:
                    yv = getattr(r, COL[t])
                    if (yv != yv):
                        continue
                    p0 = float(torch.sigmoid(out[t]))
                    a = out["task_attn"][t].cpu().numpy()
                    asort = np.sort(a)[::-1]
                    oof[t]["y"].append(int(yv)); oof[t]["p"].append(p0)
                    oof[t]["fold"].append(int(fold)); oof[t]["pid"].append(r.patient_id)
                    ent = out["task_entropy"].get(t)
                    oof[t]["ent"].append(float(ent) if ent is not None else np.nan)
                    oof[t]["t1"].append(float(asort[:1].sum()))
                    oof[t]["t5"].append(float(asort[:5].sum()))
                    oof[t]["t10"].append(float(asort[:10].sum()))
                    if capture:   # permutation: top-k 제거 후 재예측
                        sid = out["stain_ids"]
                        perm[t]["y"].append(int(yv)); perm[t]["p0"].append(p0)
                        for k, key in ((1, "p1"), (5, "p5"), (10, "p10")):
                            mb = _mask_topk(bag_np, a, sid, k)
                            if mb:
                                mbt = {s: torch.from_numpy(v).to(device) for s, v in mb.items()}
                                perm[t][key].append(float(torch.sigmoid(model(mbt, temperature)[t])))
                            else:
                                perm[t][key].append(p0)

    # ---- fold/전체 metrics ----
    res = {}
    for t in BIN_TASKS:
        y = np.array(oof[t]["y"]); p = np.array(oof[t]["p"]); fo = np.array(oof[t]["fold"])
        ov = {"auroc": _safe(roc_auc_score, y, p), "auprc": _safe(average_precision_score, y, p),
              "mean_entropy": round(float(np.nanmean(oof[t]["ent"])), 3),
              "mean_top1": round(float(np.mean(oof[t]["t1"])), 3),
              "n_pos": int(y.sum()), "n": int(len(y))}
        per_fold = []
        for f in sorted(set(fo)):
            m = fo == f; yy = y[m]; pp = p[m]
            pred = (pp > 0.5).astype(int)
            per_fold.append({"fold": int(f), "pos": int(yy.sum()), "neg": int((yy == 0).sum()),
                             "auroc": _safe(roc_auc_score, yy, pp),
                             "auprc": _safe(average_precision_score, yy, pp),
                             "precision": round(float(precision_score(yy, pred, zero_division=0)), 3),
                             "recall": round(float(recall_score(yy, pred, zero_division=0)), 3),
                             "f1": round(float(f1_score(yy, pred, zero_division=0)), 3),
                             "mean_entropy": round(float(np.nanmean(np.array(oof[t]["ent"])[m])), 3)})
        ov["per_fold"] = per_fold
        res[t] = ov
    return res, oof, perm


def _safe(fn, y, p):
    y = np.asarray(y); p = np.asarray(p)
    return round(float(fn(y, p)), 3) if len(np.unique(y)) > 1 else None


def main():
    import torch
    from scipy.stats import pearsonr, spearmanr
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = "Malgun Gothic"; matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="ctranspath")
    ap.add_argument("--mags", default="10")
    ap.add_argument("--epochs", type=int, default=40)
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
    coh = sm[sm["fold"] >= 0].copy(); coh["patient_id"] = coh["patient_id"].astype(str)
    coh["ati_severity_n"] = (coh["task_ati_severity"] - 1) / 2.0
    bags = load_bags(coh["patient_id"].tolist(), idx, keep=None)
    coh = coh[coh["patient_id"].isin(bags.keys())]
    print(f"cohort={len(coh)} | chronic+={int((coh['task_chronic']==1).sum())} "
          f"immune+={int((coh['task_immune']==1).sum())}", flush=True)

    # ===== 실험 1/2/3/6/7: baseline(T=1, λ=0) 캡처 =====
    print("\n[baseline CV capture]", flush=True)
    res0, oof, perm = run_cv(bags, coh, embed_dim, device, args.epochs,
                             temperature=1.0, lambda_entropy=0.0, capture=True)

    # 실험1: fold csv
    rows1 = []
    for t in BIN_TASKS:
        for f in res0[t]["per_fold"]:
            rows1.append({"task": t, **f})
    pd.DataFrame(rows1).to_csv(OUTDIR / "diag_auprc_fold.csv", index=False, encoding="utf-8")

    # 실험2: 환자별 entropy csv
    rows2 = []
    for t in BIN_TASKS:
        for i in range(len(oof[t]["y"])):
            rows2.append({"task": t, "patient_id": oof[t]["pid"][i], "fold": oof[t]["fold"][i],
                          "label": oof[t]["y"][i], "pred": round(oof[t]["p"][i], 4),
                          "entropy": round(oof[t]["ent"][i], 4),
                          "correct": int((oof[t]["p"][i] > 0.5) == (oof[t]["y"][i] == 1))})
    df2 = pd.DataFrame(rows2)
    df2.to_csv(OUTDIR / "diag_entropy.csv", index=False, encoding="utf-8")
    ent_stats = {}
    for t in BIN_TASKS:
        d = df2[df2.task == t]
        ent_stats[t] = {"mean": round(d.entropy.mean(), 3), "std": round(d.entropy.std(), 3),
                        "min": round(d.entropy.min(), 3), "max": round(d.entropy.max(), 3),
                        "mean_pos": round(d[d.label == 1].entropy.mean(), 3),
                        "mean_neg": round(d[d.label == 0].entropy.mean(), 3),
                        "mean_entropy_correct": round(d[d.correct == 1].entropy.mean(), 3),
                        "mean_entropy_wrong": round(d[d.correct == 0].entropy.mean(), 3)}

    # 실험6: top-patch 의존도 csv
    rows6 = []
    for t in BIN_TASKS:
        rows6.append({"task": t, "top1_ratio": round(float(np.mean(oof[t]["t1"])), 3),
                      "top5_ratio": round(float(np.mean(oof[t]["t5"])), 3),
                      "top10_ratio": round(float(np.mean(oof[t]["t10"])), 3),
                      "collapse_suspect": bool(np.mean(oof[t]["t1"]) > 0.7)})
    pd.DataFrame(rows6).to_csv(OUTDIR / "diag_top_patch.csv", index=False, encoding="utf-8")

    # 실험3: fold mean-entropy vs fold AUPRC 상관 + 그림
    corr = {}
    fig, axes = plt.subplots(1, len(BIN_TASKS), figsize=(5 * len(BIN_TASKS), 4))
    for ax, t in zip(np.atleast_1d(axes), BIN_TASKS):
        pf = [f for f in res0[t]["per_fold"] if f["auprc"] is not None]
        e = [f["mean_entropy"] for f in pf]; a = [f["auprc"] for f in pf]
        if len(e) >= 3 and len(set(e)) > 1 and len(set(a)) > 1:
            sp = spearmanr(e, a).correlation; pe = pearsonr(e, a)[0]
        else:
            sp = pe = None
        corr[t] = {"spearman": None if sp is None else round(float(sp), 3),
                   "pearson": None if pe is None else round(float(pe), 3),
                   "n_folds": len(pf)}
        ax.scatter(e, a); ax.set_xlabel("fold mean entropy"); ax.set_ylabel("fold AUPRC")
        ax.set_title(f"{t}\nSpearman={corr[t]['spearman']} Pearson={corr[t]['pearson']}")
    fig.tight_layout(); fig.savefig(OUTDIR / "entropy_vs_auprc.png", dpi=110); plt.close(fig)

    # 실험7: permutation(top-k masking) — AUPRC 변화
    from sklearn.metrics import average_precision_score
    perm_res = {}
    for t in BIN_TASKS:
        y = perm[t]["y"]
        if len(set(y)) < 2:
            perm_res[t] = {"note": "단일클래스 fold합산 불가"}; continue
        base = _safe(average_precision_score, y, perm[t]["p0"])
        perm_res[t] = {"auprc_orig": base,
                       "auprc_mask_top1": _safe(average_precision_score, y, perm[t]["p1"]),
                       "auprc_mask_top5": _safe(average_precision_score, y, perm[t]["p5"]),
                       "auprc_mask_top10": _safe(average_precision_score, y, perm[t]["p10"])}

    # ===== 실험4: temperature sweep =====
    print("\n[temperature sweep]", flush=True)
    temp_sweep = {}
    for T in [1.0, 1.5, 2.0, 3.0]:
        r, _, _ = run_cv(bags, coh, embed_dim, device, args.epochs, temperature=T)
        temp_sweep[f"T={T}"] = {t: {"auroc": r[t]["auroc"], "auprc": r[t]["auprc"],
                                    "mean_entropy": r[t]["mean_entropy"]} for t in BIN_TASKS}
        print(f"  T={T}: " + " | ".join(f"{t} AUPRC={r[t]['auprc']} ent={r[t]['mean_entropy']}"
                                        for t in ["immune", "chronic"]), flush=True)
    (OUTDIR / "temperature_sweep.json").write_text(
        json.dumps(temp_sweep, indent=2, ensure_ascii=False), encoding="utf-8")

    # ===== 실험5: entropy regularization sweep =====
    print("\n[entropy reg sweep]", flush=True)
    ent_sweep = {}
    for lam in [0.0, 0.001, 0.005, 0.01, 0.05]:
        r, _, _ = run_cv(bags, coh, embed_dim, device, args.epochs, lambda_entropy=lam)
        ent_sweep[f"lambda={lam}"] = {t: {"auroc": r[t]["auroc"], "auprc": r[t]["auprc"],
                                          "mean_entropy": r[t]["mean_entropy"]} for t in BIN_TASKS}
        print(f"  λ={lam}: " + " | ".join(f"{t} AUPRC={r[t]['auprc']} ent={r[t]['mean_entropy']}"
                                          for t in ["immune", "chronic"]), flush=True)
    (OUTDIR / "entropy_sweep.json").write_text(
        json.dumps(ent_sweep, indent=2, ensure_ascii=False), encoding="utf-8")

    # ===== 최종 판정 =====
    diagnosis = {"baseline": {t: {k: res0[t][k] for k in ("auroc", "auprc", "mean_entropy",
                                                          "mean_top1", "n_pos", "n")}
                              for t in BIN_TASKS},
                 "entropy_stats": ent_stats, "entropy_auprc_corr": corr,
                 "top_patch": {r["task"]: r for r in rows6},
                 "permutation": perm_res, "verdict": {}}
    for t in ["immune", "chronic"]:
        ev = []
        collapse = res0[t]["mean_top1"] > 0.7
        # Case D: top1 제거 시 AUPRC 붕괴
        pr = perm_res.get(t, {})
        d_drop = (pr.get("auprc_orig") and pr.get("auprc_mask_top1") is not None and
                  pr["auprc_orig"] - pr["auprc_mask_top1"] > 0.15)
        # Case A: temperature/entropy로 entropy↑ 시 AUPRC↑
        t_aurpc = [temp_sweep[k][t]["auprc"] for k in temp_sweep if temp_sweep[k][t]["auprc"] is not None]
        t_ent = [temp_sweep[k][t]["mean_entropy"] for k in temp_sweep]
        a_improves = len(t_aurpc) >= 2 and max(t_aurpc) - (temp_sweep["T=1.0"][t]["auprc"] or 0) > 0.05
        # Case C: fold AUPRC가 pos수와 강한 상관 / 분산 큼
        pf = [f for f in res0[t]["per_fold"] if f["auprc"] is not None]
        auprc_std = round(float(np.std([f["auprc"] for f in pf])), 3) if pf else None
        if collapse:
            ev.append(f"Top1 ratio {res0[t]['mean_top1']}>0.7 → collapse 의심")
        if d_drop:
            ev.append("Top1 masking 시 AUPRC 급락 → 단일패치 의존(Case D)")
        if a_improves:
            ev.append("temperature↑(entropy↑) 시 AUPRC↑ → collapse 기여(Case A)")
        if not collapse and not d_drop:
            ev.append(f"Top1 의존 낮음 → collapse 아님(Case E). fold AUPRC std={auprc_std}, "
                      f"pos/fold~{res0[t]['n_pos']/max(len(pf),1):.1f} → 표본/불균형 영향(Case B/C)")
        diagnosis["verdict"][t] = {"collapse_suspect": collapse, "top1_masking_drop": d_drop,
                                   "temp_improves_auprc": a_improves, "fold_auprc_std": auprc_std,
                                   "evidence": ev}
    (OUTDIR / "collapse_diagnosis.json").write_text(
        json.dumps(diagnosis, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n저장: {OUTDIR}")
    for t in ["immune", "chronic"]:
        print(f"[{t}] {diagnosis['verdict'][t]['evidence']}")


if __name__ == "__main__":
    main()
