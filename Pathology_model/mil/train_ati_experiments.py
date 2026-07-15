"""
ATI 헤드 비교 실험 — 동일 CLAM-Lite 백본(CDSSv5 코어: proj + attention pooling), 헤드만 교체.
백본 교체가 아니라 '회귀 vs CORAL ordinal vs CORAL+descriptor 멀티태스크' 비교.

모드:
  regression : AIS = Linear(d,1) MSE(target = tubular_injury_pct/100)
  coral      : AIS = Linear(d,K-1) CORAL ordinal (5-class, edges [5,20,40,60])
  coral_mt   : coral + 보조 이진헤드(ATI 세부 descriptor) — ATI vs 염증 분리 강제
               (necrosis 제외: 학습코호트 양성 2개)

데이터:
  embeddings 40x(ctranspath, HE+PAS+MT) → bag concat → 공유 proj/attention → pooled → 헤드들
  라벨: descriptor_labels.csv (tubular_injury_pct + ti_* + tubulitis)
  fold: split_manifest.csv (기존 ati40/vfind40 와 동일 split)

평가(OOF): ATI Spearman(연속 GT 대비) · QWK(5-class) · MAE(class) · 보조 AUROC.
사용: python mil/train_ati_experiments.py
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("c:/team/chym_aki/Pathology_model")))
from mil.cdss_paths import p as _p
from mil.train import ROOT, SEED, load_bags, seed_all

STAINS = ["HE", "PAS", "MT"]
MAG = "40"
EDGES = [5, 20, 40, 60]              # 0:0-5 1:5-20 2:20-40 3:40-60 4:60+
KBINS = len(EDGES) + 1               # 5
AUX = ["ti_simplification", "ti_cell_sloughing", "ti_detachment_denudation", "tubulitis"]  # necrosis 제외(pos=2)
ART = Path(__file__).resolve().parent.parent / "artifacts"
OUT = Path("C:/team/chym_aki/data/eval/attention")
OUT.mkdir(parents=True, exist_ok=True)


def to_class(v):
    if v != v:
        return np.nan
    return int(sum(v > e for e in EDGES))


def main():
    import argparse
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.metrics import cohen_kappa_score, roc_auc_score
    from scipy.stats import spearmanr

    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-subdir", default="ctranspath",
                    help="임베딩 디렉터리(예: ctranspath_s512 = 512 입력 임베딩)")
    ap.add_argument("--tag", default="", help="출력 파일 접미사(예: _s512)")
    ap.add_argument("--fold-file", default=str(ART / "ati_cohort_folds.csv"),
                    help="cohort+fold 동결 파일. 있으면 그대로 사용(재현성), 없으면 생성·저장")
    args = ap.parse_args()

    seed_all(SEED)
    device = "cpu"

    # --- 라벨 ---
    desc = pd.read_csv(ART / "descriptor_labels.csv").drop_duplicates("patient_id")
    desc["patient_id"] = desc["patient_id"].astype(str)
    sm = pd.read_csv(ROOT / "split_manifest.csv").drop_duplicates("patient_id")
    sm["patient_id"] = sm["patient_id"].astype(str)
    fold = sm.set_index("patient_id")["fold"].to_dict()

    # --- 40x 임베딩 백 ---
    idx = pd.read_csv(_p("embeddings") / args.emb_subdir / "index.csv", dtype={"magnification": str})
    embed_dim = int(idx["embed_dim"].iloc[0])
    idx40 = idx[idx["magnification"] == MAG]

    d = desc.copy()
    d["ati"] = pd.to_numeric(d["tubular_injury_pct"], errors="coerce").where(lambda s: s <= 100)
    d["ati_t"] = d["ati"] / 100.0
    d["ati_cls"] = d["ati"].map(to_class)
    for a in AUX:
        d[a] = pd.to_numeric(d[a], errors="coerce")
    # 코호트 = ATI 라벨 보유 ∩ 40x 임베딩 보유 (split_manifest fold 비의존 → 신규 환자 포함)
    d = d[d["ati"].notna()].reset_index(drop=True)
    bags = load_bags(d["patient_id"].tolist(), idx40, keep=set(STAINS))
    d = d[d["patient_id"].isin(bags)].reset_index(drop=True)
    # fold: 재현성 위해 동결 split 파일 우선 사용(224·512 동일 분할 보장). 없으면 StratifiedKFold로 생성·저장.
    fold_file = Path(args.fold_file)
    if fold_file.exists():
        fmap = pd.read_csv(fold_file, dtype={"patient_id": str}).set_index("patient_id")["fold"].to_dict()
        d = d[d["patient_id"].isin(fmap)].reset_index(drop=True)
        d["fold"] = d["patient_id"].map(fmap).astype(int)
        print(f"[fold] 동결 split 사용: {fold_file.name} (N={len(d)})", flush=True)
    else:
        from sklearn.model_selection import StratifiedKFold, KFold
        d["fold"] = -1
        try:
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(d, d["ati_cls"])
        except Exception:
            splitter = KFold(n_splits=5, shuffle=True, random_state=SEED).split(d)
        for fo, (_, va) in enumerate(splitter):
            d.loc[d.index[va], "fold"] = fo
        d[["patient_id", "fold", "ati_cls"]].to_csv(fold_file, index=False)
        print(f"[fold] 신규 split 생성·저장: {fold_file.name} (N={len(d)})", flush=True)
    print(f"cohort N={len(d)}  ATI 5-class분포={d['ati_cls'].value_counts().sort_index().to_dict()}  "
          f"fold분포={d['fold'].value_counts().sort_index().to_dict()}", flush=True)

    def bagtensor(pid):
        mats = [torch.from_numpy(bags[pid][s]) for s in STAINS
                if bags[pid].get(s) is not None and bags[pid][s].shape[0] > 0]
        return torch.cat(mats, 0).float().to(device)

    # --- 백본(공유): CDSSv5 코어와 동일 구조 ---
    class Backbone(nn.Module):
        def __init__(self, in_dim, dim=256, dropout=0.25):
            super().__init__()
            self.proj = nn.Sequential(nn.Linear(in_dim, dim), nn.ReLU(), nn.Dropout(dropout))
            self.attn = nn.Linear(dim, 1)
        def forward(self, x):
            h = self.proj(x)
            a = torch.softmax(self.attn(h), 0)
            return (a * h).sum(0)            # pooled (dim,)

    class ATIModel(nn.Module):
        def __init__(self, in_dim, mode):
            super().__init__()
            self.mode = mode
            self.bb = Backbone(in_dim)
            dim = 256
            if mode == "regression":
                self.head = nn.Linear(dim, 1)
            else:
                self.head = nn.Linear(dim, KBINS - 1)     # CORAL K-1 logit
            if mode == "coral_mt":
                self.aux = nn.ModuleDict({a: nn.Linear(dim, 1) for a in AUX})
        def forward(self, x):
            z = self.bb(x)
            o = {"ati": self.head(z)}
            if self.mode == "coral_mt":
                o["aux"] = {a: self.aux[a](z) for a in AUX}
            return o

    def run_mode(mode, epochs=40):
        oof = []
        for fo in [0, 1, 2, 3, 4]:
            tr = d[d["fold"] != fo]; va = d[d["fold"] == fo]
            torch.manual_seed(SEED + fo)
            model = ATIModel(embed_dim, mode).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
            for ep in range(epochs):
                model.train()
                for _, r in tr.sample(frac=1, random_state=SEED + ep).iterrows():
                    x = bagtensor(r["patient_id"]); out = model(x)
                    loss = torch.tensor(0.0, device=device)
                    if mode == "regression":
                        loss = loss + F.mse_loss(out["ati"].squeeze(), torch.tensor(float(r["ati_t"]), device=device))
                    else:
                        yc = int(r["ati_cls"])
                        lv = torch.tensor([1.0 if yc > k else 0.0 for k in range(KBINS - 1)], device=device)
                        loss = loss + F.binary_cross_entropy_with_logits(out["ati"], lv)
                    if mode == "coral_mt":
                        for a in AUX:
                            yv = r[a]
                            if pd.notna(yv):
                                loss = loss + 0.5 * F.binary_cross_entropy_with_logits(
                                    out["aux"][a].squeeze(), torch.tensor(float(yv), device=device))
                    opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                for _, r in va.iterrows():
                    x = bagtensor(r["patient_id"]); out = model(x)
                    rec = {"patient_id": r["patient_id"], "fold": fo,
                           "ati_true": r["ati"], "ati_cls": int(r["ati_cls"])}
                    if mode == "regression":
                        mu = float(out["ati"].squeeze()) * 100
                        rec["pred_cont"] = mu; rec["pred_cls"] = to_class(mu)
                    else:
                        probs = torch.sigmoid(out["ati"]).numpy().reshape(-1)
                        rec["pred_cont"] = float(probs.sum())          # 0..K-1 연속 ordinal score
                        rec["pred_cls"] = int((probs > 0.5).sum())
                    if mode == "coral_mt":
                        for a in AUX:
                            rec[f"p_{a}"] = float(torch.sigmoid(out["aux"][a].squeeze()))
                            rec[f"y_{a}"] = r[a]
                    oof.append(rec)
        return pd.DataFrame(oof)

    results = {}
    metrics = []
    for mode in ["regression", "coral", "coral_mt"]:
        print(f"\n===== MODE: {mode} =====", flush=True)
        oof = run_mode(mode)
        oof.to_csv(OUT / f"ati_exp_oof_{mode}{args.tag}.csv", index=False)
        results[mode] = oof
        sp = spearmanr(oof["ati_true"], oof["pred_cont"]).correlation
        qwk = cohen_kappa_score(oof["ati_cls"], oof["pred_cls"], weights="quadratic",
                                labels=list(range(KBINS)))
        mae_c = float(np.mean(np.abs(oof["ati_cls"] - oof["pred_cls"])))
        row = {"mode": mode, "N": len(oof), "Spearman_vs_true%": round(sp, 3),
               "QWK_5class": round(qwk, 3), "MAE_class": round(mae_c, 3)}
        metrics.append(row)
        print(row, flush=True)
        if mode == "coral_mt":
            aux_auroc = {}
            for a in AUX:
                y = pd.to_numeric(oof[f"y_{a}"], errors="coerce"); p = oof[f"p_{a}"]
                m = y.notna()
                if m.sum() >= 3 and y[m].nunique() > 1:
                    aux_auroc[a] = round(float(roc_auc_score(y[m], p[m])), 3)
                else:
                    aux_auroc[a] = None
            print("AUX AUROC:", aux_auroc, flush=True)
            row["aux_auroc"] = aux_auroc

    mdf = pd.DataFrame(metrics)
    mdf.to_csv(OUT / f"ati_exp_compare{args.tag}.csv", index=False)
    (OUT / f"ati_exp_compare{args.tag}.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n===== COMPARISON =====", flush=True)
    print(mdf.to_string(index=False), flush=True)
    print("saved ->", OUT / "ati_exp_compare.csv", flush=True)


if __name__ == "__main__":
    main()
