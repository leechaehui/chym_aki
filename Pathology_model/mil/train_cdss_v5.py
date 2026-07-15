import argparse
import sys
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path("c:/team/chym_aki/Pathology_model")))
from mil.train import ROOT, SEED, load_bags, seed_all
from mil.cdss_paths import p as _p
from mil.cdss_v5_model import CDSSv5Model

def nll_loss(y_true, mu, log_var):
    """Log-likelihood loss for aleatoric uncertainty prediction."""
    precision = torch.exp(-log_var)
    return (precision * (y_true - mu)**2 + log_var).mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="ctranspath")
    ap.add_argument("--mags", default="10")
    ap.add_argument("--tag", default="vfinal_97")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    # AIS(head1) 타깃 소스: kdigo=기존 split_manifest task_ati_severity(임상 proxy),
    # descriptor=descriptor_labels.csv tubular_injury_pct(실제 병리 ATI %, /100 스케일).
    ap.add_argument("--ais-source", default="kdigo", choices=["kdigo", "descriptor"])
    args = ap.parse_args()

    seed_all(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb_dir = _p("embeddings") / args.encoder
    idx = pd.read_csv(emb_dir / "index.csv", dtype={"magnification": str})
    embed_dim = int(idx["embed_dim"].iloc[0])
    msel = [m.strip() for m in args.mags.split(",")]
    idx = idx[idx["magnification"].isin(msel)]

    sm = pd.read_csv(ROOT / "split_manifest.csv")
    sm = sm.drop_duplicates("patient_id")
    p_all = sm["patient_id"].astype(str).tolist()

    # We need all available stains for CDSS v5
    keep_stains = ["HE", "PAS", "MT", "SILVER"]
    bags = load_bags(p_all, idx, keep=set(keep_stains))

    # Filter patients that have at least one base stain
    valid_p = []
    for p in p_all:
        if p in bags:
            b = bags[p]
            if any(b.get(s) is not None and b[s].shape[0] > 0 for s in ["HE", "PAS", "MT"]):
                valid_p.append(p)
                
    sm = sm[sm["patient_id"].astype(str).isin(valid_p)]

    # Map labels to pseudo-continuous targets [0, 1] for NLL
    def map_ati(val):
        if pd.isna(val): return float('nan')
        if val == 1: return 0.0
        if val == 3: return 0.5
        if val == 5: return 1.0
        return float('nan')

    # AIS(head1) 타깃 — 소스에 따라 분기. INS/CDS 파이프라인은 불변.
    if args.ais_source == "descriptor":
        # 실제 병리 ATI: descriptor_labels.csv 의 tubular_injury_pct(0–100) → /100 으로 [0,1] 스케일.
        # (코드루트 artifacts; descriptor_labels.csv 는 D 마이그레이션 대상이 아니라 코드 옆에 있음)
        ART = Path(__file__).resolve().parent.parent / "artifacts" / "descriptor_labels.csv"
        desc = pd.read_csv(ART)
        desc["patient_id"] = desc["patient_id"].astype(str)
        desc = desc.drop_duplicates("patient_id")
        ati = pd.to_numeric(desc.set_index("patient_id")["tubular_injury_pct"], errors="coerce")
        ati = ati.where(ati <= 100)  # 0–100 범위 외/sentinel 제거
        sm["target_ais"] = (sm["patient_id"].astype(str).map(ati) / 100.0).to_numpy()
        print(f"[AIS] source=descriptor(tubular_injury_pct/100)  non-null={int(sm['target_ais'].notna().sum())}/{len(sm)}", flush=True)
    else:
        sm["target_ais"] = sm["task_ati_severity"].apply(map_ati)
        print(f"[AIS] source=kdigo(task_ati_severity proxy)  non-null={int(sm['target_ais'].notna().sum())}/{len(sm)}", flush=True)
    sm["target_cds"] = sm["task_chronic"].astype(float)
    sm["target_ins"] = sm["task_immune"].astype(float)
    def map_kdigo(val):
        if pd.isna(val): return float('nan')
        v = str(val).lower()
        if '3' in v: return 3.0
        if '2' in v: return 2.0
        if '1' in v: return 1.0
        return 0.0

    sm["target_kdigo"] = sm["kdigo_stage"].apply(map_kdigo)

    folds = sorted(sm["fold"].dropna().unique())
    
    # Store OOF predictions
    oof_res = []

    for fold in folds:
        print(f"--- FOLD {fold} ---")
        train_df = sm[sm["fold"] != fold]
        val_df = sm[sm["fold"] == fold]
        
        model = CDSSv5Model(in_dim=embed_dim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        
        for ep in range(args.epochs):
            model.train()
            train_loss = 0.0
            
            for _, r in train_df.sample(frac=1, random_state=SEED+ep).iterrows():
                pid = str(r["patient_id"])
                bag = {s: torch.from_numpy(v).to(device) for s,v in bags[pid].items() if v is not None}
                
                out = model(bag, mc_dropout=False)
                
                loss = torch.tensor(0.0, device=device)
                
                if pd.notna(r["target_ais"]):
                    loss += nll_loss(torch.tensor(r["target_ais"], device=device), out["ais"], out["log_var_ais"])
                if pd.notna(r["target_cds"]):
                    loss += nll_loss(torch.tensor(r["target_cds"], device=device), out["cds"], out["log_var_cds"])
                if pd.notna(r["target_ins"]):
                    loss += nll_loss(torch.tensor(r["target_ins"], device=device), out["ins"], out["log_var_ins"])
                
                # Weak proxy PSI loss (Spearman approximation with KDIGO if available)
                # In single sample SGD, we can just use MSE with KDIGO/5 as a very weak anchoring
                if pd.notna(r["target_kdigo"]):
                    proxy_target = torch.tensor(r["target_kdigo"] / 5.0, device=device)
                    loss += F.mse_loss(out["psi"], proxy_target) * 0.1
                    
                if "kl_penalty" in out:
                    loss += out["kl_penalty"] * 0.01
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                train_loss += loss.item()
                
            # Validation
            model.eval()
            with torch.no_grad():
                val_unc = []
                val_out_cache = []
                for _, r in val_df.iterrows():
                    pid = str(r["patient_id"])
                    bag = {s: torch.from_numpy(v).to(device) for s,v in bags[pid].items() if v is not None}
                    out = model(bag, mc_dropout=False)
                    val_unc.append(out["aleatoric_unc"].item())
                    if "w_ais" in out:
                        val_out_cache.append(out)
                    
                    if ep == args.epochs - 1:
                        # Save OOF
                        # To get MC Dropout Epistemic Uncertainty
                        mc_preds = []
                        for _ in range(10): # 10 MC passes
                            mc_out = model(bag, mc_dropout=True)
                            mc_preds.append(mc_out["psi"].item())
                        epistemic_unc = np.var(mc_preds)
                        
                        total_unc = epistemic_unc + out["aleatoric_unc"].item()
                        
                        oof_res.append({
                            "patient_id": pid,
                            "fold": fold,
                            "pred_ais": out["ais"].item(),
                            "log_var_ais": out["log_var_ais"].item(),
                            "target_ais": float(r["target_ais"]) if pd.notna(r["target_ais"]) else np.nan,
                            "pred_cds": out["cds"].item(),
                            "pred_ins": out["ins"].item(),
                            "pred_psi": out["psi"].item(),
                            "aleatoric_unc": out["aleatoric_unc"].item(),
                            "epistemic_unc": epistemic_unc,
                            "silver_consistency": out["silver_consistency"].item(),
                            "total_unc": total_unc
                        })
            
            w_ais_mean = np.mean([out["w_ais"].cpu().numpy() for out in val_out_cache], axis=0) if val_out_cache else [0,0,0]
            print(f"[Fold {fold} Ep {ep}] Loss: {train_loss/len(train_df):.4f} | Avg Val Unc: {np.mean(val_unc):.4f}")
            print(f"  [Gate AIS] HE: {w_ais_mean[0]:.2f} | PAS: {w_ais_mean[1]:.2f} | MT: {w_ais_mean[2]:.2f}")
            
        # Save model weights for this fold
        res_dir = ROOT / "results" / "cdss_v5"
        res_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), res_dir / f"cdss_v5_fold{fold}_{args.tag}.pt")

    # Save OOF
    oof_df = pd.DataFrame(oof_res)
    oof_df.to_csv(res_dir / f"oof_{args.tag}.csv", index=False)
    print(f"OOF saved to {res_dir / f'oof_{args.tag}.csv'}")

if __name__ == "__main__":
    main()
