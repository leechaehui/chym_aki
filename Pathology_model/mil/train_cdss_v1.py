import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path("c:/team/chym_aki/Pathology_model")))
from mil.train import ROOT, SEED, load_bags, seed_all
from mil.cdss_paths import p as _p
from mil.cdss_v1_model import CDSSDeploymentModel

def map_ati(val):
    if pd.isna(val): return float('nan')
    if val == 1: return 0.0
    if val == 3: return 0.5
    if val == 5: return 1.0
    return float('nan')

def map_kdigo(val):
    if pd.isna(val): return float('nan')
    v = str(val).lower()
    if '3' in v: return 1.0
    if '2' in v: return 0.66
    if '1' in v: return 0.33
    return 0.0

def main():
    seed_all(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    emb_dir = _p("embeddings") / "ctranspath"
    idx = pd.read_csv(emb_dir / "index.csv", dtype={"magnification": str})
    embed_dim = int(idx["embed_dim"].iloc[0])
    idx = idx[idx["magnification"] == "10"]
    
    sm = pd.read_csv(ROOT / "split_manifest.csv").drop_duplicates("patient_id")
    p_all = sm["patient_id"].astype(str).tolist()
    
    bags = load_bags(p_all, idx, keep=set(["HE", "PAS", "MT"]))
    
    sm["target_ais"] = sm["task_ati_severity"].apply(map_ati)
    sm["target_cds"] = sm["task_chronic"].astype(float)
    sm["target_ins"] = sm["task_immune"].astype(float)
    sm["target_kdigo"] = sm["kdigo_stage"].apply(map_kdigo)
    
    valid_p = [p for p in p_all if p in bags and any(bags[p].get(s) is not None for s in ["HE", "PAS", "MT"])]
    sm = sm[sm["patient_id"].astype(str).isin(valid_p)]
    
    model = CDSSDeploymentModel(in_dim=embed_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion_mse = nn.MSELoss()
    
    print("Training CDSS v1.1 Production Model (Reference Only Architecture)...")
    for ep in range(15): # 15 epochs for quick transfer
        model.train()
        train_loss = 0.0
        
        for _, r in sm.sample(frac=1, random_state=SEED+ep).iterrows():
            pid = str(r["patient_id"])
            mats = [torch.from_numpy(v) for s, v in bags[pid].items() if v is not None and v.shape[0] > 0]
            if not mats: continue
            
            x = torch.cat(mats, dim=0).to(device) # (N, dim)
            out = model(x)
            
            loss = torch.tensor(0.0, device=device)
            if pd.notna(r["target_ais"]):
                loss += criterion_mse(out["ais_prob"].squeeze(), torch.tensor(r["target_ais"], device=device, dtype=torch.float32))
            if pd.notna(r["target_cds"]):
                loss += criterion_mse(out["cds_prob"].squeeze(), torch.tensor(r["target_cds"], device=device, dtype=torch.float32))
            if pd.notna(r["target_ins"]):
                loss += criterion_mse(out["ins_prob"].squeeze(), torch.tensor(r["target_ins"], device=device, dtype=torch.float32))
                
            # Calibrated Risk targeting KDIGO
            if pd.notna(r["target_kdigo"]):
                loss += criterion_mse(out["calibrated_risk_prob"].squeeze(), torch.tensor(r["target_kdigo"], device=device, dtype=torch.float32))
                
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            
        print(f"Epoch {ep+1} Loss: {train_loss/len(sm):.4f}")
        
    res_dir = ROOT / "results" / "cdss_v1"
    res_dir.mkdir(parents=True, exist_ok=True)
    out_path = res_dir / "cdss_v1.1_final.pt"
    torch.save(model.state_dict(), out_path)
    print(f"v1.1 Model Weights saved to {out_path}")

if __name__ == "__main__":
    main()
