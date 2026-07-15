"""
diag_gradient_flow.py
네트워크 깊이에 따른 Gradient Flow Map을 추출합니다.
Projection -> Attention -> Slide Vector(z) -> Fusion -> Head
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path("c:/team/chym_aki/Pathology_model")))
LOCAL_ROOT = Path("c:/team/chym_aki")

import mil.train
mil.train.ROOT = LOCAL_ROOT

from mil.train import CLF_TASKS, COL, SEED, load_bags, seed_all
from mil.model import StainAwareMIL, STAINS

OUTDIR = LOCAL_ROOT / "Pathology_model/results/15_diag_gradient_flow"

def get_norm(tensor):
    return float(torch.norm(tensor).detach().cpu().numpy()) if tensor is not None else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="ctranspath")
    ap.add_argument("--mags", default="10")
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    
    seed_all(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data Loading
    idx = pd.read_csv(LOCAL_ROOT / "data/embeddings" / args.encoder / "index.csv", dtype={"magnification": str})
    msel = [m.strip() for m in args.mags.split(",")]
    idx = idx[idx["magnification"].isin(msel)]
    embed_dim = int(idx["embed_dim"].iloc[0])
    
    sm = pd.read_csv(LOCAL_ROOT / "split_manifest.csv").drop_duplicates("patient_id")
    coh = sm[sm["fold"] >= 0].copy(); coh["patient_id"] = coh["patient_id"].astype(str)
    coh["ati_severity_n"] = (coh["task_ati_severity"] - 1) / 2.0
    bags = load_bags(coh["patient_id"].tolist(), idx, keep=None)
    coh = coh[coh["patient_id"].isin(bags.keys())]

    fold = 0
    tr = coh[coh["fold"] != fold]
    
    print(f"Tracking Gradient Flow on {len(tr)} training patients for {args.epochs} epochs...", flush=True)
    
    model = StainAwareMIL(in_dim=embed_dim, silver_mode="off").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    def pw(col):
        yy = tr[col].dropna(); n1 = (yy == 1).sum(); n0 = (yy == 0).sum()
        return torch.tensor(n0 / max(n1, 1), device=device, dtype=torch.float32)
    PW = {t: pw(COL[t]) for t in CLF_TASKS}

    epoch_metrics = []

    for ep in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0
        
        # Accumulate gradients manually for the whole epoch
        grad_norms = {
            "Fusion": 0, "Head": 0,
            "HE_Proj": 0, "HE_Attn": 0, "HE_Z": 0,
            "PAS_Proj": 0, "PAS_Attn": 0, "PAS_Z": 0,
            "MT_Proj": 0, "MT_Attn": 0, "MT_Z": 0
        }
        grad_count = 0
        
        for r in tr.sample(frac=1, random_state=SEED + ep).itertuples():
            bag = {s: torch.from_numpy(v).to(device) for s, v in bags[r.patient_id].items()}
            
            # --- CUSTOM FORWARD TO HOOK Z ---
            # Instead of model(bag), we do a partial forward to hook Z
            stain_vecs, present = [], []
            z_tensors = {}
            for s in STAINS:
                if s in bag and bag[s].shape[0] > 0:
                    z, a = model.encoders[s](bag[s])
                    z.retain_grad()  # Hook Z!
                    z_tensors[s] = z
                    stain_vecs.append(z)
                    present.append(s)
            
            if not stain_vecs: continue
            
            Z = torch.stack(stain_vecs, 0)
            fa = model.fw(torch.tanh(model.fV(Z)) * torch.sigmoid(model.fU(Z)))
            fa = torch.softmax(fa, dim=0)
            fused = (fa * Z).sum(0)
            
            present_all = [s for s in STAINS if s in bag and bag[s].shape[0] > 0]
            mask = torch.tensor([1.0 if s in present_all else 0.0 for s in STAINS], device=fused.device)
            avail = torch.cat([mask, mask.sum().unsqueeze(0)])
            rep = torch.cat([fused, avail])
            
            out = {t: getattr(model, f"head_{t}")(rep).squeeze(-1) for t in CLF_TASKS}
            
            loss = 0.0; nt = 0
            for t in CLF_TASKS:
                yv = getattr(r, COL[t])
                if not (yv != yv):
                    loss = loss + F.binary_cross_entropy_with_logits(
                        out[t], torch.tensor(float(yv), device=device), pos_weight=PW[t]); nt += 1
            
            if nt:
                opt.zero_grad(); loss.backward(); opt.step()
                loss_sum += float(loss)
                
                # Accumulate grad norms
                grad_norms["Fusion"] += get_norm(model.fw.weight.grad)
                grad_norms["Head"] += get_norm(model.head_immune.weight.grad)
                
                for s in STAINS:
                    if s in z_tensors:
                        enc = model.encoders[s]
                        grad_norms[f"{s}_Proj"] += get_norm(enc.proj[0].weight.grad)
                        grad_norms[f"{s}_Attn"] += get_norm(enc.w.weight.grad)
                        grad_norms[f"{s}_Z"] += get_norm(z_tensors[s].grad)
                        
                grad_count += 1
                
        # Average the metrics
        ep_stat = {"epoch": ep, "loss": loss_sum / max(1, len(tr))}
        for k in grad_norms:
            ep_stat[k] = grad_norms[k] / max(1, grad_count)
            
        epoch_metrics.append(ep_stat)
        
        if ep % 5 == 0 or ep == 1:
            print(f"Epoch {ep} Loss: {loss_sum:.4f} | Fusion: {ep_stat['Fusion']:.4f} | HE_Z: {ep_stat['HE_Z']:.5f} | MT_Z: {ep_stat['MT_Z']:.5f}", flush=True)

    df_ep = pd.DataFrame(epoch_metrics)
    df_ep.to_csv(OUTDIR / "gradient_flow_dynamics.csv", index=False)
    
    # Plot Trajectories
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, s in enumerate(["HE", "PAS", "MT"]):
        ax = axes[idx]
        ax.plot(df_ep["epoch"], df_ep["Head"], label="Head", linestyle=":")
        ax.plot(df_ep["epoch"], df_ep["Fusion"], label="Fusion", linestyle=":")
        ax.plot(df_ep["epoch"], df_ep[f"{s}_Z"], label=f"{s} Slide Z")
        ax.plot(df_ep["epoch"], df_ep[f"{s}_Attn"], label=f"{s} Attention W")
        ax.plot(df_ep["epoch"], df_ep[f"{s}_Proj"], label=f"{s} Proj W")
        ax.set_title(f"{s} Gradient Flow")
        ax.set_xlabel("Epoch")
        ax.set_yscale("log")
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTDIR / "gradient_flow_map.png", dpi=150)
    plt.close(fig)
    
    print(f"Gradient Flow mapping completed. Outputs saved to {OUTDIR}")

if __name__ == "__main__":
    main()
