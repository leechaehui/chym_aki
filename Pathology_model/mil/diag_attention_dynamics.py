"""
diag_attention_dynamics.py
Attention Network의 실제 학습 여부를 검증하기 위해
가중치 노름(Weight norm), 그래디언트 노름(Grad norm), 에폭별 Entropy 및 Raw Logit STD를 추적합니다.
"""

import argparse
import sys
import os
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

# Patch ROOT BEFORE importing load_bags
import mil.train
mil.train.ROOT = LOCAL_ROOT

from mil.train import CLF_TASKS, COL, SEED, load_bags, seed_all
from mil.model import StainAwareMIL, STAINS

OUTDIR = LOCAL_ROOT / "Pathology_model/results/13_diag_attention_dynamics"

# Hook storage
raw_logits_cache = {}

def get_hook(name):
    def hook(module, input, output):
        raw_logits_cache[name] = output.detach().cpu().numpy()
    return hook

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

    # Use fold 0 as validation, others as train
    fold = 0
    tr = coh[coh["fold"] != fold]; va = coh[coh["fold"] == fold]
    
    print(f"Training on {len(tr)} patients, Validating on {len(va)} patients for {args.epochs} epochs...", flush=True)
    
    model = StainAwareMIL(in_dim=embed_dim, silver_mode="off").to(device)
    
    # Register Hooks for validation Raw Logits
    for s in STAINS:
        model.encoders[s].w.register_forward_hook(get_hook(f"{s}_patch_raw"))
        
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    def pw(col):
        yy = tr[col].dropna(); n1 = (yy == 1).sum(); n0 = (yy == 0).sum()
        return torch.tensor(n0 / max(n1, 1), device=device, dtype=torch.float32)
    PW = {t: pw(COL[t]) for t in CLF_TASKS}

    epoch_metrics = []
    val_metrics = []
    
    val_checkpoints = [1, 2, 5, 10, 20, 30, 50]  # 1-indexed

    for ep in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0
        
        # Accumulate gradients manually for the whole epoch
        grad_norms = {s: {"w": 0, "U": 0, "V": 0} for s in STAINS}
        grad_count = 0
        
        for r in tr.sample(frac=1, random_state=SEED + ep).itertuples():
            bag = {s: torch.from_numpy(v).to(device) for s, v in bags[r.patient_id].items()}
            out = model(bag); loss = 0.0; nt = 0
            for t in CLF_TASKS:
                yv = getattr(r, COL[t])
                if not (yv != yv):
                    loss = loss + F.binary_cross_entropy_with_logits(
                        out[t], torch.tensor(float(yv), device=device), pos_weight=PW[t]); nt += 1
            if nt:
                opt.zero_grad(); loss.backward(); opt.step()
                loss_sum += float(loss)
                
                # Accumulate grad norm
                for s in STAINS:
                    if s in bag and bag[s].shape[0] > 0:
                        enc = model.encoders[s]
                        grad_norms[s]["w"] += get_norm(enc.w.weight.grad)
                        grad_norms[s]["U"] += get_norm(enc.U.weight.grad)
                        grad_norms[s]["V"] += get_norm(enc.V.weight.grad)
                grad_count += 1
                
        # Average the metrics
        ep_stat = {"epoch": ep, "loss": loss_sum / max(1, len(tr))}
        for s in STAINS:
            enc = model.encoders[s]
            ep_stat[f"{s}_W_norm"] = get_norm(enc.w.weight) + get_norm(enc.U.weight) + get_norm(enc.V.weight)
            ep_stat[f"{s}_grad_norm"] = (grad_norms[s]["w"] + grad_norms[s]["U"] + grad_norms[s]["V"]) / max(1, grad_count)
            
        epoch_metrics.append(ep_stat)
        
        if ep % 5 == 0 or ep == 1 or ep == 2:
            print(f"Epoch {ep} Loss: {loss_sum:.4f} | HE W_norm: {ep_stat['HE_W_norm']:.4f} | HE grad: {ep_stat['HE_grad_norm']:.6f}", flush=True)
            
        # Validation Checkpoint
        if ep in val_checkpoints:
            print(f"--- Running Validation at Epoch {ep} ---", flush=True)
            model.eval()
            with torch.no_grad():
                for r in va.itertuples():
                    bag = {s: torch.from_numpy(v).to(device) for s, v in bags[r.patient_id].items()}
                    out = model(bag)
                    
                    for s in STAINS:
                        if s in bag and bag[s].shape[0] > 0:
                            raw = raw_logits_cache[f"{s}_patch_raw"].squeeze()
                            attn = out["patch_attn"][s].cpu().numpy()
                            ent = -(attn * np.log(attn + 1e-8)).sum() / np.log(len(attn)) if len(attn) > 1 else 0
                            val_metrics.append({
                                "epoch": ep, "patient_id": r.patient_id, "stain": s,
                                "entropy": float(ent), "std_raw": float(raw.std()), "mean_raw": float(raw.mean())
                            })

    # Save Stats
    df_ep = pd.DataFrame(epoch_metrics)
    df_ep.to_csv(OUTDIR / "epoch_dynamics.csv", index=False)
    
    df_val = pd.DataFrame(val_metrics)
    df_val.to_csv(OUTDIR / "validation_dynamics.csv", index=False)
    
    # Plot Trajectories
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Plot W_norm & Grad_norm for HE
    ax1 = axes[0]
    ax1.plot(df_ep["epoch"], df_ep["HE_W_norm"], label="HE ||W_att||", color="blue")
    ax1.plot(df_ep["epoch"], df_ep["PAS_W_norm"], label="PAS ||W_att||", color="purple")
    ax1.plot(df_ep["epoch"], df_ep["MT_W_norm"], label="MT ||W_att||", color="green")
    ax1.set_title("Attention Weight Norm over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    
    ax2 = axes[1]
    ax2.plot(df_ep["epoch"], df_ep["HE_grad_norm"], label="HE ||grad_W_att||", color="blue", linestyle="--")
    ax2.plot(df_ep["epoch"], df_ep["PAS_grad_norm"], label="PAS ||grad_W_att||", color="purple", linestyle="--")
    ax2.plot(df_ep["epoch"], df_ep["MT_grad_norm"], label="MT ||grad_W_att||", color="green", linestyle="--")
    ax2.set_title("Attention Gradient Norm over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_yscale("log")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "attention_norms.png", dpi=150)
    plt.close(fig)
    
    # Plot Entropy and STD_raw for validation
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    val_agg = df_val.groupby(["epoch", "stain"]).mean().reset_index()
    
    ax3 = axes[0]
    for s, c in zip(["HE", "PAS", "MT"], ["blue", "purple", "green"]):
        sub = val_agg[val_agg["stain"] == s]
        if len(sub) > 0:
            ax3.plot(sub["epoch"], sub["entropy"], label=f"{s} Entropy", color=c, marker='o')
    ax3.set_title("Average Validation Entropy over Epochs")
    ax3.set_xlabel("Epoch")
    ax3.legend()
    
    ax4 = axes[1]
    for s, c in zip(["HE", "PAS", "MT"], ["blue", "purple", "green"]):
        sub = val_agg[val_agg["stain"] == s]
        if len(sub) > 0:
            ax4.plot(sub["epoch"], sub["std_raw"], label=f"{s} std_raw", color=c, marker='o')
    ax4.set_title("Average Raw Logit STD over Epochs")
    ax4.set_xlabel("Epoch")
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(OUTDIR / "validation_dynamics.png", dpi=150)
    plt.close(fig)
    
    print(f"Dynamics diagnostics completed. Outputs saved to {OUTDIR}")

if __name__ == "__main__":
    main()
