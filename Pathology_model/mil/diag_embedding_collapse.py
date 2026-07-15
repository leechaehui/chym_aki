"""
diag_embedding_collapse.py
CTransPath 입력 임베딩 자체의 상태(붕괴 여부)를 진단합니다.
Attention 이전 단계에서 피처들이 유효한지(Variance, Norm) 및 
Stain 내/Stain 간 붕괴(Collapse)가 일어났는지 수치 기반으로 증명합니다.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path("c:/team/chym_aki/Pathology_model")))
LOCAL_ROOT = Path("c:/team/chym_aki")

import mil.train
mil.train.ROOT = LOCAL_ROOT

from mil.train import load_bags, SEED, seed_all
from mil.model import STAINS

OUTDIR = LOCAL_ROOT / "Pathology_model/results/14_diag_embedding_collapse"

def compute_intra_cosine(Z, max_samples=2000):
    if Z.shape[0] <= 1:
        return 0.0
    if Z.shape[0] > max_samples:
        idx = torch.randperm(Z.shape[0])[:max_samples]
        Z = Z[idx]
    
    Z_norm = F.normalize(Z, p=2, dim=1)
    sim_matrix = torch.mm(Z_norm, Z_norm.t())
    
    n = Z.shape[0]
    # mask out diagonal
    mask = torch.ones((n, n), device=Z.device) - torch.eye(n, device=Z.device)
    off_diag_sum = (sim_matrix * mask).sum()
    mean_sim = off_diag_sum / (n * (n - 1))
    return float(mean_sim.cpu().numpy())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="ctranspath")
    ap.add_argument("--mags", default="10")
    args = ap.parse_args()
    
    OUTDIR.mkdir(parents=True, exist_ok=True)
    seed_all(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data Loading
    idx = pd.read_csv(LOCAL_ROOT / "data/embeddings" / args.encoder / "index.csv", dtype={"magnification": str})
    msel = [m.strip() for m in args.mags.split(",")]
    idx = idx[idx["magnification"].isin(msel)]
    
    sm = pd.read_csv(LOCAL_ROOT / "split_manifest.csv").drop_duplicates("patient_id")
    coh = sm[sm["fold"] >= 0].copy(); coh["patient_id"] = coh["patient_id"].astype(str)
    
    bags = load_bags(coh["patient_id"].tolist(), idx, keep=None)
    coh = coh[coh["patient_id"].isin(bags.keys())]

    # Use fold 0 as validation target (1회 통과)
    fold = 0
    va = coh[coh["fold"] == fold]
    print(f"Analyzing Embeddings for {len(va)} validation patients...", flush=True)

    stain_stats = []
    inter_stain_stats = []

    for r in va.itertuples():
        pid = r.patient_id
        bag = {s: torch.from_numpy(v).to(device) for s, v in bags[pid].items() if v.shape[0] > 0}
        
        # 1. Intra-stain Statistics
        means = {}
        for s in STAINS:
            if s in bag:
                Z = bag[s]  # (N, D)
                N = Z.shape[0]
                
                l2_norms = torch.norm(Z, p=2, dim=1)
                mean_norm = float(l2_norms.mean().cpu().numpy())
                
                feature_var = float(torch.var(Z, dim=0).mean().cpu().numpy())
                
                mean_cosine = compute_intra_cosine(Z)
                
                stain_stats.append({
                    "patient_id": pid,
                    "stain": s,
                    "num_patches": N,
                    "mean_norm": mean_norm,
                    "variance": feature_var,
                    "mean_intra_cosine": mean_cosine
                })
                
                # Save mean vector for inter-stain comparison
                means[s] = Z.mean(dim=0)
                
        # 2. Inter-stain Statistics (Cross-stain)
        pairs = [("HE", "PAS"), ("HE", "MT"), ("PAS", "MT"), ("HE", "SILVER")]
        for s1, s2 in pairs:
            if s1 in means and s2 in means:
                m1 = means[s1]
                m2 = means[s2]
                
                cos_sim = float(F.cosine_similarity(m1.unsqueeze(0), m2.unsqueeze(0)).cpu().numpy())
                euclidean = float(torch.norm(m1 - m2).cpu().numpy())
                
                inter_stain_stats.append({
                    "patient_id": pid,
                    "pair": f"{s1}_vs_{s2}",
                    "cosine_sim": cos_sim,
                    "euclidean_dist": euclidean
                })

    df_stain = pd.DataFrame(stain_stats)
    df_stain.to_csv(OUTDIR / "intra_stain_stats.csv", index=False)
    
    df_inter = pd.DataFrame(inter_stain_stats)
    df_inter.to_csv(OUTDIR / "inter_stain_stats.csv", index=False)

    print("\n=== Intra-Stain Summary ===")
    summary = df_stain.groupby("stain").agg({
        "mean_norm": "mean",
        "variance": "mean",
        "mean_intra_cosine": "mean"
    }).reset_index()
    print(summary.to_string(index=False))
    summary.to_csv(OUTDIR / "intra_stain_summary.csv", index=False)
    
    print("\n=== Inter-Stain Summary ===")
    inter_summary = df_inter.groupby("pair").agg({
        "cosine_sim": "mean",
        "euclidean_dist": "mean"
    }).reset_index()
    print(inter_summary.to_string(index=False))
    inter_summary.to_csv(OUTDIR / "inter_stain_summary.csv", index=False)

if __name__ == "__main__":
    main()
