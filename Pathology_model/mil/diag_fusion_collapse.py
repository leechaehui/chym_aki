"""
멀티스테인 Fusion Weight 및 Raw Logit 붕괴 진단 스크립트 (diag_fusion_collapse.py)
원인 규명을 위해 다음을 수행합니다:
1. Stain별 Fusion Weight (stain_contrib) 타깃별 통계 추출
2. Softmax 전 Raw Logit 분포 추출 및 엔트로피 계산
3. 샘플 환자 대상 Top-20 패치 원본 이미지 추출 및 시각화 (Tissue Mask 검증)
"""

import argparse
import json
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
import zarr
import tifffile

sys.path.insert(0, str(Path("c:/team/chym_aki/Pathology_model")))
from mil.train import CLF_TASKS, COL, SEED, load_bags, seed_all
from mil.model import StainAwareMIL, STAINS
from mil.embed_patches import read_region

LOCAL_ROOT = Path("c:/team/chym_aki")
OUTDIR = LOCAL_ROOT / "Pathology_model/results/12_diag_fusion_collapse"

# Hook storage
raw_logits_cache = {}

def get_hook(name):
    def hook(module, input, output):
        raw_logits_cache[name] = output.detach().cpu().numpy()
    return hook

def get_patch(slide_path, source_level, tile_x, tile_y, read_size, out_size=256):
    if not Path(slide_path).is_absolute():
        slide_path = LOCAL_ROOT / slide_path
    with tifffile.TiffFile(slide_path) as t:
        za = zarr.open(t.series[0].aszarr(level=int(source_level)), mode="r")
        p = read_region(za, int(tile_x), int(tile_y), int(read_size), int(out_size))
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="ctranspath")
    ap.add_argument("--mags", default="10")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--samples", type=int, default=5, help="Top-20 추출 대상 환자 수")
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "top20_patches").mkdir(parents=True, exist_ok=True)
    
    seed_all(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data Loading
    idx = pd.read_csv(LOCAL_ROOT / "data/embeddings" / args.encoder / "index.csv", dtype={"magnification": str})
    msel = [m.strip() for m in args.mags.split(",")]
    idx = idx[idx["magnification"].isin(msel)]
    embed_dim = int(idx["embed_dim"].iloc[0])
    
    # Patch ROOT for imported modules
    import mil.train
    mil.train.ROOT = LOCAL_ROOT

    sm = pd.read_csv(LOCAL_ROOT / "split_manifest.csv").drop_duplicates("patient_id")
    coh = sm[sm["fold"] >= 0].copy(); coh["patient_id"] = coh["patient_id"].astype(str)
    coh["ati_severity_n"] = (coh["task_ati_severity"] - 1) / 2.0
    bags = load_bags(coh["patient_id"].tolist(), idx, keep=None)
    coh = coh[coh["patient_id"].isin(bags.keys())]
    
    manifest = pd.read_csv(LOCAL_ROOT / "patches_manifest.csv", dtype=str)
    manifest["tile_x"] = manifest["tile_x"].astype(int)
    manifest["tile_y"] = manifest["tile_y"].astype(int)

    # Train on fold 1/2/3/4, validate on fold 0
    fold = 0
    tr = coh[coh["fold"] != fold]; va = coh[coh["fold"] == fold]
    
    print(f"Training on {len(tr)} patients, Validating on {len(va)} patients for {args.epochs} epochs...", flush=True)
    
    model = StainAwareMIL(in_dim=embed_dim, silver_mode="off").to(device)
    # Register Hooks
    for s in STAINS:
        if hasattr(model.encoders[s], 'attn'):
            model.encoders[s].attn.register_forward_hook(get_hook(f"{s}_patch_raw"))
        else:
            model.encoders[s].w.register_forward_hook(get_hook(f"{s}_patch_raw"))
    model.fw.register_forward_hook(get_hook("fusion_raw"))
    
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    def pw(col):
        yy = tr[col].dropna(); n1 = (yy == 1).sum(); n0 = (yy == 0).sum()
        return torch.tensor(n0 / max(n1, 1), device=device, dtype=torch.float32)
    PW = {t: pw(COL[t]) for t in CLF_TASKS}

    for ep in range(args.epochs):
        model.train()
        loss_sum = 0
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
                if not (yv != yv):
                    loss = loss + F.binary_cross_entropy_with_logits(
                        out[t], torch.tensor(float(yv), device=device), pos_weight=PW[t]); nt += 1
            if nt:
                opt.zero_grad(); loss.backward(); opt.step()
                loss_sum += float(loss)
        
        mean_fw = {s: float(np.mean(epoch_stain_contrib[s])) for s in epoch_stain_contrib}
        w_vals = np.array(list(mean_fw.values()))
        w_vals = w_vals / (w_vals.sum() + 1e-8)
        entropy = float(-np.sum(w_vals * np.log(w_vals + 1e-8)))
        
        print(f"Epoch {ep} | Loss: {loss_sum:.4f} | Fusion: ", end="")
        for s, w in mean_fw.items():
            print(f"{s} {w:.2f} ", end="")
        print(f"| Entropy: {entropy:.2f}")
        
        # Print Patch-level Attention stats
        for s in STAINS:
            if hasattr(model.encoders[s], 'last_stats') and model.encoders[s].last_stats:
                st = model.encoders[s].last_stats
                print(f"  [{s}] H(m:{st['h_mean']:.4f}, s:{st['h_std']:.4f}) | "
                      f"V(m:{st['v_mean']:.4f}, s:{st['v_std']:.4f}) | "
                      f"U(m:{st['u_mean']:.4f}, s:{st['u_std']:.4f}) | "
                      f"logit(m:{st['logit_mean']:.4f}, s:{st['logit_std']:.4f})")
        print("", flush=True)

    print("Extracting Validation Statistics...", flush=True)
    model.eval()
    
    fusion_stats = []
    raw_logits_stats = []
    
    sample_patients = va["patient_id"].tolist()[:args.samples]
    
    with torch.no_grad():
        for r in va.itertuples():
            bag = {s: torch.from_numpy(v).to(device) for s, v in bags[r.patient_id].items()}
            out = model(bag)
            
            # Fusion stats
            contrib = out["stain_contrib"]
            f_stat = {"patient_id": r.patient_id, "chronic": getattr(r, COL["chronic"]), "immune": getattr(r, COL["immune"])}
            f_stat.update({s: contrib.get(s, 0.0) for s in STAINS})
            # save the raw fusion logits
            if "fusion_raw" in raw_logits_cache:
                f_stat["fusion_raw"] = raw_logits_cache["fusion_raw"].tolist()
            fusion_stats.append(f_stat)
            
            # Raw logits & Entropy
            for s in STAINS:
                if s in bag and bag[s].shape[0] > 0:
                    raw = raw_logits_cache[f"{s}_patch_raw"].squeeze()
                    attn = out["patch_attn"][s].cpu().numpy()
                    ent = -(attn * np.log(attn + 1e-8)).sum() / np.log(len(attn)) if len(attn) > 1 else 0
                    
                    raw_logits_stats.append({
                        "patient_id": r.patient_id, "stain": s,
                        "mean_raw": float(raw.mean()), "std_raw": float(raw.std()), "max_raw": float(raw.max()),
                        "entropy": float(ent), "top1_gap": float(raw.max() - np.partition(raw, -2)[-2] if len(raw) > 1 else 0)
                    })
                    
                    # Top-20 Patch Extraction for Sample Patients
                    if r.patient_id in sample_patients:
                        print(f"Extracting Top-20 for {r.patient_id} ({s})...")
                        top20_idx = np.argsort(attn)[::-1][:20]
                        sub_manifest = manifest[(manifest["patient_id"] == r.patient_id) & (manifest["stain"] == s) & (manifest["magnification"] == "10")].copy()
                        if len(sub_manifest) == len(attn):
                            top_df = sub_manifest.iloc[top20_idx].copy()
                            top_df["attn"] = attn[top20_idx]
                            top_df["raw_logit"] = raw[top20_idx]
                            top_df.to_csv(OUTDIR / "top20_patches" / f"{r.patient_id}_{s}_coords.csv", index=False)
                            
                            # Draw 4x5 grid
                            fig, axes = plt.subplots(4, 5, figsize=(15, 12))
                            fig.suptitle(f"Top 20 Patches - {r.patient_id} ({s})", fontsize=16)
                            for i, (_, row) in enumerate(top_df.iterrows()):
                                ax = axes[i // 5, i % 5]
                                try:
                                    img = get_patch(row["slide_path"], row["source_level"], row["tile_x"], row["tile_y"], row["read_size"], out_size=256)
                                    ax.imshow(img)
                                    ax.set_title(f"A:{row['attn']:.3f} | L:{row['raw_logit']:.2f}")
                                except Exception as e:
                                    ax.set_title("Load Fail")
                                ax.axis('off')
                            plt.tight_layout()
                            plt.savefig(OUTDIR / "top20_patches" / f"{r.patient_id}_{s}_top20.png", dpi=150)
                            plt.close(fig)
                        else:
                            print(f"Len mismatch for {s}: manifest={len(sub_manifest)}, attn={len(attn)}")

    # Save Stats
    pd.DataFrame(fusion_stats).to_csv(OUTDIR / "fusion_stats.csv", index=False)
    pd.DataFrame(raw_logits_stats).to_csv(OUTDIR / "raw_logits_stats.csv", index=False)
    
    # Plot Fusion Weight Distributions
    fdf = pd.DataFrame(fusion_stats)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, t in zip(axes, ["HE", "PAS", "MT"]):
        vals = fdf[t].dropna()
        ax.hist(vals, bins=20, alpha=0.7)
        ax.set_title(f"{t} Fusion Weight\nMean: {vals.mean():.3f}")
    plt.tight_layout()
    plt.savefig(OUTDIR / "stain_weight_distribution.png", dpi=150)
    plt.close(fig)

    # Plot Raw Logit Stats
    rdf = pd.DataFrame(raw_logits_stats)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, t in zip(axes, ["HE", "PAS", "MT"]):
        vals = rdf[rdf["stain"] == t]["mean_raw"].dropna()
        ax.hist(vals, bins=20, alpha=0.7)
        ax.set_title(f"{t} Mean Raw Logit")
    plt.tight_layout()
    plt.savefig(OUTDIR / "raw_logit_distribution.png", dpi=150)
    plt.close(fig)
    
    print(f"Diagnostics completed. Outputs saved to {OUTDIR}")

if __name__ == "__main__":
    main()
