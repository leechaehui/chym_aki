import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
import zarr
import cv2
from pathlib import Path
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path("c:/team/chym_aki/Pathology_model")))
from mil.cdss_paths import p as _p
from mil.train import ROOT, load_bags
from mil.cdss_v5_model import CDSSv5Model
from mil.embed_patches import read_region

def get_patch(slide_path, source_level, tile_x, tile_y, read_size, out_size=256):
    if not Path(slide_path).is_absolute():
        slide_path = ROOT / slide_path
    with tifffile.TiffFile(slide_path) as t:
        za = zarr.open(t.series[0].aszarr(level=int(source_level)), mode="r")
        p = read_region(za, int(tile_x), int(tile_y), int(read_size), int(out_size))
    return p

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = CDSSv5Model(in_dim=768).to(device)
    weights = torch.load(ROOT / "results/cdss_v5/cdss_v5_fold0_vfinal_97.pt", map_location=device, weights_only=True)
    model.load_state_dict(weights)
    model.eval()
    
    patient_id = "30-10123"
    stains = ["HE", "PAS", "MT"]
    
    emb_dir = _p("embeddings") / "ctranspath"
    idx = pd.read_csv(emb_dir / "index.csv", dtype={"magnification": str})
    idx = idx[idx["magnification"] == "10"]
    
    manifest = pd.read_csv("C:/team/chym_aki/patches_manifest.csv", dtype=str)
    manifest["tile_x"] = manifest["tile_x"].astype(int)
    manifest["tile_y"] = manifest["tile_y"].astype(int)
    
    bags = load_bags([patient_id], idx, keep=set(stains))
    if patient_id not in bags:
        print("Patient not found.")
        return

    mats = []
    lengths = []
    valid_stains = []
    for s in stains:
        if s in bags[patient_id] and bags[patient_id][s] is not None and bags[patient_id][s].shape[0] > 0:
            mats.append(torch.from_numpy(bags[patient_id][s]).to(device))
            lengths.append(bags[patient_id][s].shape[0])
            valid_stains.append(s)
            
    x_base = torch.cat(mats, dim=0)
    with torch.no_grad():
        h_base = model.proj(x_base)
        a_ais = torch.softmax(model.attn_ais(h_base), dim=0)
        a_cds = torch.softmax(model.attn_cds(h_base), dim=0)
        a_ins = torch.softmax(model.attn_ins(h_base), dim=0)
        a_avg = (a_ais + a_cds + a_ins) / 3.0
        
    a_stains = torch.split(a_avg, lengths, dim=0)
    
    # Normalize attention globally across all stains for fair color mapping
    all_attn = a_avg.cpu().numpy()
    vmin, vmax = np.percentile(all_attn, 5), np.percentile(all_attn, 99)
    
    fig = plt.figure(figsize=(24, 7 * len(valid_stains)))
    gs = gridspec.GridSpec(len(valid_stains), 3, width_ratios=[1, 1, 1.2])
    
    for row_idx, (stain, attn) in enumerate(zip(valid_stains, a_stains)):
        attn_np = attn.cpu().numpy().squeeze()
        
        sub_manifest = manifest[(manifest["patient_id"] == patient_id) & 
                                (manifest["stain"] == stain) & 
                                (manifest["magnification"] == "10")].copy()
        
        sub_manifest["attn"] = attn_np
        x_c = sub_manifest["tile_x"].values
        y_c = sub_manifest["tile_y"].values
        slide_id = sub_manifest["slide_id"].iloc[0]
        
        thumb_path = f"D:/chym_aki_data/metadata/thumbnails/{patient_id}_{stain}_{slide_id}_thumb.jpg"
        thumb = None
        if os.path.exists(thumb_path):
            thumb = plt.imread(thumb_path)
            
        ax0 = plt.subplot(gs[row_idx, 0])
        ax1 = plt.subplot(gs[row_idx, 1])
        ax2 = plt.subplot(gs[row_idx, 2])
        
        # 1. Original
        if thumb is not None:
            ax0.imshow(thumb)
        ax0.set_title(f"{stain} - Original Slide", fontsize=18)
        ax0.axis('off')
        
        # 2. Smooth Heatmap Overlay
        if thumb is not None:
            # Create a 2D grid for the heatmap
            patch_size = 2560 # Assuming 10x is 2560 at level 0 (from 40x 256)
            grid_w = int(x_c.max() / patch_size) + 1
            grid_h = int(y_c.max() / patch_size) + 1
            
            heatmap = np.zeros((grid_h, grid_w))
            for x, y, a in zip(x_c, y_c, attn_np):
                heatmap[int(y / patch_size), int(x / patch_size)] = a
                
            # Resize heatmap to match thumbnail (Use NEAREST to preserve the discrete blocky patch grid look like the paper)
            th, tw = thumb.shape[:2]
            heatmap_resized = cv2.resize(heatmap, (tw, th), interpolation=cv2.INTER_NEAREST)
            
            # Normalize and apply colormap
            heatmap_norm = np.clip((heatmap_resized - vmin) / (vmax - vmin + 1e-8), 0, 1)
            heatmap_color = plt.cm.jet(heatmap_norm)[:, :, :3]
            
            # Mask out background (where heatmap is 0)
            mask = heatmap_resized > 1e-6
            mask = np.expand_dims(mask, axis=2)
            
            # Blend
            alpha = 0.5
            blended = (thumb / 255.0) * (1 - mask * alpha) + heatmap_color * (mask * alpha)
            blended = np.clip(blended, 0, 1)
            
            ax1.imshow(blended)
        ax1.set_title(f"{stain} - Attention Heatmap", fontsize=18)
        ax1.axis('off')
        
        # 3. Top-16 Patches (Pure raw tissue)
        ax2.axis('off')
        ax2.set_title(f"{stain} - Top 16 Patches", fontsize=18)
        
        top16 = sub_manifest.sort_values("attn", ascending=False).head(16)
        gs_inner = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=gs[row_idx, 2], wspace=0.1, hspace=0.2)
        
        for i, (_, row) in enumerate(top16.iterrows()):
            if i >= 16: break
            try:
                p_img = get_patch(row["slide_path"], row["source_level"], row["tile_x"], row["tile_y"], row["read_size"], out_size=256)
                ax_patch = plt.subplot(gs_inner[i])
                ax_patch.imshow(p_img)
                ax_patch.axis('off')
                ax_patch.set_title(f"Score: {row['attn']:.4f}", fontsize=12)
                
                # Add rank
                ax_patch.text(0.05, 0.95, f"#{i+1}", transform=ax_patch.transAxes, 
                              color='black', va='top', ha='left', fontweight='bold', 
                              bbox=dict(facecolor='white', alpha=0.8, pad=1, edgecolor='none'))
            except Exception as e:
                print(f"Failed to load patch {i}: {e}")

    plt.tight_layout()
    out_path = "C:/Users/301-4/.gemini/antigravity-ide/brain/2b055707-f32c-4739-8475-071c95614018/artifacts/cdss_v5_topk_viz.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved Top-K visualization to {out_path}")

if __name__ == "__main__":
    main()
