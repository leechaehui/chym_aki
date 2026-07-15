import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

sys.path.insert(0, str(Path("c:/team/chym_aki/Pathology_model")))
from mil.cdss_v1_model import CDSSDeploymentModel
from mil.cdss_paths import p as _p
from mil.train import ROOT, load_bags

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CDSSDeploymentModel(in_dim=768, config_path="mil/cdss_config.json").to(device)
weights = torch.load(ROOT / "results/cdss_v1/cdss_v1.1_final.pt", map_location=device, weights_only=True)
model.load_state_dict(weights)
model.eval()

patient_id = "30-10123"
stains_to_plot = ["HE", "PAS", "MT"]
emb_dir = _p("embeddings") / "ctranspath"
idx = pd.read_csv(emb_dir / "index.csv", dtype={"magnification": str})
idx = idx[idx["magnification"] == "10"]

# Load spatial coordinates
manifest = pd.read_csv("C:/team/chym_aki/patches_manifest.csv", dtype=str)
manifest["tile_x"] = manifest["tile_x"].astype(int)
manifest["tile_y"] = manifest["tile_y"].astype(int)

plt.figure(figsize=(24, 18))

for row_idx, stain in enumerate(stains_to_plot):
    bags = load_bags([patient_id], idx, keep={stain})
    if patient_id not in bags or stain not in bags[patient_id]:
        continue
    
    emb = bags[patient_id][stain]
    N = emb.shape[0]
    
    sub_manifest = manifest[(manifest["patient_id"] == patient_id) & 
                            (manifest["stain"] == stain) & 
                            (manifest["magnification"] == "10")]
    
    if len(sub_manifest) != N:
        continue
        
    x_c = sub_manifest["tile_x"].values
    y_c = sub_manifest["tile_y"].values
    slide_id = sub_manifest["slide_id"].iloc[0]
    
    thumb_path = f"D:/chym_aki_data/metadata/thumbnails/{patient_id}_{stain}_{slide_id}_thumb.jpg"
    thumb = None
    if os.path.exists(thumb_path):
        thumb = plt.imread(thumb_path)
        th, tw = thumb.shape[0], thumb.shape[1]
        max_x = x_c.max() + 512
        max_y = max_x * (th / tw)
        if max_y < y_c.max() + 512:
            max_y = y_c.max() + 512
            max_x = max_y * (tw / th)
    else:
        max_x = x_c.max() + 512
        max_y = y_c.max() + 512
        
    probs_ais, probs_cds, probs_ins, risk = [], [], [], []
    with torch.no_grad():
        for i in range(N):
            x = torch.from_numpy(emb[i:i+1]).to(device)
            out = model(x)
            probs_ais.append(out["ais_prob"].item())
            probs_cds.append(out["cds_prob"].item())
            probs_ins.append(out["ins_prob"].item())
            risk.append(out["calibrated_risk_prob"].item())

    titles = ["AIS Risk", "CDS Risk", "INS Risk", "Calibrated Risk"]
    metrics = [probs_ais, probs_cds, probs_ins, risk]
    cmaps = ['Reds', 'Blues', 'Greens', 'hot']
    
    for col_idx in range(4):
        ax = plt.subplot(3, 4, row_idx*4 + col_idx + 1)
        if thumb is not None:
            ax.imshow(thumb, extent=[0, max_x, max_y, 0])
        else:
            ax.invert_yaxis()
            
        sc = ax.scatter(x_c, y_c, c=metrics[col_idx], cmap=cmaps[col_idx], s=120, marker='s', alpha=0.6, vmin=0, vmax=1.0)
        plt.colorbar(sc, fraction=0.046, pad=0.04)
        ax.set_title(f"{stain} - {titles[col_idx]}")
        ax.axis('off')

plt.tight_layout()
plt.savefig("C:/Users/301-4/.gemini/antigravity-ide/brain/2b055707-f32c-4739-8475-071c95614018/artifacts/v1.1_actual_result.png", dpi=150, bbox_inches='tight')
print("Saved heatmap to v1.1_actual_result.png")
