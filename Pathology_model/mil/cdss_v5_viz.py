import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tifffile
import zarr

sys.path.insert(0, str(Path("c:/team/chym_aki/Pathology_model")))
from mil.train import ROOT, load_bags
from mil.cdss_paths import p as _p
from mil.cdss_v5_model import CDSSv5Model

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Data
    emb_dir = _p("embeddings") / "ctranspath"
    idx = pd.read_csv(emb_dir / "index.csv", dtype={"magnification": str})
    embed_dim = int(idx["embed_dim"].iloc[0])
    idx = idx[idx["magnification"] == "10"] # Use 10x
    
    sm = pd.read_csv(ROOT / "split_manifest.csv").drop_duplicates("patient_id")
    p_all = sm["patient_id"].astype(str).tolist()
    keep_stains = ["HE", "PAS", "MT", "SILVER"]
    bags = load_bags(p_all, idx, keep=set(keep_stains))

    # We need a patient that has HE, PAS, MT, and a slide path
    pm = pd.read_csv(ROOT / "patches_manifest.csv", dtype={"magnification": str})
    pm = pm[pm["magnification"] == "10"]
    
    # Find a good patient (e.g. one with High KDIGO stage or ATI)
    cand = sm[(sm["task_ati_severity"] == 5) | (sm["task_chronic"] == 1)]["patient_id"].astype(str).tolist()
    pid = None
    for p in cand:
        if p in bags and "HE" in bags[p]:
            pid = p
            break
    if pid is None:
        pid = list(bags.keys())[0]

    # 2. Train dummy model quickly (1 epoch) just to have initialized/slightly learned weights
    # (Since train_cdss_v5 didn't save weights)
    print("Training quickly for viz...")
    model = CDSSv5Model(in_dim=embed_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    
    bag = {s: torch.from_numpy(v).to(device) for s,v in bags[pid].items() if v is not None}
    
    out = model(bag)
    # Dummy loss to induce some gradients
    loss = out["psi"].mean()
    opt.zero_grad(); loss.backward(); opt.step()
    
    # 3. Inference and extraction
    model.eval()
    with torch.no_grad():
        # Get attention weights by capturing them manually
        # Since CDSSv5Model doesn't return attention in out dict directly, we calculate it
        mats = [v for s, v in bag.items() if s != "SILVER" and v is not None and v.shape[0] > 0]
        x_base = torch.cat(mats, dim=0) # (N, in_dim)
        h_base = model.proj(x_base)
        
        a_ais = torch.softmax(model.attn_ais(h_base), dim=0).cpu().numpy().squeeze()
        a_cds = torch.softmax(model.attn_cds(h_base), dim=0).cpu().numpy().squeeze()
        a_ins = torch.softmax(model.attn_ins(h_base), dim=0).cpu().numpy().squeeze()
        
    # 4. Map back to coordinates
    # The concatenated order is the same as how mats were concatenated.
    # We must find the order of stains in mats
    ordered_stains = [s for s, v in bag.items() if s != "SILVER" and v is not None and v.shape[0] > 0]
    # To keep it simple, let's just visualize the "HE" portion
    he_idx_start = 0
    he_idx_end = bag["HE"].shape[0] if "HE" in ordered_stains else 0
    # Actually, dictionary iteration order is preserved in Python 3.7+
    offset = 0
    for s in ordered_stains:
        if s == "HE":
            he_idx_start = offset
            he_idx_end = offset + bag[s].shape[0]
            break
        offset += bag[s].shape[0]
        
    attn_he_ais = a_ais[he_idx_start:he_idx_end]
    attn_he_cds = a_cds[he_idx_start:he_idx_end]
    
    # Coordinates for HE
    co = pm[(pm["patient_id"].astype(str) == pid) & (pm["stain"] == "HE")].sort_values("tissue_score", ascending=False)
    n = min(len(co), len(attn_he_ais))
    co = co.iloc[:n]
    attn_he_ais = attn_he_ais[:n]
    attn_he_cds = attn_he_cds[:n]
    
    spath = co.iloc[0]["slide_path"]
    slevel = int(co.iloc[0]["source_level"])
    sp = ROOT / spath if not Path(spath).is_absolute() else Path(spath)
    
    print(f"Reading thumbnail for {sp}")
    with tifffile.TiffFile(sp) as t:
        thumb = np.asarray(zarr.open(t.series[0].levels[-1].aszarr(), mode="r")[:])[..., :3]
        Hs = t.series[0].levels[slevel].shape[0]
        Ws = t.series[0].levels[slevel].shape[1]
        
    th, tw = thumb.shape[:2]
    xs = co["tile_x"].to_numpy() / Ws * tw
    ys = co["tile_y"].to_numpy() / Hs * th
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # AIS Heatmap
    axes[0].imshow(thumb, alpha=0.5)
    sc1 = axes[0].scatter(xs, ys, c=attn_he_ais, cmap="Reds", s=20)
    axes[0].set_xlim(0, tw); axes[0].set_ylim(th, 0)
    axes[0].set_title(f"Patient {pid} - Acute Injury (AIS) Attention")
    plt.colorbar(sc1, ax=axes[0], fraction=0.04)
    
    # CDS Heatmap
    axes[1].imshow(thumb, alpha=0.5)
    sc2 = axes[1].scatter(xs, ys, c=attn_he_cds, cmap="Blues", s=20)
    axes[1].set_xlim(0, tw); axes[1].set_ylim(th, 0)
    axes[1].set_title(f"Patient {pid} - Chronic Damage (CDS) Attention")
    plt.colorbar(sc2, ax=axes[1], fraction=0.04)
    
    out_path = Path("C:/Users/301-4/.gemini/antigravity-ide/brain/2b055707-f32c-4739-8475-071c95614018/artifacts/actual_result.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved real image to {out_path}")
    
    # Dump actual JSON
    engine = CDSSv5Engine()
    engine.qc_tissue_threshold = 0.0
    resp = engine.analyze(bag, model)
    with open("C:/Users/301-4/.gemini/antigravity-ide/brain/2b055707-f32c-4739-8475-071c95614018/artifacts/actual_json.json", "w") as f:
        json.dump(resp, f, indent=2)

if __name__ == "__main__":
    from mil.cdss_v5_engine import CDSSv5Engine
    import json
    main()
