import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("c:/team/chym_aki/Pathology_model")))
from mil.cbr_retrieval import HybridCBR

def main():
    try:
        from mil.cdss_paths import data_root as _data_root, p as _p
        ROOT = _data_root()
        EMB_DIR = _p("embeddings") / "ctranspath"
        INDEX_DIR = _p("index")
    except Exception:
        ROOT = Path("c:/team/chym_aki")
        EMB_DIR = ROOT / "data/embeddings/ctranspath"
        INDEX_DIR = ROOT / "data/cbr_index"
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    idx_path = EMB_DIR / "index.csv"
    if not idx_path.exists():
        print(f"Error: {idx_path} not found.")
        sys.exit(1)
        
    df = pd.read_csv(idx_path, dtype={"magnification": str})
    # Filter for 40x HE as primary modality for CBR if multiple exist, or just average whatever is there
    # For CDSS v4, we use HE and PAS. Let's aggregate by patient.
    
    patients = df['patient_id'].unique()
    
    slide_embs_list = []
    patch_embs_dict = {}
    metadatas = []
    
    print(f"Building FAISS index for {len(patients)} patients...")
    for pid in patients:
        sub = df[df['patient_id'] == pid]
        
        patient_patches = []
        for _, row in sub.iterrows():
            npy_path = ROOT / row['npy_path']
            if npy_path.exists():
                arr = np.load(npy_path)
                if arr.shape[0] > 0:
                    patient_patches.append(arr)
                    
        if not patient_patches:
            continue
            
        # Concatenate all patches for this patient across stains/mags
        all_patches = np.concatenate(patient_patches, axis=0) # (N, 768)
        
        # 1. Slide-level embedding (Mean Pooling for Coarse Search)
        slide_emb = np.mean(all_patches, axis=0) # (768,)
        
        # 2. Store for Fine Search (Keep Top-K patches, here we keep all or subsample to max 100 for memory)
        MAX_PATCHES = 100
        if all_patches.shape[0] > MAX_PATCHES:
            # Random subsample or keep top
            indices = np.random.choice(all_patches.shape[0], MAX_PATCHES, replace=False)
            fine_patches = all_patches[indices]
        else:
            fine_patches = all_patches
            
        slide_embs_list.append(slide_emb)
        patch_embs_dict[str(pid)] = fine_patches
        metadatas.append({
            "patient_id": str(pid),
            "stains": list(sub['stain'].unique())
        })
        
    if not slide_embs_list:
        print("No valid embeddings found.")
        sys.exit(1)
        
    slide_embs_matrix = np.vstack(slide_embs_list).astype(np.float32) # (N, 768)
    
    cbr = HybridCBR(embed_dim=768, top_k_patches=10)
    cbr.build_index(slide_embs_matrix, patch_embs_dict, metadatas)
    
    cbr.save(str(INDEX_DIR))
    print(f"FAISS index successfully built and saved to {INDEX_DIR}")

if __name__ == "__main__":
    main()
