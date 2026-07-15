"""
Attention-guided Morphology-Grounded Case-Based Reasoning (CBR) Retrieval Module
(Phase 4 Implementation for CDSS v4)

Handles Hybrid Search:
1. Coarse Search: Slide-level embedding (Top 50)
2. Fine Search: Top-K Patch-level embedding matching (Top 3)
"""
import os
import json
import numpy as np

try:
    import faiss
except ImportError:
    print("Warning: faiss is not installed. Please run `pip install faiss-cpu` or `faiss-gpu`.")

class HybridCBR:
    def __init__(self, embed_dim=768, top_k_patches=10):
        self.embed_dim = embed_dim
        self.top_k_patches = top_k_patches
        self.slide_index = None
        self.metadata = []  # List of dicts mapping faiss ID -> patient/clinical info
        self.patient_patches = {}  # patient_id -> np.ndarray of Top-K patches (K, dim)
        
    def build_index(self, slide_embs, patch_embs_dict, metadatas):
        """
        slide_embs: np.ndarray shape (N, dim)
        patch_embs_dict: dict {patient_id: np.ndarray shape (K, dim)}
        metadatas: list of dicts of length N
        """
        if 'faiss' not in globals():
            raise ImportError("faiss is required to build the index.")
            
        # L2 Normalize slide embeddings for Cosine Similarity in FlatIP
        faiss.normalize_L2(slide_embs)
        
        self.slide_index = faiss.IndexFlatIP(self.embed_dim)
        self.slide_index.add(slide_embs)
        
        self.metadata = metadatas
        
        # Store patch embeddings and L2 normalize them for cosine similarity
        for pid, patches in patch_embs_dict.items():
            if patches.shape[0] > 0:
                faiss.normalize_L2(patches)
                self.patient_patches[pid] = patches
                
        print(f"[CBR] Indexed {self.slide_index.ntotal} slide embeddings.")

    def save(self, index_dir):
        os.makedirs(index_dir, exist_ok=True)
        if self.slide_index:
            faiss.write_index(self.slide_index, os.path.join(index_dir, "slide_coarse.index"))
        with open(os.path.join(index_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        # Save patches dictionary
        np.save(os.path.join(index_dir, "patient_patches.npy"), self.patient_patches, allow_pickle=True)

    def load(self, index_dir):
        if 'faiss' not in globals():
            raise ImportError("faiss is required.")
        self.slide_index = faiss.read_index(os.path.join(index_dir, "slide_coarse.index"))
        with open(os.path.join(index_dir, "metadata.json"), "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.patient_patches = np.load(os.path.join(index_dir, "patient_patches.npy"), allow_pickle=True).item()

    def search_hybrid(self, query_slide_emb, query_patch_embs, top_coarse=50, top_fine=3):
        """
        query_slide_emb: (1, dim)
        query_patch_embs: (K, dim)
        """
        if self.slide_index is None:
            raise ValueError("Index not built or loaded.")
            
        # 1. Coarse Search
        q_s_emb = query_slide_emb.copy()
        faiss.normalize_L2(q_s_emb)
        D_coarse, I_coarse = self.slide_index.search(q_s_emb, min(top_coarse, len(self.metadata)))
        
        candidate_pids = []
        candidates_meta = []
        for rank, idx in enumerate(I_coarse[0]):
            meta = self.metadata[idx]
            candidate_pids.append(meta['patient_id'])
            candidates_meta.append(meta)
            
        # 2. Fine Search (Patch Matching)
        q_p_embs = query_patch_embs.copy()
        faiss.normalize_L2(q_p_embs)
        
        fine_scores = []
        for pid, meta in zip(candidate_pids, candidates_meta):
            if pid not in self.patient_patches:
                fine_scores.append((pid, meta, -1.0))
                continue
                
            ref_patches = self.patient_patches[pid] # (M, dim)
            # Compute pairwise cosine similarity between query K patches and ref M patches
            # sim_matrix: (K, M)
            sim_matrix = np.dot(q_p_embs, ref_patches.T)
            
            # Aggregate patch similarity: For each query patch, find max match, then average
            best_matches = np.max(sim_matrix, axis=1) # shape (K,)
            agg_sim = float(np.mean(best_matches)) # Mean of Top-K patch matches
            
            fine_scores.append((pid, meta, agg_sim))
            
        # Sort by patch-level similarity
        fine_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Return Top-N Fine
        return fine_scores[:top_fine]

# Retrieval 정량 평가 지표 계산기 (향후 완성)
class RetrievalEvaluator:
    @staticmethod
    def calc_topk_precision(retrieval_results, gt_stage):
        # TODO: Implement Precision logic
        pass
        
    @staticmethod
    def calc_descriptor_agreement(retrieval_results, gt_descriptors):
        # TODO: Implement Descriptor distance logic
        pass
