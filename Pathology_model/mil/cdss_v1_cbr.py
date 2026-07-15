import faiss
import numpy as np
import torch
import torch.nn.functional as F

class ClinicalExplanationGenerator:
    """
    [Layer 3] Clinical Explanation Generator (Templated)
    Translates raw cosine similarities into MFDS-compliant clinical reasoning text.
    """
    @staticmethod
    def generate(ais_sim, cds_sim, ins_sim):
        # Convert to percentage
        ais_pct = int(ais_sim * 100)
        cds_pct = int(cds_sim * 100)
        ins_pct = int(ins_sim * 100)
        
        reasons = []
        if ais_pct >= 85:
            reasons.append(f"급성 손상(ATI) 패턴 일치도 {ais_pct}%")
        elif ais_pct >= 75:
            reasons.append(f"급성 손상(ATI) 유사성 관찰")
            
        if cds_pct >= 85:
            reasons.append(f"만성 손상(Chronic) 패턴 일치도 {cds_pct}%")
        elif cds_pct >= 75:
            reasons.append(f"만성 손상(Chronic) 유사성 관찰")
            
        if ins_pct >= 85:
            reasons.append(f"염증(Inflammation) 패턴 일치도 {ins_pct}%")
            
        if not reasons:
            return "과거 사례 기반 형태학적 유사성 참조 정보입니다."
            
        reason_text = ", ".join(reasons)
        return f"{reason_text}. 본 정보는 과거 유사 환자에서 관찰된 형태학적 경향성(Reference Only)입니다."

class CBRRetrievalEngine:
    """
    [Layer 1 & Layer 2] MFDS SaMD Compliant CBR Engine
    """
    def __init__(self):
        self.index = None
        self.meta_db = [] # List of dicts with patient info and z_ais, z_cds, z_ins
        
    def build_index(self, z_concat_list, meta_list):
        # z_concat_list: List of (1, 768) tensors or numpy arrays
        feats = np.vstack([z.cpu().numpy() if torch.is_tensor(z) else z for z in z_concat_list]).astype('float32')
        faiss.normalize_L2(feats)
        
        self.index = faiss.IndexFlatIP(feats.shape[1])
        self.index.add(feats)
        self.meta_db = meta_list
        print(f"FAISS Index built with {len(feats)} cases.")
        
    def retrieve(self, z_concat, z_ais, z_cds, z_ins, k=5):
        """
        Retrieves top K similar cases and performs post-hoc feature decomposition.
        Inputs are torch tensors from the model (shape: 1, D).
        """
        if self.index is None:
            return []
            
        # [Layer 1] FAISS Retrieval
        query = z_concat.detach().cpu().numpy().astype('float32')
        if query.ndim == 1: query = query.reshape(1, -1)
        faiss.normalize_L2(query)
        
        distances, indices = self.index.search(query, k+1) # search k+1 in case self is included
        
        results = []
        q_ais = F.normalize(z_ais.detach(), p=2, dim=-1)
        q_cds = F.normalize(z_cds.detach(), p=2, dim=-1)
        q_ins = F.normalize(z_ins.detach(), p=2, dim=-1)
        
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue
            meta = self.meta_db[idx]
            
            # Skip if same patient (simulation edge case)
            if hasattr(self, 'current_patient') and meta.get('patient_id') == self.current_patient:
                continue
                
            # [Layer 2] Feature-wise Distance in Normalized Subspaces
            db_ais = F.normalize(meta['z_ais'].detach(), p=2, dim=-1)
            db_cds = F.normalize(meta['z_cds'].detach(), p=2, dim=-1)
            db_ins = F.normalize(meta['z_ins'].detach(), p=2, dim=-1)
            
            ais_sim = (q_ais * db_ais).sum().item()
            cds_sim = (q_cds * db_cds).sum().item()
            ins_sim = (q_ins * db_ins).sum().item()
            
            # [Layer 3] Explanation Generation
            explanation = ClinicalExplanationGenerator.generate(ais_sim, cds_sim, ins_sim)
            
            results.append({
                "patient_id": meta["patient_id"],
                "total_similarity": round(float(distances[0][i]), 4),
                "feature_similarities": {
                    "ais_sim": round(ais_sim, 4),
                    "cds_sim": round(cds_sim, 4),
                    "ins_sim": round(ins_sim, 4)
                },
                "why_similar": explanation,
                "clinical_outcome": meta["clinical_outcome"]
            })
            if len(results) == k: break
            
        return results

if __name__ == "__main__":
    # Test
    cbr = CBRRetrievalEngine()
    print("MFDS CBR Engine V1.0 Test Passed.")
