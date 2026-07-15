import torch
import torch.nn as nn
import torch.nn.functional as F

class CDSSv5Model(nn.Module):
    """
    CDSS vFinal 9.7 (Production CDSS)
    - Late Fusion with Dynamic Gating (Independent Modality Encoders)
    - Multi-Head Pathology Space: AIS, CDS, INS
    - Aleatoric Uncertainty: Log-likelihood prediction (log_var)
    - Silver Refinement Network
    - PSI (Risk Engine)
    """
    def __init__(self, in_dim=2048, dim=256, dropout=0.25):
        super().__init__()
        self.in_dim = in_dim
        self.dim = dim
        
        # 1. Feature Extractor (Independent per stain to prevent leakage)
        self.proj_he = nn.Sequential(nn.Linear(in_dim, dim), nn.ReLU(), nn.Dropout(dropout))
        self.proj_pas = nn.Sequential(nn.Linear(in_dim, dim), nn.ReLU(), nn.Dropout(dropout))
        self.proj_mt = nn.Sequential(nn.Linear(in_dim, dim), nn.ReLU(), nn.Dropout(dropout))
        
        # 2. Attention Pooling Heads (Independent per stain and task)
        self.attn_ais_he = nn.Linear(dim, 1)
        self.attn_ais_pas = nn.Linear(dim, 1)
        self.attn_ais_mt = nn.Linear(dim, 1)
        
        self.attn_cds_he = nn.Linear(dim, 1)
        self.attn_cds_pas = nn.Linear(dim, 1)
        self.attn_cds_mt = nn.Linear(dim, 1)
        
        self.attn_ins_he = nn.Linear(dim, 1)
        self.attn_ins_pas = nn.Linear(dim, 1)
        self.attn_ins_mt = nn.Linear(dim, 1)
        
        # 3. Dynamic Modality Gates
        self.gate_ais = nn.Sequential(nn.Linear(dim * 3, 128), nn.ReLU(), nn.Linear(128, 3))
        self.gate_cds = nn.Sequential(nn.Linear(dim * 3, 128), nn.ReLU(), nn.Linear(128, 3))
        self.gate_ins = nn.Sequential(nn.Linear(dim * 3, 128), nn.ReLU(), nn.Linear(128, 3))
        
        # 4. Pathology Target Heads
        self.head_ais = nn.Linear(dim, 2)
        self.head_cds = nn.Linear(dim, 2)
        self.head_ins = nn.Linear(dim, 2)
        
        # 5. Silver Refinement Network
        self.attn_sil = nn.Linear(dim, 1)
        self.sil_structure = nn.Linear(dim, 1)
        self.sil_fibrosis = nn.Linear(dim, 1)
        
        # Learnable Parameters for Silver Refinement
        self.gamma = nn.Parameter(torch.tensor(0.1)) # CDS refinement weight
        self.delta = nn.Parameter(torch.tensor(0.1)) # AIS refinement weight
        self.eta = nn.Parameter(torch.tensor(0.1))   # Uncertainty reduction weight
        
        # 6. Risk Multiplier Parameters (PSI)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        
        # Variance Clamp Constraints
        self.log_var_min = -5.0 # ~0.006
        self.log_var_max = 2.0  # ~7.38

    def forward(self, bag, mc_dropout=False):
        """
        bag: dict of {"HE": tensor(N, in_dim), "PAS": ..., "SILVER": ...}
        mc_dropout: Boolean to keep dropout active during inference for Epistemic uncertainty
        """
        if mc_dropout:
            for m in self.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()
                    
        device = next(self.parameters()).device
        
        def process_stain(stain_name, proj, attn_ais, attn_cds, attn_ins):
            if stain_name in bag and bag[stain_name] is not None and bag[stain_name].shape[0] > 0:
                x = bag[stain_name]
                h = proj(x)
                a_ais = torch.softmax(attn_ais(h), dim=0)
                z_ais = (a_ais * h).sum(0)
                a_cds = torch.softmax(attn_cds(h), dim=0)
                z_cds = (a_cds * h).sum(0)
                a_ins = torch.softmax(attn_ins(h), dim=0)
                z_ins = (a_ins * h).sum(0)
                return z_ais, z_cds, z_ins, 1.0, a_ais, a_cds, a_ins, h
            else:
                z0 = torch.zeros(self.dim, device=device)
                return z0, z0, z0, 0.0, None, None, None, None

        z_ais_he, z_cds_he, z_ins_he, m_he, a_ais_he, a_cds_he, a_ins_he, h_he = process_stain("HE", self.proj_he, self.attn_ais_he, self.attn_cds_he, self.attn_ins_he)
        z_ais_pas, z_cds_pas, z_ins_pas, m_pas, a_ais_pas, a_cds_pas, a_ins_pas, h_pas = process_stain("PAS", self.proj_pas, self.attn_ais_pas, self.attn_cds_pas, self.attn_ins_pas)
        z_ais_mt, z_cds_mt, z_ins_mt, m_mt, a_ais_mt, a_cds_mt, a_ins_mt, h_mt = process_stain("MT", self.proj_mt, self.attn_ais_mt, self.attn_cds_mt, self.attn_ins_mt)
        
        mask = torch.tensor([m_he, m_pas, m_mt], device=device)
        if mask.sum() == 0:
            raise ValueError("No base stains found.")
            
        def apply_gate(gate_net, z_he, z_pas, z_mt, mask):
            gate_in = torch.cat([z_he, z_pas, z_mt], dim=-1)
            logits = gate_net(gate_in)
            logits = logits + (1.0 - mask) * -1e9 # Mask missing stains
            weights = torch.softmax(logits, dim=-1)
            z_fused = weights[0]*z_he + weights[1]*z_pas + weights[2]*z_mt
            return z_fused, weights

        z_ais_fused, w_ais = apply_gate(self.gate_ais, z_ais_he, z_ais_pas, z_ais_mt, mask)
        z_cds_fused, w_cds = apply_gate(self.gate_cds, z_cds_he, z_cds_pas, z_cds_mt, mask)
        z_ins_fused, w_ins = apply_gate(self.gate_ins, z_ins_he, z_ins_pas, z_ins_mt, mask)

        out_ais = self.head_ais(z_ais_fused)
        ais_base, log_var_ais = out_ais[0], out_ais[1]
        
        out_cds = self.head_cds(z_cds_fused)
        cds_base, log_var_cds = out_cds[0], out_cds[1]
        
        out_ins = self.head_ins(z_ins_fused)
        ins_base, log_var_ins = out_ins[0], out_ins[1]
        
        log_var_ais = torch.clamp(log_var_ais, self.log_var_min, self.log_var_max)
        log_var_cds = torch.clamp(log_var_cds, self.log_var_min, self.log_var_max)
        log_var_ins = torch.clamp(log_var_ins, self.log_var_min, self.log_var_max)
        
        aleatoric_unc = (torch.exp(log_var_ais) + torch.exp(log_var_cds) + torch.exp(log_var_ins)) / 3.0
        
        silver_consistency = torch.tensor(0.0, device=device)
        cds, ais, ins = cds_base, ais_base, ins_base
        if "SILVER" in bag and bag["SILVER"] is not None and bag["SILVER"].shape[0] > 0:
            x_sil = bag["SILVER"]
            h_sil = self.proj_he(x_sil)
            a_sil = torch.softmax(self.attn_sil(h_sil), dim=0)
            z_sil = (a_sil * h_sil).sum(0)
            
            sil_str = self.sil_structure(z_sil).squeeze(-1)
            sil_fib = self.sil_fibrosis(z_sil).squeeze(-1)
            
            cds = cds_base + self.gamma * sil_fib
            ais = ais_base + self.delta * sil_str
            
            consistency_penalty = (sil_fib - cds_base)**2 + (sil_str - ais_base)**2
            silver_consistency = consistency_penalty.mean()
            
            aleatoric_unc = aleatoric_unc * torch.exp(-self.eta * sil_fib.detach())

        pred_ais = torch.sigmoid(ais)
        pred_cds = torch.sigmoid(cds)
        pred_ins = torch.sigmoid(ins)
        
        psi = self.alpha * pred_cds * pred_ins + self.beta * pred_ais

        # Calculate KL penalty for gate weights vs uniform distribution
        num_valid = mask.sum()
        if num_valid > 0:
            uniform_prob = 1.0 / num_valid
            kl_ais = (w_ais * torch.log((w_ais + 1e-8) / uniform_prob)) * mask
            kl_cds = (w_cds * torch.log((w_cds + 1e-8) / uniform_prob)) * mask
            kl_ins = (w_ins * torch.log((w_ins + 1e-8) / uniform_prob)) * mask
            kl_penalty = (kl_ais.sum() + kl_cds.sum() + kl_ins.sum()) / 3.0
        else:
            kl_penalty = torch.tensor(0.0, device=device)

        return {
            "pred_ais": pred_ais, "pred_cds": pred_cds, "pred_ins": pred_ins,
            "unc_ais": torch.exp(log_var_ais), "unc_cds": torch.exp(log_var_cds), "unc_ins": torch.exp(log_var_ins),
            "aleatoric_unc": aleatoric_unc,
            "silver_consistency": silver_consistency,
            "w_ais": w_ais, "w_cds": w_cds, "w_ins": w_ins,
            "kl_penalty": kl_penalty,
            "ais": ais, "cds": cds, "ins": ins,
            "log_var_ais": log_var_ais, "log_var_cds": log_var_cds, "log_var_ins": log_var_ins,
            "psi": psi
        }
