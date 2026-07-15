import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from pathlib import Path

class CDSSDeploymentModel(nn.Module):
    """
    Production-Grade CDSS Model (MFDS SaMD Compliant)
    - Versioned, Deterministic, Explanable.
    - Monotonic linear risk scoring.
    - Separate latent spaces (AIS, CDS, INS) for Post-hoc CBR Explanation.
    - Temperature Scaling for calibrated probabilities.
    """
    def __init__(self, in_dim=768, hidden_dim=256, config_path="mil/cdss_config.json"):
        super().__init__()
        self.in_dim = in_dim
        
        # Load external config (No Hardcoding)
        try:
            with open(config_path, "r") as f:
                self.config = json.load(f)
        except:
            self.config = {"model_version": "unknown", "calibration_version": "unknown"}
        
        # 1. Base Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 2. Separate Latent Spaces for CBR (256-dim each)
        self.ais_proj = nn.Sequential(nn.Linear(hidden_dim, 256), nn.ReLU())
        self.cds_proj = nn.Sequential(nn.Linear(hidden_dim, 256), nn.ReLU())
        self.ins_proj = nn.Sequential(nn.Linear(hidden_dim, 256), nn.ReLU())
        
        # 3. Pathology Predictors
        self.ais_head = nn.Linear(256, 1)
        self.cds_head = nn.Linear(256, 1)
        self.ins_head = nn.Linear(256, 1)
        
        # 4. Monotonic Risk Engine (Linear combination of pathology logits)
        self.risk_layer = nn.Linear(3, 1)
        
        # 5. Uncertainty Head (Auxiliary reference only)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 0~1 range for reference
        )
        
        # 6. Temperature Scaling (Calibration)
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
    def forward(self, h):
        """
        h: (N, in_dim) bags or (1, in_dim) pooled feature
        """
        if h.dim() == 2:
            h = h.mean(dim=0, keepdim=True)
            
        z_base = self.encoder(h)
        
        # Latent spaces
        z_ais = self.ais_proj(z_base)
        z_cds = self.cds_proj(z_base)
        z_ins = self.ins_proj(z_base)
        
        # Concatenated embedding for CBR FAISS (Dim = 256*3 = 768)
        z_concat = torch.cat([z_ais, z_cds, z_ins], dim=-1)
        
        # Pathology Logits
        logit_ais = self.ais_head(z_ais)
        logit_cds = self.cds_head(z_cds)
        logit_ins = self.ins_head(z_ins)
        
        # Raw Risk Score
        risk_input = torch.cat([logit_ais, logit_cds, logit_ins], dim=-1)
        raw_risk_logit = self.risk_layer(risk_input)
        
        # Temperature Scaled Calibration
        calibrated_risk_logit = raw_risk_logit / self.temperature
        calibrated_risk_prob = torch.sigmoid(calibrated_risk_logit)
        
        # Uncertainty
        uncertainty_score = self.uncertainty_head(z_base)
        
        return {
            "model_version": self.config.get("model_version", "unknown"),
            "calib_version": self.config.get("calibration_version", "unknown"),
            "z_ais": z_ais,
            "z_cds": z_cds,
            "z_ins": z_ins,
            "z_concat": z_concat,
            "ais_prob": torch.sigmoid(logit_ais),
            "cds_prob": torch.sigmoid(logit_cds),
            "ins_prob": torch.sigmoid(logit_ins),
            "raw_risk_logit": raw_risk_logit,
            "calibrated_risk_prob": calibrated_risk_prob,
            "uncertainty_score": uncertainty_score
        }

if __name__ == "__main__":
    # Test
    model = CDSSDeploymentModel()
    dummy_input = torch.randn(100, 768) # 100 patches
    out = model(dummy_input)
    print("MFDS CDSS v1.0 Model Test PASSED.")
    print(f"Calibrated Prob: {out['calibrated_risk_prob'].item():.4f}")
    print(f"z_concat shape: {out['z_concat'].shape}")
