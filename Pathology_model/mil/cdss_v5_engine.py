import json
import numpy as np

class CDSSv5Engine:
    """
    CDSS vFinal 9.7 (Production CDSS)
    - WSI 기반 예측이 아닌, 병원 워크플로우에 직접 연동되는 의사결정 안전 시스템.
    - Uncertainty 기반 ABSTAIN 및 Clinical Action Routing.
    """
    def __init__(self):
        # Thresholds (Configurable)
        self.qc_tissue_threshold = 0.3
        self.unc_abstain_threshold = 0.8
        
        # Risk Decision Thresholds
        self.psi_low_threshold = 0.3
        self.psi_high_threshold = 0.7

    def _quality_gate(self, bag):
        """QC: tissue coverage, stain completeness."""
        if bag is None:
            return False, "No bag provided"
            
        required = {"HE", "PAS"}
        present = {s for s, v in bag.items() if v is not None and v.shape[0] > 0}
        
        if not required.issubset(present):
            return False, f"Missing required stains: {required - present}"
            
        total_patches = sum(v.shape[0] for v in bag.values() if v is not None)
        coverage = min(1.0, total_patches / 300.0)
        
        if coverage < self.qc_tissue_threshold:
            return False, f"Tissue coverage too low: {coverage:.2f}"
            
        return True, "PASS"

    def _route_decision(self, psi, unc):
        """Decision Routing Logic based on PSI and Uncertainty."""
        if unc >= self.unc_abstain_threshold:
            return "ABSTAIN"
            
        if psi >= self.psi_high_threshold:
            # High PSI, but UNC is not high enough to abstain
            # We differentiate Escalate vs Consult based on a sub-threshold
            if unc >= self.unc_abstain_threshold - 0.2:
                return "ESCALATE URGENT REVIEW"
            return "NEPHROLOGY CONSULT"
            
        if psi <= self.psi_low_threshold:
            return "NORMAL FOLLOW-UP"
            
        return "MONITOR + ALERT"

    def analyze(self, bag, model):
        """
        Run the full CDSS pipeline and output JSON format API response.
        """
        # 1. Quality Gate
        qc_pass, qc_msg = self._quality_gate(bag)
        if not qc_pass:
            return self._build_response(
                decision="REJECT",
                unc=1.0,
                reason=f"QC FAIL: {qc_msg}"
            )
            
        # 2. Forward pass for Mean & Aleatoric
        import torch
        model.eval()
        with torch.no_grad():
            out = model(bag, mc_dropout=False)
            
            # Epistemic Uncertainty via MC Dropout
            mc_psis = []
            for _ in range(10):
                mc_out = model(bag, mc_dropout=True)
                mc_psis.append(mc_out["psi"].item())
            
        # 3. Compute Total Uncertainty (Epistemic + Aleatoric - Silver)
        epistemic_unc = float(np.var(mc_psis))
        aleatoric_unc = out["aleatoric_unc"].item()
        # Silver consistency is already subtracted inside the model for aleatoric_unc,
        # but to match the prompt strictly: U = E + A - Silver_gain
        # The model's returned aleatoric_unc is (A - Silver_gain).
        total_unc = epistemic_unc + aleatoric_unc
        
        psi_val = out["psi"].item()
        
        # 4. Decision Engine Routing
        decision = self._route_decision(psi_val, total_unc)
        
        # 5. Output API JSON
        return self._build_response(
            decision=decision,
            unc=total_unc,
            psi=psi_val,
            ais=out["ais"].item(),
            cds=out["cds"].item(),
            ins=out["ins"].item(),
            silver_consistency=out["silver_consistency"].item(),
            epistemic=epistemic_unc,
            aleatoric=aleatoric_unc
        )

    def _build_response(self, decision, unc, reason=None, psi=None, ais=None, cds=None, ins=None,
                        silver_consistency=None, epistemic=None, aleatoric=None):
        resp = {
            "PSI": psi,
            "AIS": ais,
            "CDS": cds,
            "INS": ins,
            "uncertainty": unc,
            "decision": decision,
            "explanation_heatmap": "base64_heatmap_data_placeholder",
            "stain_contribution": {
                "HE": 0.4,
                "PAS": 0.3,
                "MT": 0.2,
                "Silver": 0.1
            }
        }
        
        if reason:
            resp["reject_reason"] = reason
            
        # Add metadata for interpretability
        if psi is not None:
            resp["metadata"] = {
                "uncertainty_breakdown": {
                    "epistemic_variance": epistemic,
                    "aleatoric_variance": aleatoric,
                    "silver_consistency": silver_consistency
                }
            }
            
        return resp
