import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
from pathlib import Path

sys.path.insert(0, str(Path("c:/team/chym_aki/Pathology_model")))
from mil.train import ROOT, SEED, load_bags, seed_all
from mil.cdss_paths import p as _p
from mil.cdss_v1_model import CDSSDeploymentModel
from train_cdss_v1 import map_ati, map_kdigo

def compute_ece(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0., 1., n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        if i == 0:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

def main():
    seed_all(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    emb_dir = _p("embeddings") / "ctranspath"
    idx = pd.read_csv(emb_dir / "index.csv", dtype={"magnification": str})
    embed_dim = int(idx["embed_dim"].iloc[0])
    idx = idx[idx["magnification"] == "10"]
    
    sm = pd.read_csv(ROOT / "split_manifest.csv").drop_duplicates("patient_id")
    p_all = sm["patient_id"].astype(str).tolist()
    
    bags = load_bags(p_all, idx, keep=set(["HE", "PAS", "MT"]))
    
    sm["target_kdigo"] = sm["kdigo_stage"].apply(map_kdigo)
    # Binarize KDIGO for AUROC/Calibration (Stage 2/3 = Severe KDIGO = 1)
    sm["kdigo_binary"] = (sm["target_kdigo"] >= 0.66).astype(int)
    
    valid_p = [p for p in p_all if p in bags and any(bags[p].get(s) is not None for s in ["HE", "PAS", "MT"])]
    sm = sm[sm["patient_id"].astype(str).isin(valid_p)]
    
    model = CDSSDeploymentModel(in_dim=embed_dim).to(device)
    weights = torch.load(ROOT / "results/cdss_v1/cdss_v1.1_final.pt", map_location=device, weights_only=True)
    model.load_state_dict(weights)
    model.eval()
    
    y_true_kdigo = []
    y_prob_kdigo = []
    
    with torch.no_grad():
        for _, r in sm.iterrows():
            pid = str(r["patient_id"])
            mats = [torch.from_numpy(v) for s, v in bags[pid].items() if v is not None and v.shape[0] > 0]
            if not mats: continue
            
            x = torch.cat(mats, dim=0).to(device)
            out = model(x)
            
            if pd.notna(r["target_kdigo"]):
                y_true_kdigo.append(r["kdigo_binary"])
                # We use the calibrated risk prob as the predicted probability for Severe KDIGO
                y_prob_kdigo.append(out["calibrated_risk_prob"].item())
                
    y_true = np.array(y_true_kdigo)
    y_prob = np.array(y_prob_kdigo)
    
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ece = compute_ece(y_true, y_prob, n_bins=10)
    
    print("=== MFDS SaMD v1.1 Validation Metrics ===")
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    print(f"Brier Score: {brier:.4f}")
    print(f"ECE (Expected Calibration Error): {ece:.4f}")
    
    # Generate Reliability Curve (Calibration Plot)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
    
    plt.figure(figsize=(8, 8))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='CDSS v1.1 Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives (Empirical)', fontsize=12)
    plt.title(f'Reliability Curve (Calibration Plot)\nECE: {ece:.4f} | Brier: {brier:.4f}', fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    out_path = ROOT / "results/cdss_v1/mfds_calibration_plot.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved calibration plot to {out_path}")

if __name__ == "__main__":
    main()
