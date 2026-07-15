import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from pathlib import Path

ROOT = Path("D:/cdss_core")

def map_kdigo(val):
    if pd.isna(val): return float('nan')
    v = str(val).lower()
    if '3' in v: return 3.0
    if '2' in v: return 2.0
    if '1' in v: return 1.0
    return 0.0

def main():
    df = pd.read_csv(ROOT / "results" / "cdss_v5" / "oof_vfinal_97.csv")
    sm = pd.read_csv(ROOT / "split_manifest.csv").drop_duplicates("patient_id")
    df = df.merge(sm[['patient_id', 'task_ati_severity', 'task_chronic', 'kdigo_stage']], on='patient_id', how='left')

    df['kdigo_num'] = df['kdigo_stage'].apply(map_kdigo)

    print("=== CDSS vFinal 9.7 Performance Metrics ===")

    # Evaluate Chronic (CDS vs task_chronic) -> AUROC
    df_chronic = df.dropna(subset=['task_chronic'])
    if len(df_chronic) > 0 and df_chronic['task_chronic'].nunique() > 1:
        auroc_chronic = roc_auc_score(df_chronic['task_chronic'], df_chronic['pred_cds'])
        print(f"Chronic Damage (CDS) AUROC: {auroc_chronic:.3f}")

    # Evaluate ATI (AIS vs task_ati_severity) -> Spearman
    df_ati = df.dropna(subset=['task_ati_severity'])
    if len(df_ati) > 0:
        spearman_ati, _ = spearmanr(df_ati['pred_ais'], df_ati['task_ati_severity'])
        print(f"ATI Severity (AIS) Spearman: {spearman_ati:.3f}")
        
    # Evaluate AKIN (PSI vs KDIGO) -> Spearman & AUROC
    df_kdigo = df.dropna(subset=['kdigo_num'])
    if len(df_kdigo) > 0:
        spearman_akin, _ = spearmanr(df_kdigo['pred_psi'], df_kdigo['kdigo_num'])
        print(f"KDIGO Stage (PSI) Spearman: {spearman_akin:.3f}")
        
        df_kdigo['stage3'] = (df_kdigo['kdigo_num'] >= 3).astype(int)
        if df_kdigo['stage3'].nunique() > 1:
            auroc_akin = roc_auc_score(df_kdigo['stage3'], df_kdigo['pred_psi'])
            print(f"KDIGO Stage 3 (PSI) AUROC: {auroc_akin:.3f}")
            
    print(f"Average Total Uncertainty: {df['total_unc'].mean():.4f}")

if __name__ == "__main__":
    main()
