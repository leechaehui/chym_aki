import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
RES=Path("D:/cdss_core/results/cdss_v5")
sm=pd.read_csv("C:/team/chym_aki/split_manifest.csv").drop_duplicates("patient_id"); sm["patient_id"]=sm["patient_id"].astype(str)
sm["t_ais"]=sm["task_ati_severity"].apply(lambda v: {1:0.0,3:0.5,5:1.0}.get(v,np.nan) if pd.notna(v) else np.nan)
sm["t_cds"]=sm["task_chronic"].astype(float); sm["t_ins"]=sm["task_immune"].astype(float)
def mk(v):
    if pd.isna(v): return np.nan
    s=str(v).lower(); return 3.0 if '3' in s else 2.0 if '2' in s else 1.0 if '1' in s else 0.0
sm["t_kdigo"]=sm["kdigo_stage"].apply(mk)
tgt=sm.set_index("patient_id")[["t_ais","t_cds","t_ins","t_kdigo"]]
def auroc(y,p):
    m=~y.isna(); return (roc_auc_score(y[m],p[m]) if m.sum()>=3 and y[m].nunique()>1 else np.nan)
def spear(y,p):
    m=~y.isna(); return (spearmanr(y[m],p[m]).correlation if m.sum()>=3 and y[m].nunique()>1 else np.nan)
rows=[]
for tag,label in [("vfinal_97","10x"),("vfind40","40x"),("vfind1040","10,40 multiscale")]:
    o=pd.read_csv(RES/f"oof_{tag}.csv"); o["patient_id"]=o["patient_id"].astype(str)
    d=o.set_index("patient_id").join(tgt)
    rows.append({"model":label,
      "INS_AUROC(immune)":round(auroc(d.t_ins,d.pred_ins),3),
      "CDS_AUROC(chronic)":round(auroc(d.t_cds,d.pred_cds),3),
      "AIS_Spearman(ATIsev)":round(spear(d.t_ais,d.pred_ais),3),
      "PSI_Spearman(KDIGO)":round(spear(d.t_kdigo,d.pred_psi),3)})
res=pd.DataFrame(rows)
out=Path("C:/team/chym_aki/data/eval/attention/oof_compare_3way.csv"); res.to_csv(out,index=False)
print(res.to_string(index=False)); print("\nsaved ->",out)
