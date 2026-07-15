import pandas as pd
import os
files = ['C:/team/chym_aki/Pathology_model/results/06_data_limitation/repeated_cv_summary.csv',
         'C:/team/chym_aki/data/eval/attention/ati_exp_compare_s512.csv',
         'C:/team/chym_aki/Pathology_model/results/01_baseline_cv/oof_exp10main_ctranspath.csv']
for f in files:
    if os.path.exists(f):
        df = pd.read_csv(f)
        print(f"--- {os.path.basename(f)} ---")
        print(df.head())
