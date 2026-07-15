import os
import json

res_dir = 'C:/team/chym_aki/Pathology_model/results/01_baseline_cv'
for f in os.listdir(res_dir):
    if f.endswith('.json'):
        with open(os.path.join(res_dir, f), 'r') as file:
            try:
                data = json.load(file)
                cfg = data.get('config', {})
                print(f"Tag: {cfg.get('tag')}, Encoder: {cfg.get('encoder')}, Mags: {cfg.get('mags')}, Cohort: {cfg.get('cohort')}")
            except Exception:
                pass
