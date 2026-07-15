import os
import pandas as pd

directories = [
    'C:/team/chym_aki/Pathology_model/results',
    'C:/team/chym_aki/data/eval'
]

output = []

for d in directories:
    for root, dirs, files in os.walk(d):
        for file in files:
            if file.endswith('.csv'):
                filepath = os.path.join(root, file)
                try:
                    df = pd.read_csv(filepath)
                    output.append(f"\n--- FILE: {filepath} ---")
                    output.append(f"ROWS: {len(df)} | COLUMNS: {list(df.columns)}")
                    output.append("FIRST 3 ROWS:")
                    output.append(df.head(3).to_string())
                except Exception as e:
                    output.append(f"\n--- FILE: {filepath} --- ERROR: {e}")

with open('C:/team/chym_aki/all_experiments_dump.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output))

print("Dumped all experiments to C:/team/chym_aki/all_experiments_dump.txt")
