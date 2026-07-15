import pandas as pd
df = pd.read_csv('C:/team/chym_aki/Pathology_model/artifacts/descriptor_labels.csv')
print("Columns:", df.columns.tolist())
for col in df.columns:
    if 'necrosis' in col.lower() or 'atn' in col.lower() or 'ati' in col.lower() or 'tubul' in col.lower() or 'slough' in col.lower() or 'fibrosis' in col.lower() or 'atrophy' in col.lower() or 'inflammation' in col.lower() or 'simplification' in col.lower():
        print(f"\nLabel distribution for {col}:")
        print(df[col].value_counts(dropna=False).sort_index())
