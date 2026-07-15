import pandas as pd
df = pd.read_csv('C:/team/chym_aki/split_manifest.csv')
print("Columns:", df.columns.tolist())
# Check for ATN or ATI columns
atn_cols = [c for c in df.columns if 'atn' in c.lower() or 'ati' in c.lower()]
print("ATN/ATI Columns:", atn_cols)
# Check label distribution for ordinal classes (fibrosis, atrophy, inflammation)
for col in ['ti_simplification', 'ti_cell_sloughing', 'ti_cellular_necrosis', 'fibrosis', 'atrophy', 'inflammation']:
    if col in df.columns:
        print(f"\nLabel distribution for {col}:")
        print(df[col].value_counts(dropna=False).sort_index())
