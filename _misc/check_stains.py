import pandas as pd
manifest = pd.read_csv('C:/team/chym_aki/patches_manifest.csv')
print(manifest['stain'].value_counts())
