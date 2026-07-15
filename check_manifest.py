import pandas as pd
m = pd.read_csv('C:/team/chym_aki/patches_manifest.csv')
sub = m[m.patient_id == '30-10034']
print(sub.groupby(['stain', 'slide_id', 'magnification']).size())
