import pandas as pd
m = pd.read_csv('C:/team/chym_aki/patches_manifest.csv')
sub = m[(m.patient_id == '30-10034') & (m.stain == 'HE') & (m.magnification == 40)]
print("Min X:", sub.tile_x.min(), "Max X:", sub.tile_x.max())
print("Min Y:", sub.tile_y.min(), "Max Y:", sub.tile_y.max())
