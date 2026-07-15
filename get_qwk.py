import pandas as pd
df78 = pd.read_csv('C:/team/chym_aki/data/eval/attention/ati_exp_compare_N78.csv')
df83 = pd.read_csv('C:/team/chym_aki/data/eval/attention/ati_exp_compare.csv')
print("N=78 Data:")
print(df78[['mode', 'QWK_5class', 'MAE_class']])
print("N=83 Data:")
print(df83[['mode', 'QWK_5class', 'MAE_class']])
