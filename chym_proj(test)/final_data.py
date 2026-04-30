import pandas as pd

df_struct = pd.read_csv("final_features_48h.csv")
df_nlp = pd.read_csv("nlp_keyword_features.csv")

df_final = df_struct.merge(df_nlp, on="stay_id", how="left")

df_final.to_csv("final_features.csv", index=False)