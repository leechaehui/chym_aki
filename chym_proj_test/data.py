import pandas as pd
import numpy as np

np.random.seed(42)

n = 2000  # 데이터 개수 (적어도 1000 이상 추천)

df = pd.DataFrame({
    "subject_id": np.arange(1, n+1),
    "stay_id": np.arange(1000, 1000+n),
    "prediction_cutoff": pd.date_range("2024-01-01", periods=n, freq="h"),

    # 주요 feature들
    "creatinine_max": np.random.normal(1.5, 0.7, n).clip(0, 10),
    "bun_max": np.random.normal(25, 15, n).clip(0, 150),
    "potassium_max": np.random.normal(4.5, 0.7, n).clip(2, 7),
    "map_mean": np.random.normal(75, 15, n).clip(20, 200),
    "urine_output_sum": np.random.normal(1200, 500, n).clip(0, 5000),
})

# AKI label 생성 (feature 기반으로 현실적으로 만들기)
def generate_label(row):
    score = 0

    # 더 공격적으로 조건 설정
    if row["creatinine_max"] > 1.8: score += 1
    if row["creatinine_max"] > 2.5: score += 1
    if row["creatinine_max"] > 3.5: score += 1

    if row["bun_max"] > 30: score += 1
    if row["bun_max"] > 50: score += 1

    if row["potassium_max"] > 5.0: score += 1
    if row["potassium_max"] > 5.8: score += 1

    if row["map_mean"] < 70: score += 1
    if row["map_mean"] < 60: score += 1

    if row["urine_output_sum"] < 1000: score += 1
    if row["urine_output_sum"] < 600: score += 1

    # 🔥 분류 기준 완화
    if score <= 1:
        return 0
    elif score <= 3:
        return 1
    elif score <= 5:
        return 2
    else:
        return 3

df["aki_label"] = df.apply(generate_label, axis=1)

# 저장
df.to_csv("data.csv", index=False)

print("데이터 생성 완료")
print(df["aki_label"].value_counts())