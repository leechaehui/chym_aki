import pandas as pd
import pickle
import os

from sklearn.preprocessing import RobustScaler

# outputs 폴더 없으면 생성
os.makedirs("outputs", exist_ok=True)

# =====================================
# 파일 로드(이상치+결측치 처리가 된 파일로 사용)
# =====================================

train_df = pd.read_csv(
    "../02_Data Preprocessing_missing/train_missing_processed.csv"
)

valid_df = pd.read_csv(
    "../02_Data Preprocessing_missing/valid_missing_processed.csv"
)

test_df = pd.read_csv(
    "../02_Data Preprocessing_missing/test_missing_processed.csv"
)

# =====================================
# 설정
# =====================================

TARGET_COL = "aki_label"

EXCLUDED_COLS = [
    "stay_id",
    "subject_id",
    "hadm_id",
    "index_time",
    "prediction_cutoff",
    "aki_onset_time"
]

# =====================================
# 스케일링 제외 컬럼
# =====================================

exclude_scaling_cols = (
    EXCLUDED_COLS
    + [TARGET_COL]
    + ["aki_stage"]
    + ["gender"]
)

# =====================================
# 스케일링 대상 컬럼 선정
# =====================================

scale_cols = []

for col in train_df.columns:

    # 제외 컬럼
    if col in exclude_scaling_cols:
        continue

    # Missing Indicator 제외
    if col.endswith("_missing"):
        continue

    # 숫자형만
    if pd.api.types.is_numeric_dtype(
        train_df[col]
    ):
        scale_cols.append(col)

# =====================================
# Robust Scaling
# =====================================

scaler = RobustScaler()

train_df[scale_cols] = scaler.fit_transform(
    train_df[scale_cols]
)

valid_df[scale_cols] = scaler.transform(
    valid_df[scale_cols]
)

test_df[scale_cols] = scaler.transform(
    test_df[scale_cols]
)

# =====================================
# 스케일러 저장
# =====================================

with open("outputs/scaler.pkl", "wb") as f:
    pickle.dump(
        {
            "scaler": scaler,
            "scale_cols": scale_cols
        },
        f
    )

print("scaler.pkl saved")

# =====================================
# Scaling Verification
# =====================================

print("\n" + "=" * 60)
print("SCALING VERIFICATION")
print("=" * 60)

verify_cols = [
    'age',
    'map_mean',
    'creatinine_max',
    'bun_max',
    'lactate_max'
]

for col in verify_cols:

    print(f"\n[{col}]")

    print(
        f"Median : {train_df[col].median():.4f}"
    )

    print(
        f"Q1     : {train_df[col].quantile(0.25):.4f}"
    )

    print(
        f"Q3     : {train_df[col].quantile(0.75):.4f}"
    )

    print(
        f"Min    : {train_df[col].min():.4f}"
    )

    print(
        f"Max    : {train_df[col].max():.4f}"
    )
# =====================================
# 스케일링 전후 비교
# =====================================
train_before = pd.read_csv(
    "../02_Data Preprocessing_missing/train_missing_processed.csv"
)

print("\n" + "=" * 60)
print("BEFORE / AFTER COMPARISON")
print("=" * 60)

compare_cols = [
    'age',
    'creatinine_max',
    'lactate_max'
]

for col in compare_cols:

    print(f"\n[{col}]")

    print(
        f"Before Median : "
        f"{train_before[col].median():.4f}"
    )

    print(
        f"After Median  : "
        f"{train_df[col].median():.4f}"
    )

# =====================================
# 스케일링 결과 저장
# =====================================

train_df.to_csv(
    "outputs/train_final.csv",
    index=False
)

valid_df.to_csv(
    "outputs/valid_final.csv",
    index=False
)

test_df.to_csv(
    "outputs/test_final.csv",
    index=False
)



# =====================================
# Summary
# =====================================

print("\n" + "=" * 60)
print("SCALING SUMMARY")
print("=" * 60)

print(f"Scaled Columns : {len(scale_cols)}")
print(scale_cols)

print(f"\nTrain Shape : {train_df.shape}")
print(f"Valid Shape : {valid_df.shape}")
print(f"Test Shape  : {test_df.shape}")