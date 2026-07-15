import pandas as pd
import numpy as np
import os

# =====================================
# 설정
# =====================================

TRAIN_PATH = "../01_Data Preprocessing_outlier/train_outlier_processed.csv"
VALID_PATH = "../01_Data Preprocessing_outlier/valid_outlier_processed.csv"
TEST_PATH  = "../01_Data Preprocessing_outlier/test_outlier_processed.csv"

train_df = pd.read_csv(TRAIN_PATH)
valid_df = pd.read_csv(VALID_PATH)
test_df  = pd.read_csv(TEST_PATH)

TARGET_COL = "aki_label"

EXCLUDED_COLS = [
    "stay_id",
    "subject_id",
    "hadm_id",
    "index_time",
    "prediction_cutoff",
    "aki_onset_time",
    "aki_stage"
]
# outputs 폴더 없으면 생성
os.makedirs("outputs", exist_ok=True)

# =====================================
# 전처리 대상 컬럼
# =====================================

preprocess_cols = [
    col
    for col in train_df.columns
    if col not in EXCLUDED_COLS + [TARGET_COL]
]

# =====================================
# 변수 타입 분리
# =====================================

numeric_cols = train_df[preprocess_cols].select_dtypes(
    include=["number"]
).columns.tolist()

categorical_cols = train_df[preprocess_cols].select_dtypes(
    include=["object", "string", "category", "bool"]
).columns.tolist()

# =====================================
# Missing Indicator 생성
# =====================================

missing_indicator_cols = []

for col in preprocess_cols:

    if train_df[col].isna().sum() > 0:

        train_df[f"{col}_missing"] = (
            train_df[col]
            .isna()
            .astype(int)
        )

        missing_indicator_cols.append(col)
# =====================================
# Valid / Test Missing Indicator 생성
# =====================================

for col in missing_indicator_cols:

    valid_df[f"{col}_missing"] = (
        valid_df[col]
        .isna()
        .astype(int)
    )

    test_df[f"{col}_missing"] = (
        test_df[col]
        .isna()
        .astype(int)
    )

# =====================================
# Numeric → Median Imputation
# =====================================

median_values = {}

for col in numeric_cols:

    if train_df[col].isna().sum() > 0:

        median_val = train_df[col].median()

        median_values[col] = median_val

        train_df[col] = train_df[col].fillna(
            median_val
        )
# =====================================
# Valid,test데이터 처리
# =====================================

for col, median_val in median_values.items():

    valid_df[col] = valid_df[col].fillna(
        median_val
    )

    test_df[col] = test_df[col].fillna(
        median_val
    )
# =====================================
# Categorical 처리
# =====================================

mode_values = {}
unknown_cols = []

for col in categorical_cols:

    missing_ratio = train_df[col].isna().mean()

    if missing_ratio == 0:
        continue

    # 결측률 <= 20%
    if missing_ratio <= 0.20:

        mode_val = train_df[col].mode()[0]

        mode_values[col] = mode_val

        train_df[col] = train_df[col].fillna(
            mode_val
        )

    # 결측률 > 20%
    else:

        unknown_cols.append(col)

        train_df[col] = train_df[col].fillna(
            "Unknown"
        )
# =====================================
# Valid,test데이터 처리
# =====================================
for col, mode_val in mode_values.items():

    valid_df[col] = valid_df[col].fillna(
        mode_val
    )

    test_df[col] = test_df[col].fillna(
        mode_val
    )

for col in unknown_cols:

    valid_df[col] = valid_df[col].fillna(
        "Unknown"
    )

    test_df[col] = test_df[col].fillna(
        "Unknown"
    )
# =====================================
# 전처리 정보 저장
# =====================================

preprocess_info = {
    "median_values": median_values,
    "mode_values": mode_values,
    "unknown_cols": unknown_cols,
    "missing_indicator_cols": missing_indicator_cols,
    "excluded_cols": EXCLUDED_COLS,
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols
}
# =====================================
# Summary
# =====================================

print("=" * 50)
print("Missing Preprocessing Summary")
print("=" * 50)

print(f"Numeric Columns      : {len(numeric_cols)}")
print(f"Categorical Columns  : {len(categorical_cols)}")

print(f"Missing Indicators   : {len(missing_indicator_cols)}")
print(missing_indicator_cols)

print(f"Median Imputation    : {len(median_values)}")
print(list(median_values.keys()))

print(f"Mode Imputation      : {len(mode_values)}")
print(list(mode_values.keys()))

print(f"Unknown Category     : {len(unknown_cols)}")
print(unknown_cols)


# 추후에 vali,test에 적용시킬 수 있게 저장하는 코드
import pickle

with open(
    "outputs/missing_preprocess.pkl",
    "wb"
) as f:
    pickle.dump(preprocess_info, f)

print("preprocess info saved")

train_df.to_csv(
    "train_missing_processed.csv",
    index=False
)

valid_df.to_csv(
    "valid_missing_processed.csv",
    index=False
)

test_df.to_csv(
    "test_missing_processed.csv",
    index=False
)

# =====================================
# Missing 처리 결과 검증
# =====================================

print("\n" + "=" * 80)
print("MISSING PREPROCESSING VERIFICATION")
print("=" * 80)

# -------------------------------------
# Missing Indicator
# -------------------------------------

print("\n[1] Missing Indicator Columns")

if len(missing_indicator_cols) == 0:
    print("None")

else:

    indicator_df = pd.DataFrame({
        "column": missing_indicator_cols,
        "missing_count": [
            train_df[f"{col}_missing"].sum()
            for col in missing_indicator_cols
        ]
    })

    print(indicator_df)

# -------------------------------------
# Median Imputation
# -------------------------------------

print("\n[2] Median Imputation")

if len(median_values) == 0:
    print("None")

else:

    median_df = pd.DataFrame({
        "column": list(median_values.keys()),
        "median_value": list(median_values.values()),
        "remaining_missing": [
            train_df[col].isna().sum()
            for col in median_values.keys()
        ]
    })

    print(median_df)

# -------------------------------------
# Mode Imputation
# -------------------------------------

print("\n[3] Mode Imputation")

if len(mode_values) == 0:
    print("None")

else:

    mode_df = pd.DataFrame({
        "column": list(mode_values.keys()),
        "mode_value": list(mode_values.values())
    })

    print(mode_df)

# -------------------------------------
# Unknown Category
# -------------------------------------

print("\n[4] Unknown Category")

if len(unknown_cols) == 0:
    print("None")

else:

    unknown_df = pd.DataFrame({
        "column": unknown_cols
    })

    print(unknown_df)

# -------------------------------------
# CSV 저장
# -------------------------------------

if len(missing_indicator_cols) > 0:
    indicator_df.to_csv(
        "outputs/missing_indicator_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

if len(median_values) > 0:
    median_df.to_csv(
        "outputs/median_imputation_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

print("\nVerification files saved.")