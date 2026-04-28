# 모델링 과정에서 반복적으로 사용하는 공통 함수들을 모아둔 파일입니다.
#
# 이 파일의 핵심 역할
# 1. 모델에 넣을 수 있는 숫자형 feature 컬럼을 자동으로 선별
# 2. 결측 여부 자체를 정보로 활용하기 위해 missing indicator 생성
# 3. 결측값을 -1로 대체하여 모델 학습 가능하도록 처리
# 4. 임상적으로 비정상적인 이상값을 안전한 범위로 클리핑
# 5. prediction_cutoff 기준으로 시간순 train/valid/test 분할

import pandas as pd
from aki_config import TARGET


def _get_model_feature_columns(df: pd.DataFrame):
    """
    모델에 넣을 수 있는 feature 컬럼만 자동으로 고르는 내부 함수입니다.

    제외하는 컬럼
    - aki_label: 정답값이므로 X에 들어가면 안 됩니다. 들어가면 data leakage입니다.
    - prediction_cutoff: 시간 분할 기준으로만 사용합니다.
    - subject_id, hadm_id, stay_id: 식별자입니다.
    - icu_intime, icu_outtime, aki_onset_time: 시간/이벤트 정보입니다.
    - first_careunit, gender: 현재는 문자열일 수 있어 제외합니다.

    숫자형 컬럼만 선택하는 이유
    - XGBoost, Logistic Regression, RandomForest는 기본적으로 숫자형 입력을 기대합니다.
    - 문자열이나 datetime이 들어가면 에러가 날 수 있습니다.
    """
    exclude_cols = [
        TARGET,
        "prediction_cutoff",
        "aki_onset_time",
        "subject_id",
        "hadm_id",
        "stay_id",
        "icu_intime",
        "icu_outtime",
        "first_careunit",
        "gender",
    ]

    feature_cols = [
        col for col in df.columns
        if col not in exclude_cols
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    return feature_cols

def encode_target_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    aki_label 검증 함수입니다.

    현재 라벨 구조:
    0 = AKI 없음
    1 = stage1
    2 = stage2
    3 = stage3

    이미 숫자로 구성되어 있으므로 문자열 변환은 하지 않고,
    값이 0,1,2,3 범위 안에 있는지만 확인합니다.
    """
    df = df.copy()

    if TARGET not in df.columns:
        raise ValueError(f"{TARGET} 컬럼이 없습니다.")

    if not pd.api.types.is_numeric_dtype(df[TARGET]):
        raise ValueError("aki_label은 숫자형이어야 합니다. 값은 0,1,2,3이어야 합니다.")

    valid_values = {0, 1, 2, 3}
    label_values = set(df[TARGET].dropna().unique())

    if not label_values.issubset(valid_values):
        raise ValueError("aki_label 값이 0,1,2,3 범위를 벗어났습니다.")

    if df[TARGET].isna().sum() > 0:
        raise ValueError("aki_label에 결측값이 있습니다.")

    df[TARGET] = df[TARGET].astype(int)

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    모델 학습 전 전처리 함수입니다.

    수행 내용:
    1. aki_label 검증
    2. 모델 feature 컬럼 자동 선택
    3. 결측 여부 지시변수 생성
    4. 이상치 클리핑
    5. 숫자형 결측값 -1 대체
    """
    df = df.copy()

    # 정답 라벨 검증
    df = encode_target_label(df)

    # 모델 feature 컬럼 선택
    feature_cols = _get_model_feature_columns(df)

    # 결측 여부 지시변수 생성
    for col in feature_cols:
        if df[col].isna().sum() > 0:
            df[f"{col}_missing"] = df[col].isna().astype(int)

    # 이상치 클리핑
    # 필요한 변수는 같은 형식으로 계속 추가 가능
    CLIP_RANGES = {
        "lactate_max": (0, 30),
        "hemoglobin_min": (0, 25),
        "map_min": (20, 200),
        "map_mean": (20, 200),
        "creatinine_max": (0, 15),
        "creatinine_mean": (0, 15),
        "bun_max": (0, 200),
        "bun_mean": (0, 200),
        "potassium_max": (2, 7),
        "potassium_mean": (2, 7),
        "sodium_min": (100, 180),
        "sodium_mean": (100, 180),
        "glucose_max": (0, 1000),
        "glucose_mean": (0, 1000),
        "urine_output_sum": (0, 10000),
    }

    for col, (low, high) in CLIP_RANGES.items():
        if col in df.columns:
            df[col] = df[col].clip(low, high)

    # 숫자형 결측값 대체
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(-1)

    return df



def time_based_split(df: pd.DataFrame):
    """
    prediction_cutoff 기준으로 train / valid / test를 시간순으로 나누는 함수입니다.

    왜 랜덤 분할이 아니라 시간순 분할을 쓰는가?
    - 의료 예측 모델은 과거 데이터로 학습해서 미래 환자에게 적용하는 것이 목표입니다.
    - 랜덤 분할을 하면 미래 환자와 과거 환자가 섞여 실제보다 성능이 좋게 보일 수 있습니다.
    - 따라서 prediction_cutoff 기준으로 정렬한 뒤 앞 70%는 train,
      다음 15%는 validation, 마지막 15%는 test로 사용합니다.

    반환값
    - X_train, X_valid, X_test: 모델 입력 변수
    - y_train, y_valid, y_test: 정답 label
    """
    df = df.copy()

    # prediction_cutoff가 없는 행은 시간순 분할 기준이 없으므로 제외합니다.
    df = df[df["prediction_cutoff"].notnull()]

    # 시간 기준으로 정렬합니다.
    df = df.sort_values("prediction_cutoff")

    n = len(df)

    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)

    train_df = df.iloc[:train_end]
    valid_df = df.iloc[train_end:valid_end]
    test_df = df.iloc[valid_end:]

    # preprocess_data와 동일한 기준으로 feature 컬럼을 선택합니다.
    feature_cols = _get_model_feature_columns(df)

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET]

    X_valid = valid_df[feature_cols]
    y_valid = valid_df[TARGET]

    X_test = test_df[feature_cols]
    y_test = test_df[TARGET]

    return X_train, X_valid, X_test, y_train, y_valid, y_test


