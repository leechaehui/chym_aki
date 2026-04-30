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
        "aki_stage",
        "aki_onset_time",
        "prediction_cutoff",
        "index_time",
        "subject_id",
        "hadm_id",
        "stay_id",
        "gender",
        "nlp_text_combined",
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

    이미 숫자로 구성되어 있으므로 문자열 변환은 하지 않고,
    값이 범위 안에 있는지만 확인합니다.
    """
    df = df.copy()

    if TARGET not in df.columns:
        raise ValueError(f"{TARGET} 컬럼이 없습니다.")

    df[TARGET] = df[TARGET].astype(int)

    valid_values = {0, 1}
    label_values = set(df[TARGET].dropna().unique())

    if not label_values.issubset(valid_values):
        raise ValueError("aki_label은 0 또는 1이어야 합니다.")

    if df[TARGET].isna().sum() > 0:
        raise ValueError("aki_label에 결측값이 있습니다.")

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
    print("\n=== [Preprocess] 시작 ===")
    # 1. label 체크
    df = encode_target_label(df)
    print("label 분포:")
    print(df[TARGET].value_counts())

    # 2. feature 선택
    feature_cols = _get_model_feature_columns(df)

    print("\n사용될 feature 개수:", len(feature_cols))
    print("feature 일부:", feature_cols[:10])
    # 3. missing flag 생성
    # 예: creatinine_mean이 NaN이면 creatinine_mean_missing = 1
    # 값이 있으면 creatinine_mean_missing = 0
    missing_count = 0

    for col in feature_cols:
        if df[col].isna().sum() > 0:
            df[f"{col}_missing"] = df[col].isna().astype(int)
            missing_count += 1

    print("\nmissing flag 생성된 컬럼 수:", missing_count)
    # 이상치 클리핑
    # 필요한 변수는 같은 형식으로 계속 추가 가능
    CLIP_RANGES = {
        "creatinine_min": (0, 15),
        "creatinine_max": (0, 15),
        "creatinine_mean": (0, 15),
        "bun_max": (0, 200),
        "bun_mean": (0, 200),
        "bicarbonate_min": (0, 50),
        "bicarbonate_mean": (0, 50),
        "potassium_max": (2, 7),
        "potassium_mean": (2, 7),
        "hemoglobin_min": (0, 25),
        "hemoglobin_mean": (0, 25),
        "lactate_max": (0, 30),
        "lactate_mean": (0, 30),
        "hr_max": (20, 250),
        "hr_mean": (20, 250),
        "rr_max": (0, 80),
        "rr_mean": (0, 80),
        "spo2_min": (0, 100),
        "spo2_mean": (0, 100),
        "temp_max": (86, 115),
        "temp_mean": (86, 115),
        "sbp_min": (40, 300),
        "sbp_mean": (40, 300),
        "map_min": (20, 200),
        "map_mean": (20, 200),
        "urine_48h": (0, 50000),
        "fluid_48h": (0, 100000),
        "fluid_flag": (0, 1),
        "vasopressor_flag": (0, 1),
    }

    for col, (low, high) in CLIP_RANGES.items():
        if col in df.columns:
            df[col] = df[col].clip(low, high)
    # 숫자형 컬럼의 NaN을 -1로 대체
    # 원래 값이 있는 경우는 그대로 유지됨
    # NaN만 -1로 바뀜
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(-1)

    print("\nNaN 처리 완료")

    print("=== [Preprocess 완료] ===\n")
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

    print("\n=== [Split 시작] ===")

    if "index_time" not in df.columns:
        raise ValueError("index_time 컬럼이 없습니다. final_features_48h 테이블을 다시 확인하세요.")

    df["index_time"] = pd.to_datetime(df["index_time"], errors="coerce")
    df = df[df["index_time"].notnull()]
    df = df.sort_values("index_time")

    n = len(df)

    print("전체 데이터:", len(df))

    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)

    train_df = df.iloc[:train_end]
    valid_df = df.iloc[train_end:valid_end]
    test_df = df.iloc[valid_end:]

    feature_cols = _get_model_feature_columns(df)

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET]

    X_valid = valid_df[feature_cols]
    y_valid = valid_df[TARGET]

    X_test = test_df[feature_cols]
    y_test = test_df[TARGET]

    print("\nlabel 분포")
    print("train:\n", y_train.value_counts())
    print("valid:\n", y_valid.value_counts())
    print("test:\n", y_test.value_counts())

    print("=== [Split 완료] ===\n")

    return X_train, X_valid, X_test, y_train, y_valid, y_test

def compute_scale_pos_weight(y):
    y = pd.Series(y)

    pos = (y == 1).sum()
    neg = (y == 0).sum()

    if pos == 0:
        raise ValueError("AKI=1이 0개라 scale_pos_weight를 계산할 수 없습니다.")
    if neg == 0:
        raise ValueError("AKI=0이 0개라 scale_pos_weight를 계산할 수 없습니다.")

    return neg / pos

