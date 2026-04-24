# 모델링 과정에서 반복적으로 사용하는 공통 함수들을 모아둔 파일입니다.
#
# 이 파일의 핵심 역할
# 1. 모델에 넣을 수 있는 숫자형 feature 컬럼을 자동으로 선별
# 2. 결측 여부 자체를 정보로 활용하기 위해 missing indicator 생성
# 3. 결측값을 -1로 대체하여 모델 학습 가능하도록 처리
# 4. 임상적으로 비정상적인 이상값을 안전한 범위로 클리핑
# 5. prediction_cutoff 기준으로 시간순 train/valid/test 분할
# 6. AKI 양성/음성 클래스 불균형 보정값 계산

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


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    모델 학습 전에 데이터 전처리를 수행하는 함수입니다.

    수행 내용
    1. 모델에 사용할 숫자형 feature 자동 선택
    2. 결측 여부 지시변수 생성
    3. 주요 임상 변수 이상치 클리핑
    4. 숫자형 결측값을 -1로 대체

    - 결측값을 그냥 버리지 않는 이유:
      의료 데이터에서는 “검사를 하지 않았다”는 사실 자체가 환자 상태를 반영할 수 있습니다.
      예를 들어 lactate는 위험한 환자에게 더 자주 측정될 수 있습니다.

    - -1로 채우는 이유:
      대부분의 생리학적 수치는 음수가 될 수 없기 때문에,
      -1은 “측정값 없음”을 나타내는 별도 신호처럼 사용할 수 있습니다.
    """
    df = df.copy()

    feature_cols = _get_model_feature_columns(df)

    # 1. 결측 여부 지시변수 생성
    # 예: lactate_max가 비어 있으면 lactate_max_missing = 1
    for col in feature_cols:
        if df[col].isna().sum() > 0:
            df[f"{col}_missing"] = df[col].isna().astype(int)

    # 2. 이상치 클리핑
    # SQL에서 1차적으로 이상치를 제거했더라도,
    # Python 단계에서 한 번 더 안전장치를 둡니다.
    if "lactate_max" in df.columns:
        df["lactate_max"] = df["lactate_max"].clip(0, 30)

    if "hemoglobin_min" in df.columns:
        df["hemoglobin_min"] = df["hemoglobin_min"].clip(0, 25)

    if "map_min" in df.columns:
        df["map_min"] = df["map_min"].clip(20, 200)

    if "map_mean" in df.columns:
        df["map_mean"] = df["map_mean"].clip(20, 200)

    # 3. 숫자형 컬럼 전체 결측값 대체
    # 새로 만든 *_missing 컬럼까지 포함해서 숫자형 컬럼에 NaN이 남지 않도록 합니다.
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


def compute_scale_pos_weight(y):
    """
    XGBoost에서 클래스 불균형을 보정하기 위한 scale_pos_weight 값을 계산합니다.

    AKI 예측에서는 보통 AKI 발생 환자(1)가 비발생 환자(0)보다 적습니다.
    이 상태에서 그냥 학습하면 모델이 “대부분 0으로 예측”하는 방향으로 치우칠 수 있습니다.

    scale_pos_weight = 음성 클래스 수 / 양성 클래스 수

    예:
    - AKI 없음(0): 900명
    - AKI 있음(1): 100명
    - scale_pos_weight = 900 / 100 = 9

    즉, AKI 환자 1명을 더 중요하게 보도록 가중치를 주는 방식입니다.
    """
    y = pd.Series(y)

    pos = (y == 1).sum()
    neg = (y == 0).sum()

    if pos == 0:
        raise ValueError("양성 클래스가 0개라 scale_pos_weight를 계산할 수 없습니다.")
    if neg == 0:
        raise ValueError("음성 클래스가 0개라 scale_pos_weight를 계산할 수 없습니다.")

    return neg / pos