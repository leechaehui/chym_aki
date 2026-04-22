#공통 함수
import pandas as pd
from sklearn.model_selection import train_test_split
# config에서 설정값 가져오기
from aki_config import FEATURES, TARGET, TEST_SIZE, VALID_SIZE, RANDOM_STATE

def load_data(path="data.csv"):
    """
      데이터 불러오기 함수
      - CSV 파일을 읽고
      - feature(X)와 target(y) 분리
      """
    df = pd.read_csv(path)
    missing_cols = [col for col in FEATURES + [TARGET] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"다음 컬럼이 데이터에 없습니다: {missing_cols}")

    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    return X, y

# 데이터에 결측치가 있을 때 실행
# def add_missing_indicators(X: pd.DataFrame) -> pd.DataFrame:
#     """
#     결측 자체가 의미가 있을 수 있으므로 결측 여부 파생변수 추가
#     """
#     X = X.copy()
#     for col in X.columns:
#         X[f"{col}_missing"] = X[col].isna().astype(int)
#     return X

def split_data(X, y):
    """
    train / valid / test 분할
    - test: 최종 성능 검증
    - valid: 모델 튜닝용 (threshold, early stopping 등)
    """

    # 1차 분할: train_full vs test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  # 클래스 비율 유지
    )

    # 2차 분할: train vs valid
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full,
        test_size=VALID_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_full
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test

def compute_scale_pos_weight(y):
    """
    클래스 불균형 대응용 가중치 계산

    예:
    0이 900개, 1이 100개 → weight = 9
    """
    y = pd.Series(y)
    pos = (y == 1).sum()
    neg = (y == 0).sum()

    if pos == 0:
        raise ValueError("양성 클래스가 0개라 scale_pos_weight를 계산할 수 없습니다.")
    if neg == 0:
        raise ValueError("음성 클래스가 0개라 scale_pos_weight를 계산할 수 없습니다.")

    return neg / pos