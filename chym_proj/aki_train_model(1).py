# AKI 예측 모델을 실제로 학습시키는 메인 파일입니다.
#
# 전체 흐름
# 1. data.csv 불러오기
# 2. 결측/이상치 전처리
# 3. 시간순 train/valid/test 분할
# 4. 클래스 불균형 보정값 계산
# 5. XGBoost 하이퍼파라미터 탐색(GridSearchCV)
# 6. 최적 모델 재학습
# 7. 모델과 데이터 분할 결과 저장
import joblib
import pandas as pd
from aki_utils import preprocess_data, time_based_split
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
# 직접 만든 유틸 함수 불러오기
# - load_data: CSV 읽어서 X, y 분리
# - split_data: train / valid / test 분리
# - add_missing_indicators: 결측 여부 파생변수 생성
# - compute_scale_pos_weight: 클래스 불균형 대응용 가중치 계산
from aki_utils import preprocess_data, time_based_split, compute_scale_pos_weight
# 결측 파생변수 실행하려면 add_missing_indicators 추가
# 설정값 불러오기
# - MODEL_PATH: 학습된 모델 저장 경로
# - DATA_SPLIT_PATH: 분할된 데이터 저장 경로
# - RANDOM_STATE: 랜덤 고정값
# - N_JOBS: 병렬 처리 개수
# - CV_FOLDS: 교차검증 fold 수
# - SCORING: GridSearchCV 평가 기준
from aki_config import (MODEL_PATH,DATA_SPLIT_PATH,RANDOM_STATE,N_JOBS,CV_FOLDS,SCORING)

def main():
    # 1. 데이터 로드
    # SQL에서 만든 최종 feature 테이블을 CSV로 저장한 파일입니다.
    # 반드시 aki_label, prediction_cutoff 컬럼이 있어야 합니다.
    df = pd.read_csv("data.csv")

    # 2. 전처리
    # - 결측 여부 indicator 생성
    # - 결측값 -1 대체
    # - 일부 임상 변수 이상치 클리핑
    df = preprocess_data(df)

    # 3. 시간 기반 데이터 분할
    # 과거 시점은 train, 중간 시점은 valid, 가장 나중 시점은 test로 사용합니다.
    # 이렇게 해야 미래 데이터가 학습에 섞이는 data leakage를 줄일 수 있습니다.
    X_train, X_valid, X_test, y_train, y_valid, y_test = time_based_split(df)

    # 4. 클래스 불균형 보정
    # AKI 발생 환자가 적을 가능성이 있으므로 양성 클래스에 더 큰 가중치를 줍니다.
    scale_pos_weight = compute_scale_pos_weight(y_train)

    # 5. 기본 XGBoost 모델 설정
    # objective="binary:logistic": 0/1 이진분류 문제
    # eval_metric="logloss": 학습 중 확률 예측 오차를 평가
    # scale_pos_weight: AKI 양성 클래스 불균형 보정
    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        scale_pos_weight=scale_pos_weight
    )

    # 6. 하이퍼파라미터 탐색
    # GridSearchCV가 아래 조합을 모두 실험하면서 가장 좋은 조합을 찾습니다.
    # n_estimators: 트리 개수
    # max_depth: 트리 깊이. 너무 깊으면 과적합 위험이 있습니다.
    # learning_rate: 한 번에 얼마나 크게 학습할지 정하는 값입니다.
    # subsample: 행 샘플링 비율. 과적합 방지에 도움됩니다.
    # colsample_bytree: feature 샘플링 비율. 과적합 방지에 도움됩니다.
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }

    # 7. Grid Search + 교차검증 설정
    # cv=5이면 train set을 5조각으로 나눠가며 검증합니다.
    # scoring="roc_auc"는 AKI/Non-AKI를 얼마나 잘 구분하는지 기준으로 모델을 고릅니다.
    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=SCORING,
        cv=CV_FOLDS,
        n_jobs=N_JOBS,
        verbose=1
    )
    grid.fit(X_train, y_train)

    print("Best Params:", grid.best_params_)
    print("Best CV Score:", grid.best_score_)

    # 8. 최적 파라미터로 재학습 + early stopping
    # early_stopping_rounds=20:
    # validation 성능이 20번 연속 좋아지지 않으면 학습을 멈춰 과적합을 줄입니다.
    best_params = grid.best_params_
    final_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=20,
        **best_params
    )

    # 9. 최종 모델 학습
    # train으로 학습하고, valid를 보면서 성능을 확인
    final_model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )

    # 10. 저장
    # 나중에 evaluation, SHAP, threshold tuning 코드에서 재사용 가능
    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(
        (X_train, X_valid, X_test, y_train, y_valid, y_test),
        DATA_SPLIT_PATH
    )

    print("✅ 모델 학습 및 저장 완료")


if __name__ == "__main__":
    main()