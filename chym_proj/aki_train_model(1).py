#모델 학습
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
# 직접 만든 유틸 함수 불러오기
# - load_data: CSV 읽어서 X, y 분리
# - split_data: train / valid / test 분리
# - add_missing_indicators: 결측 여부 파생변수 생성
# - compute_scale_pos_weight: 클래스 불균형 대응용 가중치 계산
from aki_utils import (load_data,split_data,compute_scale_pos_weight) # 결측 파생변수 실행하려면 add_missing_indicators 추가
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
    X, y = load_data("data.csv")

    # 2. 결측 여부 파생변수 추가
    # 데이터에 결측치 존재하면 사용
    # X = add_missing_indicators(X)

    # 3. 데이터 분할
    # 전체 데이터를 X_train: 모델 학습용 ,X_valid: 성능 점검 / threshold tuning / early stopping용 ,X_test: 최종 평가용으로 나눔
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(X, y)

    # 4. 불균형 대응
    # AKI 양성(1)과 음성(0)의 비율이 불균형할 수 있으므로
    # 양성 클래스에 더 큰 가중치를 주기 위한 scale_pos_weight 계산
    # 예: 음성 900명, 양성 100명이면 scale_pos_weight = 9
    scale_pos_weight = compute_scale_pos_weight(y_train)

    # 5. base model
    # 아직 최적 파라미터를 찾기 전의 base model
    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        scale_pos_weight=scale_pos_weight
    )

    # 6. 하이퍼파라미터 탐색
    # GridSearchCV가 아래 조합을 전부 시험해보면서
    # 가장 좋은 성능의 조합을 찾음
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }

    # 7. Grid Search + 교차검증 설정
    # - estimator=base_model: 기본 모델
    # - param_grid: 시험할 파라미터 조합
    # - scoring: AUROC 같은 평가 지표
    # - cv=CV_FOLDS: 5겹 교차검증
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