# AKI 예측 모델을 실제로 학습시키는 메인 파일입니다.
#
# 전체 흐름
# 1. data.csv 불러오기
# 2. 결측/이상치 전처리
# 3. 시간순 train/valid/test 분할
# 4. Optuna로 XGBoost 하이퍼파라미터 탐색
# 5. 최적 모델 재학습
# 6. 모델과 데이터 분할 결과 저장
import joblib
import pandas as pd
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
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
from aki_config import DATA_PATH,MODEL_PATH,DATA_SPLIT_PATH,RANDOM_STATE, N_JOBS,N_TRIALS

def main():
    # 1. 데이터 로드
    # SQL에서 만든 최종 feature 테이블을 CSV로 저장한 파일입니다.
    # 반드시 aki_label, prediction_cutoff 컬럼이 있어야 합니다.
    df = pd.read_csv(DATA_PATH)

    print("=== 1. Raw Data Check ===")
    print("데이터 크기:", df.shape)
    print("컬럼 목록:")
    print(df.columns.tolist())

    print("\naki_label 분포:")
    print(df["aki_label"].value_counts().sort_index())
    # 2. 전처리
    # - 결측 여부 indicator 생성
    # - 결측값 -1 대체
    # - 일부 임상 변수 이상치 클리핑
    df = preprocess_data(df)

    print("\n=== 2. After Preprocessing ===")
    print("전처리 후 데이터 크기:", df.shape)
    print("전처리 후 결측치 개수:")
    print(df.isna().sum().sort_values(ascending=False).head(20))
    # 3. 시간 기반 데이터 분할
    # 과거 시점은 train, 중간 시점은 valid, 가장 나중 시점은 test로 사용합니다.
    # 이렇게 해야 미래 데이터가 학습에 섞이는 data leakage를 줄일 수 있습니다.
    X_train, X_valid, X_test, y_train, y_valid, y_test = time_based_split(df)

    print("\n=== 3. Split Check ===")
    print("X_train:", X_train.shape)
    print("X_valid:", X_valid.shape)
    print("X_test :", X_test.shape)

    print("\ntrain label")
    print(y_train.value_counts().sort_index())

    print("\nvalid label")
    print(y_valid.value_counts().sort_index())

    print("\ntest label")
    print(y_test.value_counts().sort_index())

    print("\n=== Feature 확인 ===")
    print("feature 개수:", X_train.shape[1])
    print("feature 샘플:", X_train.columns[:10].tolist())

    print("\n=== 데이터 샘플 확인 ===")
    print(X_train.head())

    print("\n=== label imbalance ===")
    print("pos:", (y_train == 1).sum())
    print("neg:", (y_train == 0).sum())
    print(X_train.columns.tolist())

    # 4. 클래스 불균형 가중치 계산
    scale_pos_weight = compute_scale_pos_weight(y_train)

    print("\n=== 4. Class Weight Check ===")
    print("scale_pos_weight:", scale_pos_weight)
    # 5. Optuna objective 함수 정의
    # trial마다 다른 하이퍼파라미터 조합으로 XGBoost를 학습하고
    # validation AUROC가 가장 높은 조합을 찾습니다.
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0),
        }

        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            scale_pos_weight=scale_pos_weight,
            early_stopping_rounds=20,
            **params
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False
        )

        y_prob_valid = model.predict_proba(X_valid)[:, 1]

        auc = roc_auc_score(y_valid, y_prob_valid)

        return auc

    print("\n=== 5. Optuna Search Start ===")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)

    print("\nBest Params:")
    print(study.best_params)

    print("\nBest Validation AUROC:")
    print(study.best_value)

    # 6. 최적 파라미터로 최종 모델 학습

    final_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=20,
        **study.best_params,
    )

    final_model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )
    # 7. 모델과 분할 데이터 저장
    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(
        (X_train, X_valid, X_test, y_train, y_valid, y_test),
        DATA_SPLIT_PATH
    )

    print("✅ final_features_48h 기반 Binary AKI XGBoost 모델 학습 완료")
    print("모델 저장 경로:", MODEL_PATH)
    print("분할 데이터 저장 경로:", DATA_SPLIT_PATH)

if __name__ == "__main__":
    main()