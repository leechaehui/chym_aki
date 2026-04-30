# 학습된 XGBoost 모델을 SHAP으로 해석하는 파일입니다.
#
# SHAP이란?
# - 모델이 어떤 feature 때문에 AKI 위험을 높게/낮게 예측했는지 설명하는 방법입니다.
# - 단순히 성능만 보는 것이 아니라 "왜 그렇게 예측했는지"를 보여줍니다.
#
# 이 파일에서 보는 것
# 1. 전체 feature 중요도
# 2. feature 값이 위험을 높이는지 낮추는지
# 3. 개별 환자 1명에 대한 예측 이유
import shap
import joblib
import matplotlib.pyplot as plt
import numpy as np

from aki_config import MODEL_PATH, DATA_SPLIT_PATH


def main():
    # 1. 저장된 최종 XGBoost 모델 불러오기
    model = joblib.load(MODEL_PATH)
    # 학습 때 저장한 동일한 데이터 분할 불러오기
    # SHAP 분석은 보통 최종 평가용인 X_test에 대해 수행
    X_train, X_valid, X_test, y_train, y_valid, y_test = joblib.load(DATA_SPLIT_PATH)

    # 2. SHAP explainer 생성
    # TreeExplainer는 XGBoost 같은 tree 기반 모델에 적합한 SHAP explainer입니다.
    explainer = shap.TreeExplainer(model)
    # 3. X_test에 대한 SHAP 값 계산
    # shap_values는 각 샘플, 각 변수별 기여도를 담고 있음
    shap_values = explainer.shap_values(X_test)

    print("X_test shape:", X_test.shape)
    print("SHAP shape:", shap_values.shape)

    # 4. 다중분류 SHAP 시각화
    # XGBoost 다중분류에서는 클래스별 SHAP 값이 나올 수 있습니다.
    shap_values = np.array(shap_values)
    print("SHAP shape:", shap_values.shape)

    shap.summary_plot(
        shap_values,
        X_test,
        plot_type="bar",
        show=False
    )
    plt.title("SHAP Feature Importance - AKI Risk")
    plt.show()

    shap.summary_plot(
        shap_values,
        X_test,
        show=False
    )
    plt.title("SHAP Beeswarm Plot - AKI Risk")
    plt.show()

    print("\n=== SHAP 확인 ===")
    print("X_test shape:", X_test.shape)
    print("SHAP shape:", shap_values.shape)

    print("\nfeature 이름 확인:")
    print(X_test.columns.tolist()[:10])

if __name__ == "__main__":
    main()