import shap
import joblib
import matplotlib.pyplot as plt

from aki_config import MODEL_PATH, DATA_SPLIT_PATH


def main():
    # 1. 저장된 최종 XGBoost 모델 불러오기
    model = joblib.load(MODEL_PATH)
    # 2. train / valid / test 데이터 불러오기
    # SHAP 분석은 보통 최종 평가용인 X_test에 대해 수행
    X_train, X_valid, X_test, y_train, y_valid, y_test = joblib.load(DATA_SPLIT_PATH)

    # 3. SHAP explainer 생성
    # Tree 기반 모델(XGBoost, RandomForest 등)에는 TreeExplainer가 적합
    explainer = shap.TreeExplainer(model)
    # 4. X_test에 대한 SHAP 값 계산
    # shap_values는 각 샘플, 각 변수별 기여도를 담고 있음
    shap_values = explainer.shap_values(X_test)

    # 5. 전체 feature 중요도 시각화
    # plot_type="bar":평균 절대 SHAP 값을 기준으로 어떤 변수가 전체적으로 중요한지 막대그래프로 보여줌
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance")
    plt.show()

    # 6. SHAP beeswarm plot
    # 각 변수의 영향 방향(+/-)과 분포를 동시에 보여줌
    # 예: creatinine 값이 높을수록 AKI 위험을 높이는지 등을 볼 수 있음
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Beeswarm Plot")
    plt.show()

    # 7. 개별 환자 1명에 대한 설명
    # shap_values[0]:X_test의 첫 번째 환자(샘플)에 대한 feature별 기여도
    # waterfall plot은 어떤 변수들이 예측값을 올렸는지 어떤 변수들이 예측값을 내렸는지를 한눈에 보여줌
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value,
        shap_values[0],
        feature_names=X_test.columns
    )


if __name__ == "__main__":
    main()