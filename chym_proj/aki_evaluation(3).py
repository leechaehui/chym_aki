# 학습된 모델의 최종 성능을 평가하는 파일입니다.
#
# 이 파일에서 하는 일
# 1. 저장된 train/valid/test 데이터 불러오기
# 2. 저장된 threshold 불러오기
# 3. Logistic Regression, RandomForest, XGBoost 성능 비교
# 4. 최종 XGBoost 성능 상세 평가
# 5. ROC curve, PR curve, Confusion Matrix 시각화
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,          # AUROC 계산
    roc_curve,              # ROC 곡선 좌표 계산
    confusion_matrix,       # 혼동행렬 계산
    ConfusionMatrixDisplay, # 혼동행렬 시각화
    precision_recall_curve, # PR curve 좌표 계산
    classification_report,  # precision / recall / f1-score 요약 출력
    average_precision_score # AUPRC 계산
)
# 설정 파일에서 경로, 랜덤 시드 불러오기
from aki_config import DATA_SPLIT_PATH, MODEL_PATH, THRESHOLD_PATH, RANDOM_STATE


def main():
    # 1. train / valid / test 데이터 불러오기
    # 이전 학습 코드에서 저장해둔 데이터 분할 결과를 그대로 불러온다.
    X_train, X_valid, X_test, y_train, y_valid, y_test = joblib.load(DATA_SPLIT_PATH)
    # 2. validation에서 찾은 최적 threshold 불러오기
    # test 평가 때는 이 threshold를 기준으로 최종 예측값(0/1)을 만든다.
    best_threshold = joblib.load(THRESHOLD_PATH)

    # 3. baseline 모델 비교를 위해 train + valid를 합친다.
    # - Logistic, RandomForest는 이 파일에서 새로 학습하는 비교용 모델
    # - XGBoost 최종 모델과 공정하게 비교하려면 가능한 많은 학습 데이터를 쓰는 것이 좋다.
    X_train_full = X_train.copy()
    X_train_full = pd.concat([X_train, X_valid], axis=0)
    y_train_full = pd.concat([y_train, y_valid], axis=0)

    models = {
        # Logistic Regression:
        # - 선형 기반의 기본 비교 모델
        # - class_weight="balanced"로 클래스 불균형을 일부 보정
        "Logistic": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        # Random Forest:
        # - 트리 기반 앙상블 모델
        # - 역시 class_weight="balanced"로 불균형 보정
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        # XGBoost:
        # - 이미 최적 파라미터로 학습 후 저장해둔 최종 모델을 그대로 불러온다.
        "XGBoost": joblib.load(MODEL_PATH)
    }

    results = {}
    # 각 모델의 AUROC/AUPRC 계산
    for name, model in models.items():
        # 저장된 최종 XGBoost는 재학습하지 않음
        if name != "XGBoost":
            model.fit(X_train_full, y_train_full)

        # test 데이터에 대해 AKI 발생 확률 예측
        y_prob = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "AUROC": roc_auc_score(y_test, y_prob),
            "AUPRC": average_precision_score(y_test, y_prob)
        }

    print("=== Model Comparison ===")
    for k, v in results.items():
        print(k, v)

    # 모델별 AUROC 비교 그래프
    plt.figure()
    plt.bar(results.keys(), [v["AUROC"] for v in results.values()])
    plt.title("Model Comparison (AUROC)")
    plt.ylabel("AUROC")
    plt.show()

    # 최종 XGBoost 상세 평가
    model = models["XGBoost"]
    # test 데이터에 대한 확률 예측
    y_prob = model.predict_proba(X_test)[:, 1]
    # validation에서 정한 threshold를 적용해 0/1 예측값 생성
    y_pred = (y_prob >= best_threshold).astype(int)

    print("\n=== Final XGBoost Report ===")
    print(f"Threshold: {best_threshold:.2f}")
    print("AUROC:", roc_auc_score(y_test, y_prob))
    print("AUPRC:", average_precision_score(y_test, y_prob))
    print(classification_report(y_test, y_pred, digits=3))

    # ROC
    # 전체 threshold에서 민감도/특이도 trade-off를 보는 그래프입니다.
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "--") # 랜덤 분류기 기준선
    plt.legend()
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()

    # PR curve
    # AKI처럼 양성 클래스가 적은 문제에서는 AUPRC도 중요합니다.
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure()
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()

    # Confusion Matrix
    # 실제 0/1과 예측 0/1이 어떻게 맞고 틀렸는지 보여줍니다.
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(f"Confusion Matrix (threshold={best_threshold:.2f})")
    plt.show()


if __name__ == "__main__":
    main()