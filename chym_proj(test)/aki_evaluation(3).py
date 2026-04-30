# 학습된 모델의 최종 성능을 평가하는 파일입니다.
#
# 이 파일에서 하는 일
# 1. 저장된 train/valid/test 데이터 불러오기
# 2. Logistic Regression, RandomForest, XGBoost 성능 비교
# 3. 최종 XGBoost 성능 상세 평가
# 4. Confusion Matrix 시각화
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,          # AUROC 계산
    roc_curve,
    confusion_matrix,       # 혼동행렬 계산
    ConfusionMatrixDisplay, # 혼동행렬 시각화
    precision_recall_curve,
    classification_report,  # precision / recall / f1-score 요약 출력
    average_precision_score
)
# 설정 파일에서 경로, 랜덤 시드 불러오기
from aki_config import DATA_SPLIT_PATH, MODEL_PATH, RANDOM_STATE, THRESHOLD_PATH


def main():
    # 1. 데이터 분할 결과 불러오기
    X_train, X_valid, X_test, y_train, y_valid, y_test = joblib.load(DATA_SPLIT_PATH)

    # 2. baseline 모델 비교를 위해 train + valid 합치기
    X_train_full = pd.concat([X_train, X_valid], axis=0)
    y_train_full = pd.concat([y_train, y_valid], axis=0)

    models = {
        "Logistic": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "XGBoost": joblib.load(MODEL_PATH)
    }

    results = {}

    for name, model in models.items():
        if name != "XGBoost":
            model.fit(X_train_full, y_train_full)

        y_prob = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "AUROC": roc_auc_score(y_test, y_prob),
            "AUPRC": average_precision_score(y_test, y_prob)
        }

    print("=== Model Comparison ===")
    for name, score in results.items():
        print(name, score)

    plt.figure()
    plt.bar(results.keys(), [v["AUROC"] for v in results.values()])
    plt.title("Model Comparison (AUROC)")
    plt.ylabel("AUROC")
    plt.show()

    model = models["XGBoost"]
    best_threshold = 0.55


    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= best_threshold).astype(int)

    print("\n=== Final XGBoost Report ===")
    print(f"Threshold: {best_threshold:.2f}")
    print("AUROC:", roc_auc_score(y_test, y_prob))
    print("AUPRC:", average_precision_score(y_test, y_prob))
    print(classification_report(y_test, y_pred, target_names=["no_aki", "aki"], digits=3))

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.legend()
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()

    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    plt.figure()
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["no_aki", "aki"]
    )
    disp.plot()
    plt.title(f"Confusion Matrix (threshold={best_threshold:.2f})")
    plt.show()

    print("\n예측 확률 일부:")
    print(y_prob[:10])

    print("\n예측 결과 일부:")
    print(y_pred[:10])

if __name__ == "__main__":
    main()