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
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,          # AUROC 계산
    confusion_matrix,       # 혼동행렬 계산
    ConfusionMatrixDisplay, # 혼동행렬 시각화
    classification_report,  # precision / recall / f1-score 요약 출력
    accuracy_score,
    f1_score
)
# 설정 파일에서 경로, 랜덤 시드 불러오기
from aki_config import DATA_SPLIT_PATH, MODEL_PATH, RANDOM_STATE, CLASS_NAMES


def main():
    # 1. 데이터 분할 결과 불러오기
    X_train, X_valid, X_test, y_train, y_valid, y_test = joblib.load(DATA_SPLIT_PATH)

    # 2. baseline 모델 비교를 위해 train + valid 합치기
    X_train_full = pd.concat([X_train, X_valid], axis=0)
    y_train_full = pd.concat([y_train, y_valid], axis=0)

    models = {
        "Logistic": LogisticRegression(
            max_iter=1000,
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

        y_prob = model.predict_proba(X_test)
        y_pred = model.predict(X_test)

        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Macro F1": f1_score(y_test, y_pred, average="macro"),
            "Macro AUROC": roc_auc_score(
                y_test,
                y_prob,
                multi_class="ovr",
                average="macro"
            )
        }

    print("=== Model Comparison ===")
    for name, score in results.items():
        print(name, score)

    # 모델별 AUROC 비교 그래프
    plt.figure()
    plt.bar(results.keys(), [v["Macro AUROC"] for v in results.values()])
    plt.title("Model Comparison (Macro AUROC)")
    plt.ylabel("Macro AUROC")
    plt.show()

    # 최종 XGBoost 상세 평가
    model = models["XGBoost"]
    y_prob = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    print("\n=== Final XGBoost Report ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Macro F1:", f1_score(y_test, y_pred, average="macro"))
    print(
        "Macro AUROC:",
        roc_auc_score(
            y_test,
            y_prob,
            multi_class="ovr",
            average="macro"
        )
    )

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=CLASS_NAMES,
            digits=3
        )
    )


    # Confusion Matrix
    # 실제 0/1과 예측 0/1이 어떻게 맞고 틀렸는지 보여줍니다.
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )
    disp.plot()
    plt.title("Confusion Matrix - AKI Stage Multiclass")
    plt.show()


if __name__ == "__main__":
    main()