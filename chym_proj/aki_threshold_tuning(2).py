# 학습된 모델의 예측 확률을 0/1 결과로 바꾸기 위한 threshold를 찾는 파일입니다.
#
# 왜 threshold가 필요한가?
# - XGBoost는 "AKI일 확률"을 출력합니다. 예: 0.73
# - 실제 경고 시스템은 확률을 0/1로 바꿔야 합니다.
# - 기본 threshold는 0.5이지만, 의료 문제에서는 놓치는 것(False Negative)을 줄이는 것이 중요할 수 있습니다.
# - 그래서 validation set에서 recall을 충분히 확보하면서 F1이 좋은 threshold를 찾습니다.
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score ,precision_score
from aki_config import MODEL_PATH, DATA_SPLIT_PATH, THRESHOLD_PATH

def main():
    model = joblib.load(MODEL_PATH)
    X_train, X_valid, X_test, y_train, y_valid, y_test = joblib.load(DATA_SPLIT_PATH)

    # validation set에서 AKI 발생 확률 예측
    # [:, 1]은 클래스 1, 즉 AKI 발생 확률을 의미합니다.
    y_prob_valid = model.predict_proba(X_valid)[:, 1]
    # threshold 후보를 0.1부터 0.9까지 촘촘하게 확인합니다.
    thresholds = np.linspace(0.1, 0.9, 81)

    records = []

    # threshold마다 precision, recall, f1을 계산합니다.
    for t in thresholds:
        y_pred = (y_prob_valid >= t).astype(int)

        records.append({
            "threshold": t,
            "precision": precision_score(y_valid, y_pred, zero_division=0),
            "recall": recall_score(y_valid, y_pred, zero_division=0),
            "f1": f1_score(y_valid, y_pred, zero_division=0)
        })

    # 의료 예측에서는 AKI 환자를 놓치지 않는 것이 중요하므로 recall 0.8 이상을 우선 조건으로 둡니다.
    # 그중에서 F1이 가장 높은 threshold를 선택합니다.
    valid_candidates = [r for r in records if r["recall"] >= 0.80]
    if valid_candidates:
        best = max(valid_candidates, key=lambda x: x["f1"])
    else:
        best = max(records, key=lambda x: x["f1"])

    best_threshold = best["threshold"]
    # 평가 파일에서 사용하도록 threshold 저장
    joblib.dump(best_threshold, THRESHOLD_PATH)

    print("Best Threshold:", best_threshold)
    print("Metrics at Best Threshold:", best)

    # 시각화
    plt.figure()
    plt.plot([r["threshold"] for r in records], [r["f1"] for r in records], label="F1")
    plt.plot([r["threshold"] for r in records], [r["recall"] for r in records], label="Recall")
    plt.plot([r["threshold"] for r in records], [r["precision"] for r in records], label="Precision")
    plt.axvline(best_threshold, linestyle="--", label=f"Best={best_threshold:.2f}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Tuning on Validation Set")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()