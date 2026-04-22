import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score ,precision_score
from aki_config import MODEL_PATH, DATA_SPLIT_PATH, THRESHOLD_PATH

def main():
    model = joblib.load(MODEL_PATH)
    X_train, X_valid, X_test, y_train, y_valid, y_test = joblib.load(DATA_SPLIT_PATH)

    # validation 데이터에서 확률값 예측
    y_prob_valid = model.predict_proba(X_valid)[:, 1]
    # threshold 후보 (0.1 ~ 0.9)
    thresholds = np.linspace(0.1, 0.9, 81)

    records = []

    # threshold별 성능 계산
    for t in thresholds:
        y_pred = (y_prob_valid >= t).astype(int)

        records.append({
            "threshold": t,
            "precision": precision_score(y_valid, y_pred, zero_division=0),
            "recall": recall_score(y_valid, y_pred, zero_division=0),
            "f1": f1_score(y_valid, y_pred, zero_division=0)
        })

    # recall 0.8 이상 중에서 f1 최대 선택
    valid_candidates = [r for r in records if r["recall"] >= 0.80]
    if valid_candidates:
        best = max(valid_candidates, key=lambda x: x["f1"])
    else:
        best = max(records, key=lambda x: x["f1"])

    best_threshold = best["threshold"]
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