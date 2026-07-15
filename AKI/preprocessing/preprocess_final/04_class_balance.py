import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pickle
import os

matplotlib.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus'] = False

os.makedirs("outputs", exist_ok=True)

# ==========================================================
# 데이터 로딩
# 이상치 처리가 완료된 train/valid/test 사용
# ==========================================================

train_df = pd.read_csv(
    "../03_Data Preprocessing_scaling/train_final.csv"
)

valid_df = pd.read_csv(
    "../03_Data Preprocessing_scaling/valid_final.csv"
)

test_df = pd.read_csv(
    "../03_Data Preprocessing_scaling/test_final.csv"
)

print("=" * 60)
print("데이터 로딩 완료")
print("=" * 60)
print(f"Train : {len(train_df):,}행")
print(f"Valid : {len(valid_df):,}행")
print(f"Test  : {len(test_df):,}행")
print()

# ==========================================================
# 현재 클래스 불균형 현황 확인
# ==========================================================

print("=" * 60)
print("현재 클래스 분포 (Train 기준)")
print("=" * 60)

aki_count    = train_df['aki_label'].sum()
normal_count = len(train_df) - aki_count
total        = len(train_df)

print(f"정상 (0) : {normal_count:,}명  ({normal_count/total*100:.1f}%)")
print(f"AKI (1)  : {aki_count:,}명  ({aki_count/total*100:.1f}%)")
print(f"비율     : 1 : {normal_count/aki_count:.2f}")
print()

print("=" * 60)
print("Stage별 분포 (Train 기준)")
print("=" * 60)

stage_counts = train_df['aki_stage'].value_counts().sort_index()
for stage, count in stage_counts.items():
    print(f"  Stage {stage} : {count:,}명  ({count/total*100:.1f}%)")
print()

# ==========================================================
# 클래스 불균형 시각화
# ==========================================================

print("=" * 60)
print("클래스 불균형 시각화")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 왼쪽: AKI 이진 분포
ax1 = axes[0]
labels = ['정상 (0)', 'AKI (1)']
counts = [normal_count, aki_count]
colors = ['#4A90D9', '#E24B4A']
bars = ax1.bar(labels, counts, color=colors, width=0.5, edgecolor='white')
ax1.set_title('AKI 이진 분류 분포 (Train)', fontsize=13, fontweight='bold')
ax1.set_ylabel('환자 수')
for bar, count in zip(bars, counts):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 200,
             f'{count:,}명\n({count/total*100:.1f}%)',
             ha='center', fontsize=11, fontweight='bold')
ax1.set_ylim(0, max(counts) * 1.2)

# 오른쪽: Stage 분포
ax2 = axes[1]
stage_labels = [f'Stage {i}' for i in stage_counts.index]
stage_values = stage_counts.values
stage_colors = ['#4A90D9', '#5BAD6F', '#F5A623', '#E24B4A']
bars2 = ax2.bar(stage_labels, stage_values,
                color=stage_colors, width=0.5, edgecolor='white')
ax2.set_title('Stage별 분포 (Train)', fontsize=13, fontweight='bold')
ax2.set_ylabel('환자 수')
for bar, count in zip(bars2, stage_values):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 100,
             f'{count:,}명\n({count/total*100:.1f}%)',
             ha='center', fontsize=10, fontweight='bold')
ax2.set_ylim(0, max(stage_values) * 1.25)

plt.suptitle('클래스 불균형 현황', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/class_distribution.png',
            dpi=150, bbox_inches='tight')
plt.show()

print("그래프 저장 완료: outputs/class_distribution.png")
print()

# ==========================================================
# class_weight 계산 (1단계: 이진 분류용)
# ==========================================================

from sklearn.utils.class_weight import compute_class_weight

print("=" * 60)
print("1단계 이진 분류 class_weight 계산")
print("=" * 60)

classes_binary = np.array([0, 1])
y_train_binary = train_df['aki_label'].values

weights_binary = compute_class_weight(
    class_weight='balanced',
    classes=classes_binary,
    y=y_train_binary
)

class_weight_binary = {
    0: weights_binary[0],
    1: weights_binary[1]
}

scale_pos_weight = normal_count / aki_count

print(f"정상 (0) 가중치 : {weights_binary[0]:.4f}")
print(f"AKI (1) 가중치  : {weights_binary[1]:.4f}")
print(f"AKI/정상 비율   : {weights_binary[1]/weights_binary[0]:.2f}배")
print()
print(f"XGBoost scale_pos_weight : {scale_pos_weight:.4f}")
print()
print("의미:")
print(f"  AKI 환자를 틀렸을 때")
print(f"  정상 환자 오분류보다 {weights_binary[1]/weights_binary[0]:.1f}배 더 큰 패널티")
print()

# ==========================================================
# class_weight 계산 (2단계: Stage 다중 분류용)
# AKI 환자만 대상
# ==========================================================

print("=" * 60)
print("2단계 Stage 분류 class_weight 계산")
print("=" * 60)

# AKI 환자만 추출
train_aki = train_df[train_df['aki_label'] == 1]

classes_stage = np.array([1, 2, 3])
y_train_stage = train_aki['aki_stage'].values

weights_stage = compute_class_weight(
    class_weight='balanced',
    classes=classes_stage,
    y=y_train_stage
)

class_weight_stage = {
    1: weights_stage[0],
    2: weights_stage[1],
    3: weights_stage[2]
}

print(f"Stage 1 가중치 : {weights_stage[0]:.4f}")
print(f"Stage 2 가중치 : {weights_stage[1]:.4f}")
print(f"Stage 3 가중치 : {weights_stage[2]:.4f}")
print()
print("의미:")
print(f"  Stage 2를 틀리면 Stage 1 오분류보다")
print(f"  {weights_stage[1]/weights_stage[0]:.1f}배 더 큰 패널티")
print(f"  Stage 3를 틀리면 Stage 1 오분류보다")
print(f"  {weights_stage[2]/weights_stage[0]:.1f}배 더 큰 패널티")
print()

# ==========================================================
# Threshold Tuning 준비
# 모델 학습 후 Validation set에서 최적 임계값 탐색
# 지금은 탐색 함수만 정의해둠
# ==========================================================

print("=" * 60)
print("Threshold Tuning 함수 정의")
print("=" * 60)

from sklearn.metrics import (
    confusion_matrix, f1_score,
    roc_auc_score, average_precision_score
)

def find_best_threshold(y_true, y_proba,
                        min_sensitivity=0.75):
    """
    민감도 0.75 이상을 만족하면서
    특이도가 가장 높은 임계값 탐색

    Parameters:
        y_true        : 실제 레이블
        y_proba       : 모델 예측 확률
        min_sensitivity: 최소 민감도 기준 (기본 0.75)

    Returns:
        best_threshold: 최적 임계값
        best_result   : 해당 임계값의 성능 지표
    """
    thresholds = np.arange(0.05, 0.95, 0.05)
    results = []

    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_true, pred, labels=[0, 1]
        ).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1          = f1_score(y_true, pred, zero_division=0)

        results.append({
            'threshold'  : round(t, 2),
            'sensitivity': round(sensitivity, 4),
            'specificity': round(specificity, 4),
            'f1'         : round(f1, 4),
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
        })

    results_df = pd.DataFrame(results)

    # 민감도 조건 충족하면서 특이도 최대
    valid = results_df[
        results_df['sensitivity'] >= min_sensitivity
    ]

    if len(valid) == 0:
        print(f"  민감도 {min_sensitivity} 이상 임계값 없음")
        print(f"  민감도가 가장 높은 임계값 선택")
        best_idx = results_df['sensitivity'].idxmax()
    else:
        best_idx = valid['specificity'].idxmax()

    best_result    = results_df.loc[best_idx]
    best_threshold = best_result['threshold']

    return best_threshold, best_result, results_df


print("find_best_threshold 함수 정의 완료")
print()
print("사용 방법:")
print("  모델 학습 후 아래와 같이 호출")
print("  best_t, best_result, all_results =")
print("    find_best_threshold(y_valid, y_pred_proba)")
print()

# ==========================================================
# Step 6. 클래스 불균형 처리 결과 저장
# ==========================================================

print("=" * 60)
print("클래스 불균형 처리 결과 저장")
print("=" * 60)

class_balance_info = {

    # 1단계 이진 분류
    'binary': {
        'class_weight'     : class_weight_binary,
        'scale_pos_weight' : scale_pos_weight,
        'normal_count'     : int(normal_count),
        'aki_count'        : int(aki_count),
        'aki_ratio'        : round(aki_count / total, 4),
    },

    # 2단계 Stage 분류
    'stage': {
        'class_weight': class_weight_stage,
        'stage_counts': {
            int(k): int(v)
            for k, v in stage_counts.items()
        },
    },

    # Threshold 기준
    'threshold': {
        'min_sensitivity': 0.75,
        'default'        : 0.5,
        'note'           : '모델 학습 후 Validation에서 탐색'
    }
}

with open('outputs/class_balance_info.pkl', 'wb') as f:
    pickle.dump(class_balance_info, f)

print("class_balance_info.pkl 저장 완료")
print()

# 요약 출력
print("=" * 60)
print("최종 요약 — 모델 학습 시 사용할 값")
print("=" * 60)
print()
print("[ 1단계 XGBoost 이진 분류 ]")
print(f"  scale_pos_weight = {scale_pos_weight:.4f}")
print(f"  class_weight     = {class_weight_binary}")
print()
print("[ 2단계 XGBoost Stage 분류 ]")
print(f"  class_weight = {class_weight_stage}")
print()
print("[ Threshold Tuning ]")
print(f"  모델 학습 후 Validation set에서 탐색")
print(f"  목표: 민감도 0.75 이상 + 특이도 최대")
print()
print("클래스 불균형 처리 완료 ✅")