import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import pickle

matplotlib.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus'] = False

os.makedirs("outputs", exist_ok=True)

# ==========================================================
# 데이터 로딩
# 변수 변환은 이상치 처리 후 스케일링 전 데이터 기준
# 변환 후 스케일링을 다시 적용해야 하므로
# outlier 처리 완료본을 사용
# ==========================================================

train_df = pd.read_csv("data/train_outlier_processed.csv")
valid_df = pd.read_csv("data/valid_outlier_processed.csv")
test_df  = pd.read_csv("data/test_outlier_processed.csv")

print("=" * 60)
print("데이터 로딩 완료")
print("=" * 60)
print(f"Train : {len(train_df):,}행 × {train_df.shape[1]}컬럼")
print(f"Valid : {len(valid_df):,}행")
print(f"Test  : {len(test_df):,}행")
print()

# ==========================================================
# 피처 컬럼 정의
# EDA 결과 반영:
#   urine_ml_kg_hr 제거 (urine_output_sum과 상관계수 1.0)
# ==========================================================

FEATURE_COLS = [
    # 혈역학
    'map_mean', 'map_min', 'map_below65_hours',
    'sbp_min', 'sbp_mean', 'shock_index_mean',
    # 활력징후
    'hr_max', 'hr_mean',
    'rr_max', 'rr_mean',
    'temp_max', 'temp_mean',
    # 소변량 / 신장
    'urine_output_sum', 'urine_output_6h',
    'oliguria_flag',                      # urine_ml_kg_hr 제거
    'creatinine_min', 'creatinine_max', 'creatinine_delta',
    'bun_max', 'bun_cr_ratio',
    # 저관류 / 쇼크
    'lactate_max', 'lactate_mean',
    'vasopressor_flag', 'vasopressor_hours', 'norepi_dose_max',
    # 전해질
    'potassium_max', 'potassium_mean',
    'bicarbonate_min', 'bicarbonate_mean',
    'sodium_min', 'sodium_max',
    # 산소 / 빈혈
    'hemoglobin_min', 'hemoglobin_mean',
    'spo2_min', 'spo2_mean',
]

print("=" * 60)
print("피처 컬럼 정의")
print("=" * 60)
print(f"최종 피처 수 : {len(FEATURE_COLS)}개")
print(f"제거된 피처  : urine_ml_kg_hr (urine_output_sum과 동일)")
print()

# 변환 대상 피처 정의 (EDA 결과 기반)
LOG1P_COLS = [
    'urine_output_6h',    # 왜도 8.675
    'vasopressor_hours',  # 왜도 4.751
    'lactate_max',        # 왜도 2.246
    'lactate_mean',       # 왜도 1.784
]

SQRT_COLS = [
    'creatinine_delta',   # 왜도 5.521 → sqrt 후 0.072
    'norepi_dose_max',    # 왜도 9.058 → sqrt 후 4.213
    'map_below65_hours',  # 왜도 2.064 → sqrt 후 0.736
]

# 이진 변수 (변환 제외)
BINARY_SKIP = [
    'vasopressor_flag',
    'oliguria_flag',
]

print("변환 대상 피처")
print(f"  log1p 변환 : {LOG1P_COLS}")
print(f"  sqrt  변환 : {SQRT_COLS}")
print(f"  변환 제외  : {BINARY_SKIP}")
print()

# ==========================================================
# 변수 변환 적용
# 중요: Train에서 확인한 변환 방법을
#       Valid / Test에도 동일하게 적용
# ==========================================================

print("=" * 60)
print("변수 변환 전 왜도 확인 (Train 기준)")
print("=" * 60)

all_transform_cols = LOG1P_COLS + SQRT_COLS
print(f"{'피처':<25} {'변환 전 왜도':>12} {'변환 방법':>10}")
print("-" * 52)

for col in all_transform_cols:
    skew_before = train_df[col].skew()
    method = 'log1p' if col in LOG1P_COLS else 'sqrt'
    print(f"  {col:<23} {skew_before:>12.3f} {method:>10}")
print()

# ==========================================================
# log1p 변환 적용
# log1p(x) = log(x + 1)
# 0인 값도 안전하게 처리 가능
# ==========================================================

print("=" * 60)
print("log1p 변환 적용")
print("=" * 60)

for col in LOG1P_COLS:
    for df in [train_df, valid_df, test_df]:
        df[col] = np.log1p(df[col])
    print(f"  {col} 변환 완료"
          f"  (변환 후 왜도: {train_df[col].skew():.3f})")

print()

# ==========================================================
# sqrt 변환 적용
# sqrt(x) = x의 제곱근
# 음수 불가 → clip(lower=0) 적용 후 변환
# ==========================================================

print("=" * 60)
print("sqrt 변환 적용")
print("=" * 60)

for col in SQRT_COLS:
    for df in [train_df, valid_df, test_df]:
        df[col] = np.sqrt(df[col].clip(lower=0))
    print(f"  {col} 변환 완료"
          f"  (변환 후 왜도: {train_df[col].skew():.3f})")

print()

# ==========================================================
# 변환 전후 왜도 비교
# ==========================================================

print("=" * 60)
print("변환 전후 왜도 비교 요약")
print("=" * 60)

# 원본 데이터 로딩 (변환 전 왜도 계산용)
train_origin = pd.read_csv("data/train_outlier_processed.csv")

print(f"{'피처':<25} {'변환 전':>10} {'변환 후':>10} {'개선':>8}")
print("-" * 58)

for col in all_transform_cols:
    before = train_origin[col].skew()
    after  = train_df[col].skew()
    diff   = abs(before) - abs(after)
    print(f"  {col:<23} {before:>10.3f} {after:>10.3f} "
          f"  {diff:>+6.3f}")
print()

# ==========================================================
# Step 5. 변환 전후 분포 시각화
# ==========================================================

print("=" * 60)
print("변환 전후 분포 시각화")
print("=" * 60)

fig, axes = plt.subplots(
    len(all_transform_cols), 2,
    figsize=(14, len(all_transform_cols) * 3)
)

for i, col in enumerate(all_transform_cols):

    # 변환 전 (원본)
    ax_before = axes[i, 0]
    before_data = train_origin[col].dropna()
    ax_before.hist(before_data, bins=50,
                   color='#e24b4a', alpha=0.7,
                   edgecolor='white')
    ax_before.set_title(
        f'{col}\n변환 전 (왜도: {before_data.skew():.2f})',
        fontsize=10)
    ax_before.set_ylabel('빈도')

    # 변환 후
    ax_after = axes[i, 1]
    after_data = train_df[col].dropna()
    method = 'log1p' if col in LOG1P_COLS else 'sqrt'
    ax_after.hist(after_data, bins=50,
                  color='#185FA5', alpha=0.7,
                  edgecolor='white')
    ax_after.set_title(
        f'{col} ({method} 변환 후)\n(왜도: {after_data.skew():.2f})',
        fontsize=10)
    ax_after.set_ylabel('빈도')

plt.suptitle('변수 변환 전후 분포 비교 (Train 기준)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('outputs/transform_before_after.png',
            dpi=150, bbox_inches='tight')
plt.show()

print("그래프 저장 완료: outputs/transform_before_after.png")
print()

# ==========================================================
# Step 6. 결측값 처리 + 스케일링 재적용
# 변환 후 분포가 달라졌으므로
# 스케일링을 Train 기준으로 다시 fit
# ==========================================================

from sklearn.preprocessing import RobustScaler
import pickle

print("=" * 60)
print("Step 6. 결측값 처리 + 스케일링 재적용")
print("=" * 60)

# 결측값 처리 (Train 중앙값 기준)
print("결측값 처리 중...")

medians = train_df[FEATURE_COLS].median()

train_df[FEATURE_COLS] = train_df[FEATURE_COLS].fillna(medians)
valid_df[FEATURE_COLS] = valid_df[FEATURE_COLS].fillna(medians)
test_df[FEATURE_COLS]  = test_df[FEATURE_COLS].fillna(medians)

remaining = train_df[FEATURE_COLS].isnull().sum().sum()
print(f"  결측값 처리 완료 (남은 결측: {remaining}개)")
print()

# RobustScaler 재적용 (Train only fit)
print("RobustScaler 재적용 중...")

scaler = RobustScaler()

X_train = scaler.fit_transform(train_df[FEATURE_COLS])
X_valid = scaler.transform(valid_df[FEATURE_COLS])
X_test  = scaler.transform(test_df[FEATURE_COLS])

y_train       = train_df['aki_label'].values
y_valid       = valid_df['aki_label'].values
y_test        = test_df['aki_label'].values

y_stage_train = train_df['aki_stage'].values
y_stage_valid = valid_df['aki_stage'].values
y_stage_test  = test_df['aki_stage'].values

print(f"  X_train shape : {X_train.shape}")
print(f"  X_valid shape : {X_valid.shape}")
print(f"  X_test  shape : {X_test.shape}")
print()

# 스케일링 확인 (중앙값 0 근처여야 정상)
print("스케일링 확인 (Train 기준 중앙값 → 0 근처)")
print("-" * 50)
X_train_df = pd.DataFrame(X_train, columns=FEATURE_COLS)
for col in ['map_mean', 'hr_mean', 'creatinine_max',
            'lactate_max', 'creatinine_delta']:
    med = X_train_df[col].median()
    print(f"  {col:<25} 중앙값: {med:.4f}")
print()

# ==========================================================
# Step 7. 최종 데이터 및 파이프라인 저장
# ==========================================================

print("=" * 60)
print("Step 7. 최종 저장")
print("=" * 60)

# 변환 정보 저장
transform_info = {
    'log1p_cols'   : LOG1P_COLS,
    'sqrt_cols'    : SQRT_COLS,
    'binary_skip'  : BINARY_SKIP,
    'removed_cols' : ['urine_ml_kg_hr'],
    'feature_cols' : FEATURE_COLS,
    'medians'      : medians.to_dict(),
    'scaler'       : scaler,
}

with open('outputs/transform_info.pkl', 'wb') as f:
    pickle.dump(transform_info, f)

print("transform_info.pkl 저장 완료")
print()

# numpy 배열로 저장
np.save('data/X_train.npy', X_train)
np.save('data/X_valid.npy', X_valid)
np.save('data/X_test.npy',  X_test)

np.save('data/y_train.npy', y_train)
np.save('data/y_valid.npy', y_valid)
np.save('data/y_test.npy',  y_test)

np.save('data/y_stage_train.npy', y_stage_train)
np.save('data/y_stage_valid.npy', y_stage_valid)
np.save('data/y_stage_test.npy',  y_stage_test)

print("numpy 배열 저장 완료")
print("  data/X_train.npy / X_valid.npy / X_test.npy")
print("  data/y_train.npy / y_valid.npy / y_test.npy")
print("  data/y_stage_train.npy / y_stage_valid.npy / y_stage_test.npy")
print()

# 최종 확인
print("=" * 60)
print("최종 확인")
print("=" * 60)
print(f"X_train : {X_train.shape}  y_train : {y_train.shape}")
print(f"X_valid : {X_valid.shape}  y_valid : {y_valid.shape}")
print(f"X_test  : {X_test.shape}  y_test  : {y_test.shape}")
print()
print(f"AKI 비율 Train : {y_train.mean():.4f}")
print(f"AKI 비율 Valid : {y_valid.mean():.4f}")
print(f"AKI 비율 Test  : {y_test.mean():.4f}")
print()

# 피처 목록 최종 출력
print("=" * 60)
print("최종 FEATURE_COLS (35개)")
print("=" * 60)
for i, col in enumerate(FEATURE_COLS, 1):
    marker = " ← 변환됨" if col in LOG1P_COLS + SQRT_COLS else ""
    print(f"  {i:2d}. {col}{marker}")

print()
print("변수 변환 완료 ✅")
print("다음 단계: 모델 학습 (07_modeling.py)")