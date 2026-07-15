import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import pickle

matplotlib.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus'] = False


# ==========================================================
# 데이터 불러오기
# ==========================================================

train_df = pd.read_csv("../data/train.csv")
valid_df = pd.read_csv("../data/valid.csv")
test_df  = pd.read_csv("../data/test.csv")

print("=" * 60)
print("데이터 로딩 완료")
print("=" * 60)
print(f"Train : {len(train_df):,}행")
print(f"Valid : {len(valid_df):,}행")
print(f"Test  : {len(test_df):,}행")
print()

# outputs 폴더 없으면 생성
os.makedirs("outputs", exist_ok=True)


# ==========================================================
# 피처 컬럼 정의
# ==========================================================

# 모델에 사용할 피처 목록
# aki_label, aki_stage는 정답(Y)이므로 제외
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
    'urine_ml_kg_hr', 'oliguria_flag',
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
print("피처 컬럼 정의 완료")
print("=" * 60)
print(f"총 피처 수 : {len(FEATURE_COLS)}개")
print()

# 실제 데이터 컬럼과 일치 확인
meta_cols = ['stay_id', 'subject_id', 'hadm_id',
             'age', 'gender', 'aki_label', 'aki_stage',
             'aki_onset_time', 'prediction_cutoff', 'index_time']

actual_feature_cols = [c for c in train_df.columns
                       if c not in meta_cols]

missing = set(FEATURE_COLS) - set(actual_feature_cols)
extra   = set(actual_feature_cols) - set(FEATURE_COLS)

print(f"실제 데이터 피처 수 : {len(actual_feature_cols)}개")
print(f"정의했지만 없는 피처 : {missing}")
print(f"데이터에 있지만 미정의 : {extra}")

if len(missing) == 0 and len(extra) == 0:
    print("✅ 피처 목록 일치")
else:
    print("❌ 피처 목록 불일치 확인 필요")
print()


# ==========================================================
# 이상치 처리 전 현황 파악
# ==========================================================

print("=" * 60)
print("이상치 처리 전 기초 통계 (Train 기준)")
print("=" * 60)

before_stats = train_df[FEATURE_COLS].describe().T[[
    'min', 'max', 'mean', '50%'
]]
before_stats.columns = ['최솟값', '최댓값', '평균', '중앙값']

print(before_stats.to_string())
print()

# 현재 결측값 현황
print("=" * 60)
print("현재 결측값 현황 (Train 기준)")
print("=" * 60)

null_counts = train_df[FEATURE_COLS].isnull().sum()
null_pct    = (null_counts / len(train_df) * 100).round(2)

null_df = pd.DataFrame({
    '결측 건수': null_counts,
    '결측률(%)': null_pct
})

null_df = null_df[null_df['결측 건수'] > 0].sort_values(
    '결측률(%)', ascending=False
)

if len(null_df) > 0:
    print(null_df.to_string())
else:
    print("결측값 없음")
print()



# ==========================================================
# Step 1. 확인된 이상치 직접 처리
# ==========================================================

print("=" * 60)
print("Step 1. 확인된 이상치 직접 처리")
print("=" * 60)

# 처리 결과 기록용
outlier_report = {}

def apply_to_all(col, mask_fn, replace_val):
    """
    Train / Valid / Test 세 셋에 동일하게 적용
    mask_fn : 이상치 조건 함수
    replace_val : None 이면 NULL, 숫자면 해당 값으로 캡
    """
    for name, df in [("Train", train_df),
                     ("Valid", valid_df),
                     ("Test",  test_df)]:
        mask = mask_fn(df[col])
        count = mask.sum()
        if replace_val is None:
            df.loc[mask, col] = np.nan
        else:
            df.loc[mask, col] = replace_val
        if name == "Train":
            outlier_report[col] = {
                "처리 건수": int(count),
                "처리 방법": "NULL" if replace_val is None
                             else f"{replace_val}로 캡"
            }
            print(f"  {col:30s} Train {count:5d}건 처리")


# ── creatinine = 0 → NULL ──────────────────────────────────
apply_to_all('creatinine_min',   lambda x: x <= 0,    None)
apply_to_all('creatinine_max',   lambda x: x <= 0,    None)

# creatinine_delta: 음수만 NULL (0은 변화없음으로 정상)
apply_to_all('creatinine_delta', lambda x: x < 0,     None)

# ── bun_cr_ratio > 100 → NULL ─────────────────────────────
apply_to_all('bun_cr_ratio',     lambda x: x > 100,   None)

# ── vasopressor_hours > 48 → 48 캡 ───────────────────────
apply_to_all('vasopressor_hours',lambda x: x > 48,    48.0)

# ── potassium 임상 범위 ────────────────────────────────────
apply_to_all('potassium_max',    lambda x: x > 9.0,   None)
apply_to_all('potassium_mean',   lambda x: x > 9.0,   None)

# ── sodium 임상 범위 ──────────────────────────────────────
apply_to_all('sodium_min',       lambda x: x < 100,   None)
apply_to_all('sodium_max',       lambda x: x > 180,   None)

print()


# ==========================================================
# Step 2. IQR 기반 이상치 탐지 및 처리
# ==========================================================

print("=" * 60)
print("Step 2. IQR 기반 이상치 탐지 (Train 기준)")
print("=" * 60)

# IQR 적용할 피처 (이진/플래그 변수 제외)
IQR_COLS = [
    'map_mean', 'map_min', 'map_below65_hours',
    'sbp_min', 'sbp_mean', 'shock_index_mean',
    'hr_max', 'hr_mean',
    'rr_max', 'rr_mean',
    'temp_max', 'temp_mean',
    'urine_output_sum', 'urine_output_6h',
    'urine_ml_kg_hr',
    'creatinine_min', 'creatinine_max', 'creatinine_delta',
    'bun_max', 'bun_cr_ratio',
    'lactate_max', 'lactate_mean',
    'vasopressor_hours', 'norepi_dose_max',
    'potassium_max', 'potassium_mean',
    'bicarbonate_min', 'bicarbonate_mean',
    'sodium_min', 'sodium_max',
    'hemoglobin_min', 'hemoglobin_mean',
    'spo2_min', 'spo2_mean',
]

# 이진/플래그 변수는 IQR 적용 제외
# oliguria_flag, vasopressor_flag → 0 or 1 이므로 제외

# Train 기준으로 IQR 계산 후 저장
iqr_bounds = {}

print(f"{'피처':<30} {'하한':>10} {'상한':>10} {'이상치 건수':>12}")
print("-" * 65)

for col in IQR_COLS:
    q1  = train_df[col].quantile(0.25)
    q3  = train_df[col].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    iqr_bounds[col] = {'lower': lower, 'upper': upper}

    # Train 이상치 건수만 확인 (아직 처리 안 함)
    outlier_mask = (
        (train_df[col] < lower) |
        (train_df[col] > upper)
    )
    count = outlier_mask.sum()

    if count > 0:
        print(f"{col:<30} {lower:>10.3f} {upper:>10.3f} {count:>12,}")

print()
print(f"IQR 경계 계산 완료: {len(iqr_bounds)}개 피처")
print()

# ==========================================================
# Step 3. IQR 기반 이상치 처리
# 하한이 음수인 경우 임상적으로 0으로 보정
# ==========================================================

print("=" * 60)
print("Step 3. IQR 기반 이상치 처리")
print("=" * 60)

# 0 미만이 불가능한 피처 (하한을 0으로 보정)
NON_NEGATIVE_COLS = [
    'map_below65_hours', 'urine_output_sum', 'urine_output_6h',
    'urine_ml_kg_hr', 'creatinine_min', 'creatinine_max',
    'creatinine_delta', 'bun_max', 'bun_cr_ratio',
    'lactate_max', 'lactate_mean', 'vasopressor_hours',
    'norepi_dose_max', 'potassium_max', 'potassium_mean',
    'bicarbonate_min', 'bicarbonate_mean',
    'hemoglobin_min', 'hemoglobin_mean',
]

iqr_outlier_report = {}

for col in IQR_COLS:

    lower = iqr_bounds[col]['lower']
    upper = iqr_bounds[col]['upper']

    # 0 미만 불가능한 피처는 하한을 0으로 보정
    if col in NON_NEGATIVE_COLS:
        lower = max(lower, 0)
        iqr_bounds[col]['lower'] = lower

    train_before = train_df[col].isnull().sum()

    for df in [train_df, valid_df, test_df]:
        df.loc[df[col] < lower, col] = np.nan
        df.loc[df[col] > upper, col] = np.nan

    train_after  = train_df[col].isnull().sum()
    added_nulls  = train_after - train_before

    iqr_outlier_report[col] = {
        'lower': lower,
        'upper': upper,
        '추가 NULL 건수': int(added_nulls)
    }

    if added_nulls > 0:
        print(f"  {col:<30} [{lower:>8.3f} ~ {upper:>8.3f}]"
              f"  +{added_nulls:,}건 NULL")

print()
print("IQR 기반 이상치 처리 완료")
print()



# ==========================================================
# Step 3 보정
# urine_output_6h, vasopressor_hours는
# IQR 적용 부적합 → 원래 train.csv 값으로 복원
# ==========================================================

print("=" * 60)
print("Step 3 보정: IQR 부적합 피처 복원")
print("=" * 60)

# 원본 데이터 다시 로딩
train_origin = pd.read_csv("../data/train.csv")
valid_origin = pd.read_csv("../data/valid.csv")
test_origin  = pd.read_csv("../data/test.csv")

# urine_output_6h 복원
for col in ['urine_output_6h', 'vasopressor_hours']:
    train_df[col] = train_origin[col]
    valid_df[col] = valid_origin[col]
    test_df[col]  = test_origin[col]

    # vasopressor_hours는 Step 1에서 48 캡 다시 적용
    if col == 'vasopressor_hours':
        for df in [train_df, valid_df, test_df]:
            df.loc[df[col] > 48, col] = 48.0

    print(f"  {col} 복원 완료")

# IQR 경계에서도 해당 피처 제외
for col in ['urine_output_6h', 'vasopressor_hours']:
    if col in iqr_bounds:
        del iqr_bounds[col]

print()
print("복원 완료")
print()

# 복원 후 확인
for col in ['urine_output_6h', 'vasopressor_hours']:
    non_zero = (train_df[col] > 0).sum()
    print(f"  {col} 0 초과 건수: {non_zero:,}건")
print()

# ==========================================================
# creatinine_delta IQR 처리 취소
# IQR 상한 0.25는 임상적으로 너무 공격적
# KDIGO 기준 +0.3 이상이 AKI 판정 기준이므로 보존 필요
# ==========================================================

train_df['creatinine_delta'] = train_origin['creatinine_delta'].copy()
valid_df['creatinine_delta'] = valid_origin['creatinine_delta'].copy()
test_df['creatinine_delta']  = test_origin['creatinine_delta'].copy()

# 음수만 NULL (Step 1 재적용)
for df in [train_df, valid_df, test_df]:
    df.loc[df['creatinine_delta'] < 0, 'creatinine_delta'] = np.nan

print("creatinine_delta 복원 완료")
print(f"  최댓값: {train_df['creatinine_delta'].max():.3f}")
print()

# ==========================================================
# Step 4. 이상치 처리 후 결과 확인
# ==========================================================

print("=" * 60)
print("Step 4. 이상치 처리 후 기초 통계 (Train 기준)")
print("=" * 60)

after_stats = train_df[FEATURE_COLS].describe().T[[
    'min', 'max', 'mean', '50%'
]]
after_stats.columns = ['최솟값', '최댓값', '평균', '중앙값']

print(after_stats.to_string())
print()

# 처리 후 결측값 현황
print("=" * 60)
print("이상치 처리 후 결측값 현황 (Train 기준)")
print("=" * 60)

null_after = train_df[FEATURE_COLS].isnull().sum()
null_after_pct = (null_after / len(train_df) * 100).round(2)

null_after_df = pd.DataFrame({
    '결측 건수': null_after,
    '결측률(%)': null_after_pct
}).sort_values('결측률(%)', ascending=False)

print(null_after_df.to_string())
print()

# ==========================================================
# Step 5. 처리 전/후 분포 비교 시각화 (최종)
# ==========================================================

print("=" * 60)
print("Step 5. 처리 전/후 분포 비교 그래프 생성")
print("=" * 60)

# 원본 데이터 다시 로딩 (처리 전 기준)
train_origin = pd.read_csv("../data/train.csv")

VIZ_COLS = [
    'creatinine_max', 'creatinine_delta',
    'bun_max', 'bun_cr_ratio',
    'lactate_max', 'map_min',
    'vasopressor_hours', 'potassium_max',
    'urine_output_sum', 'sodium_min',
    'spo2_min', 'bicarbonate_min',
]

fig, axes = plt.subplots(
    len(VIZ_COLS), 2,
    figsize=(14, len(VIZ_COLS) * 3)
)

for i, col in enumerate(VIZ_COLS):

    # 처리 전
    ax_before = axes[i, 0]
    before_data = train_origin[col].dropna()
    ax_before.hist(before_data, bins=50,
                   color='#e24b4a', alpha=0.7, edgecolor='white')
    ax_before.set_title(f'{col} - 처리 전', fontsize=11)
    ax_before.set_xlabel('값')
    ax_before.set_ylabel('빈도')

    # 처리 후
    ax_after = axes[i, 1]
    after_data = train_df[col].dropna()
    ax_after.hist(after_data, bins=50,
                  color='#185FA5', alpha=0.7, edgecolor='white')
    ax_after.set_title(f'{col} - 처리 후', fontsize=11)
    ax_after.set_xlabel('값')
    ax_after.set_ylabel('빈도')

plt.suptitle('이상치 처리 전/후 분포 비교 (Train 기준)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('outputs/outlier_before_after_final.png',
            dpi=150, bbox_inches='tight')
plt.show()

print("그래프 저장 완료: outputs/outlier_before_after_final.png")
print()

# creatinine_delta 현재 상태 직접 확인
print("train_df creatinine_delta 현재 상태:")
print(f"  최솟값: {train_df['creatinine_delta'].min():.4f}")
print(f"  최댓값: {train_df['creatinine_delta'].max():.4f}")
print(f"  결측수: {train_df['creatinine_delta'].isnull().sum()}")
print()
print("train_origin creatinine_delta 상태:")
print(f"  최솟값: {train_origin['creatinine_delta'].min():.4f}")
print(f"  최댓값: {train_origin['creatinine_delta'].max():.4f}")
print(f"  결측수: {train_origin['creatinine_delta'].isnull().sum()}")

# ==========================================================
# Step 6. 이상치 처리 결과 리포트 저장
# 팀원 B에게 전달용
# ==========================================================

print("=" * 60)
print("Step 6. 이상치 처리 결과 리포트 저장")
print("=" * 60)

# 이상치 기준 저장 (pickle)
outlier_info = {
    'step1_fixed': {
        'creatinine_min':   '0 이하 → NULL',
        'creatinine_max':   '0 이하 → NULL',
        'creatinine_delta': '0 미만 → NULL',
        'bun_cr_ratio':     '100 초과 → NULL',
        'vasopressor_hours':'48 초과 → 48 캡',
        'potassium_max':    '9.0 초과 → NULL',
        'sodium_min':       '100 미만 → NULL',
    },
    'step2_iqr_bounds': iqr_bounds,
    'iqr_excluded': [
        'urine_output_6h',
        'vasopressor_hours',
        'creatinine_delta',
        'oliguria_flag',
        'vasopressor_flag',
    ]
}

with open('outputs/outlier_info.pkl', 'wb') as f:
    pickle.dump(outlier_info, f)

print("outlier_info.pkl 저장 완료")
print()

# 텍스트 리포트 저장
report_lines = []
report_lines.append("=" * 60)
report_lines.append("이상치 처리 리포트 (팀원 B 전달용)")
report_lines.append("=" * 60)
report_lines.append("")
report_lines.append("[Step 1] 직접 처리 내역")
for col, desc in outlier_info['step1_fixed'].items():
    report_lines.append(f"  {col:<30} {desc}")

report_lines.append("")
report_lines.append("[Step 2] IQR 기반 처리 내역")
for col, bounds in iqr_bounds.items():
    report_lines.append(
        f"  {col:<30} "
        f"[{bounds['lower']:>8.3f} ~ {bounds['upper']:>8.3f}]"
    )

report_lines.append("")
report_lines.append("[IQR 미적용 피처]")
for col in outlier_info['iqr_excluded']:
    report_lines.append(f"  {col}")

report_lines.append("")
report_lines.append("[처리 후 결측률 현황 - 팀원 B 결측처리 기준]")
null_final = train_df[FEATURE_COLS].isnull().mean() * 100
for col, pct in null_final.sort_values(ascending=False).items():
    report_lines.append(f"  {col:<30} {pct:.2f}%")

report_text = "\n".join(report_lines)

with open('outputs/outlier_report.txt', 'w',
          encoding='utf-8') as f:
    f.write(report_text)

print("outlier_report.txt 저장 완료")
print()
print(report_text)

# csv파일 저장(합칠 때 편하기 위한 작업)
train_df.to_csv(
    "train_outlier_processed.csv",
    index=False
)

valid_df.to_csv(
    "valid_outlier_processed.csv",
    index=False
)

test_df.to_csv(
    "test_outlier_processed.csv",
    index=False
)