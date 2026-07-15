import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')
matplotlib.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus'] = False

os.makedirs("outputs", exist_ok=True)

# ==========================================================
# 데이터 로딩
# 스케일링 전 데이터: 원래 값 범위로 해석 필요한 경우
# 스케일링 후 데이터: 모델 입력과 동일한 상태
# ==========================================================

# 스케일링 전 (원래 값 범위 해석용)
train_raw = pd.read_csv("data/train_missing_processed.csv")

# 스케일링 후 (모델 입력과 동일)
train_df = pd.read_csv("data/train_final.csv")
valid_df = pd.read_csv("data/valid_final.csv")
test_df  = pd.read_csv("data/test_final.csv")

print("=" * 60)
print("데이터 로딩 완료")
print("=" * 60)
print(f"Train (스케일링 전) : {len(train_raw):,}행 × {train_raw.shape[1]}컬럼")
print(f"Train (스케일링 후) : {len(train_df):,}행 × {train_df.shape[1]}컬럼")
print(f"Valid               : {len(valid_df):,}행")
print(f"Test                : {len(test_df):,}행")
print()

# ==========================================================
# 피처 컬럼 정의
# ==========================================================

META_COLS = [
    'stay_id', 'subject_id', 'hadm_id',
    'age', 'gender', 'aki_label', 'aki_stage',
    'aki_onset_time', 'prediction_cutoff', 'index_time'
]

FEATURE_COLS = [
    col for col in train_raw.columns
    if col not in META_COLS
    and not col.endswith('_missing')
]

print("=" * 60)
print("피처 컬럼 정의")
print("=" * 60)
print(f"피처 수 : {len(FEATURE_COLS)}개")
print()

# ==========================================================
# EDA 1. 기초 통계 및 분포
# ==========================================================

print("=" * 60)
print("EDA 1. 기초 통계 (스케일링 전 원본값 기준)")
print("=" * 60)

stats = train_raw[FEATURE_COLS].describe().T[[
    'min', 'max', 'mean', '50%', 'std'
]]
stats.columns = ['최솟값', '최댓값', '평균', '중앙값', '표준편차']

# 왜도 추가
stats['왜도'] = train_raw[FEATURE_COLS].skew().round(3)

print(stats.to_string())
print()

# 왜도 기준으로 분류
print("=" * 60)
print("왜도 기준 분류")
print("=" * 60)

skew_high   = stats[stats['왜도'].abs() >= 2].index.tolist()
skew_medium = stats[
    (stats['왜도'].abs() >= 1) &
    (stats['왜도'].abs() < 2)
].index.tolist()
skew_normal = stats[stats['왜도'].abs() < 1].index.tolist()

print(f"변환 강력 권장 (왜도 절댓값 ≥ 2) : {len(skew_high)}개")
for col in skew_high:
    print(f"  {col:<30} 왜도: {stats.loc[col,'왜도']:.3f}")

print()
print(f"변환 검토 (1 ≤ 왜도 절댓값 < 2) : {len(skew_medium)}개")
for col in skew_medium:
    print(f"  {col:<30} 왜도: {stats.loc[col,'왜도']:.3f}")

print()
print(f"정상 (왜도 절댓값 < 1)           : {len(skew_normal)}개")
for col in skew_normal:
    print(f"  {col:<30} 왜도: {stats.loc[col,'왜도']:.3f}")
print()

# ==========================================================
# EDA 1-2. 왜도 높은 피처 분포 시각화
# ==========================================================

print("=" * 60)
print("EDA 1-2. 왜도 높은 피처 분포 시각화")
print("=" * 60)

# 이진 변수 제외한 실제 변환 대상
TRANSFORM_CANDIDATES = [
    'urine_output_6h',
    'norepi_dose_max',
    'creatinine_delta',
    'vasopressor_hours',
    'lactate_max',
    'map_below65_hours',
    'lactate_mean',
]

fig, axes = plt.subplots(
    len(TRANSFORM_CANDIDATES), 3,
    figsize=(18, len(TRANSFORM_CANDIDATES) * 3)
)

for i, col in enumerate(TRANSFORM_CANDIDATES):

    data = train_raw[col].dropna()

    # 원본 분포
    ax1 = axes[i, 0]
    ax1.hist(data, bins=50,
             color='#e24b4a', alpha=0.7, edgecolor='white')
    ax1.set_title(f'{col}\n원본 (왜도: {data.skew():.2f})',
                  fontsize=10)
    ax1.set_ylabel('빈도')

    # log1p 변환 후
    ax2 = axes[i, 1]
    data_log = np.log1p(data)
    ax2.hist(data_log, bins=50,
             color='#185FA5', alpha=0.7, edgecolor='white')
    ax2.set_title(f'log1p 변환 후\n(왜도: {data_log.skew():.2f})',
                  fontsize=10)

    # sqrt 변환 후
    ax3 = axes[i, 2]
    data_sqrt = np.sqrt(data.clip(lower=0))
    ax3.hist(data_sqrt, bins=50,
             color='#5BAD6F', alpha=0.7, edgecolor='white')
    ax3.set_title(f'sqrt 변환 후\n(왜도: {data_sqrt.skew():.2f})',
                  fontsize=10)

plt.suptitle('왜도 높은 피처 변환 전후 비교',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('outputs/eda_skewness_transform.png',
            dpi=150, bbox_inches='tight')
plt.show()

print("그래프 저장 완료: outputs/eda_skewness_transform.png")
print()

# 변환 후 왜도 비교 표
print("=" * 60)
print("변환 전후 왜도 비교")
print("=" * 60)
print(f"{'피처':<25} {'원본':>8} {'log1p':>8} {'sqrt':>8} {'권장'}")
print("-" * 65)

for col in TRANSFORM_CANDIDATES:
    data      = train_raw[col].dropna()
    orig      = round(data.skew(), 3)
    log_skew  = round(np.log1p(data).skew(), 3)
    sqrt_skew = round(np.sqrt(data.clip(lower=0)).skew(), 3)

    # 절댓값 기준 가장 작은 변환 권장
    best = min(
        [('원본', abs(orig)),
         ('log1p', abs(log_skew)),
         ('sqrt', abs(sqrt_skew))],
        key=lambda x: x[1]
    )[0]

    print(f"{col:<25} {orig:>8} {log_skew:>8} {sqrt_skew:>8}  → {best}")

print()

# ==========================================================
# EDA 2. AKI vs 비AKI 분포 비교
# ==========================================================

print("=" * 60)
print("EDA 2. AKI vs 비AKI 분포 비교")
print("=" * 60)

aki_df    = train_raw[train_raw['aki_label'] == 1]
normal_df = train_raw[train_raw['aki_label'] == 0]

print(f"AKI 환자    : {len(aki_df):,}명")
print(f"정상 환자   : {len(normal_df):,}명")
print()

# 핵심 피처 비교
KEY_COLS = [
    'creatinine_max', 'creatinine_delta',
    'bun_max', 'bun_cr_ratio',
    'map_min', 'map_below65_hours',
    'lactate_max', 'urine_output_sum',
    'vasopressor_hours', 'spo2_min',
    'bicarbonate_min', 'shock_index_mean',
]

fig, axes = plt.subplots(
    len(KEY_COLS) // 2, 4,
    figsize=(20, len(KEY_COLS) // 2 * 4)
)
axes = axes.flatten()

for i, col in enumerate(KEY_COLS):
    ax = axes[i * 2]
    ax2 = axes[i * 2 + 1]

    # 히스토그램 비교
    normal_data = normal_df[col].dropna()
    aki_data    = aki_df[col].dropna()

    ax.hist(normal_data, bins=40, alpha=0.5,
            color='#4A90D9', label='정상', density=True)
    ax.hist(aki_data, bins=40, alpha=0.5,
            color='#E24B4A', label='AKI', density=True)
    ax.set_title(f'{col}', fontsize=10, fontweight='bold')
    ax.set_ylabel('밀도')
    ax.legend(fontsize=8)

    # 박스플롯 비교
    ax2.boxplot(
        [normal_data, aki_data],
        labels=['정상', 'AKI'],
        patch_artist=True,
        boxprops=dict(facecolor='#4A90D9', alpha=0.5),
        medianprops=dict(color='black', linewidth=2)
    )
    ax2.set_title(f'{col} 박스플롯', fontsize=10)
    colors_box = ['#4A90D9', '#E24B4A']
    for patch, color in zip(
        ax2.findobj(plt.matplotlib.patches.PathPatch),
        colors_box
    ):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

plt.suptitle('AKI vs 비AKI 피처 분포 비교 (Train 기준)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('outputs/eda_aki_vs_normal.png',
            dpi=150, bbox_inches='tight')
plt.show()

print("그래프 저장 완료: outputs/eda_aki_vs_normal.png")
print()

# 통계적 차이 확인 (Mann-Whitney U test)
from scipy import stats

print("=" * 60)
print("AKI vs 비AKI 통계적 차이 (Mann-Whitney U test)")
print("=" * 60)
print(f"{'피처':<25} {'AKI 중앙값':>12} {'정상 중앙값':>12} {'p-value':>12} {'유의'}")
print("-" * 75)

for col in FEATURE_COLS:
    aki_data    = aki_df[col].dropna()
    normal_data = normal_df[col].dropna()

    if len(aki_data) < 10 or len(normal_data) < 10:
        continue

    stat, p = stats.mannwhitneyu(
        aki_data, normal_data,
        alternative='two-sided'
    )

    sig = "✓" if p < 0.05 else " "
    print(f"{col:<25} {aki_data.median():>12.3f} "
          f"{normal_data.median():>12.3f} "
          f"{p:>12.4f}  {sig}")

print()

# ==========================================================
# EDA 3. 상관관계 분석
# ==========================================================

print("=" * 60)
print("EDA 3. 상관관계 분석")
print("=" * 60)

# 3-1. 피처 vs 레이블 상관관계
corr_with_label = train_raw[FEATURE_COLS + ['aki_label']]\
    .corr()['aki_label']\
    .drop('aki_label')\
    .abs()\
    .sort_values(ascending=False)

print("피처 vs aki_label 상관관계 (절댓값 기준)")
print("-" * 50)
for col, val in corr_with_label.items():
    flag = ""
    if val >= 0.9:
        flag = "  ⚠️ Leakage 강력 의심"
    elif val >= 0.5:
        flag = "  ⚠️ 요주의"
    print(f"  {col:<30} {val:.4f}{flag}")
print()

# 최고 상관관계 확인
max_corr = corr_with_label.max()
max_col  = corr_with_label.idxmax()
print(f"레이블과 가장 높은 상관관계: {max_col} ({max_corr:.4f})")
if max_corr >= 0.9:
    print("❌ Data Leakage 의심 → 즉시 확인 필요")
elif max_corr >= 0.5:
    print("⚠️ 요주의 피처 존재 → 확인 필요")
else:
    print("✅ 비정상적 상관관계 없음")
print()

# 3-2. 피처 간 상관관계 히트맵
print("피처 간 상관관계 히트맵 생성 중...")

corr_matrix = train_raw[FEATURE_COLS].corr()

fig, ax = plt.subplots(figsize=(18, 16))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=False,
    cmap='RdBu_r',
    vmin=-1, vmax=1,
    center=0,
    square=True,
    linewidths=0.5,
    ax=ax,
    cbar_kws={"shrink": 0.8}
)

ax.set_title('피처 간 상관관계 히트맵 (Train 기준)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/eda_correlation_heatmap.png',
            dpi=150, bbox_inches='tight')
plt.show()

print("히트맵 저장 완료: outputs/eda_correlation_heatmap.png")
print()

# 3-3. 높은 상관관계 피처 쌍 출력
print("=" * 60)
print("피처 간 높은 상관관계 (절댓값 0.7 이상)")
print("=" * 60)

high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        val = corr_matrix.iloc[i, j]
        if abs(val) >= 0.7:
            high_corr_pairs.append({
                '피처1': corr_matrix.columns[i],
                '피처2': corr_matrix.columns[j],
                '상관계수': round(val, 4)
            })

if high_corr_pairs:
    high_corr_df = pd.DataFrame(high_corr_pairs)\
        .sort_values('상관계수', key=abs, ascending=False)
    print(high_corr_df.to_string(index=False))
    print()
    print(f"총 {len(high_corr_pairs)}쌍 발견")
else:
    print("높은 상관관계 피처 쌍 없음")
print()

# ==========================================================
# EDA 4. Mutual Information + VIF
# ==========================================================

print("=" * 60)
print("EDA 4-1. Mutual Information")
print("=" * 60)

from sklearn.feature_selection import mutual_info_classif

X_mi = train_raw[FEATURE_COLS].fillna(
    train_raw[FEATURE_COLS].median()
)
y_mi = train_raw['aki_label'].values

mi_scores = mutual_info_classif(
    X_mi, y_mi,
    random_state=42
)

mi_df = pd.DataFrame({
    '피처': FEATURE_COLS,
    'MI 점수': mi_scores.round(4)
}).sort_values('MI 점수', ascending=False)

print(mi_df.to_string(index=False))
print()
print(f"MI 점수 최고 피처: {mi_df.iloc[0]['피처']} "
      f"({mi_df.iloc[0]['MI 점수']:.4f})")
print(f"MI 점수 최저 피처: {mi_df.iloc[-1]['피처']} "
      f"({mi_df.iloc[-1]['MI 점수']:.4f})")
print()

# MI 시각화
fig, ax = plt.subplots(figsize=(10, 10))
colors = ['#E24B4A' if v > 0.02 else '#888780'
          for v in mi_df['MI 점수']]
ax.barh(mi_df['피처'], mi_df['MI 점수'],
        color=colors, edgecolor='white')
ax.set_xlabel('Mutual Information 점수')
ax.set_title('피처 중요도 (Mutual Information)',
             fontsize=13, fontweight='bold')
ax.axvline(x=0.02, color='#185FA5',
           linestyle='--', linewidth=1.5,
           label='기준선 (0.02)')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/eda_mutual_information.png',
            dpi=150, bbox_inches='tight')
plt.show()

print("MI 그래프 저장 완료: outputs/eda_mutual_information.png")
print()

# ==========================================================
# EDA 4-2. VIF (다중공선성 확인)
# ==========================================================

print("=" * 60)
print("EDA 4-2. VIF (다중공선성 확인)")
print("=" * 60)

from statsmodels.stats.outliers_influence import (
    variance_inflation_factor
)

X_vif = train_raw[FEATURE_COLS].fillna(
    train_raw[FEATURE_COLS].median()
)

vif_data = []
for i, col in enumerate(FEATURE_COLS):
    vif = variance_inflation_factor(
        X_vif.values, i
    )
    vif_data.append({'피처': col, 'VIF': round(vif, 2)})

vif_df = pd.DataFrame(vif_data)\
    .sort_values('VIF', ascending=False)

print(f"{'피처':<30} {'VIF':>8}  {'판정'}")
print("-" * 50)

for _, row in vif_df.iterrows():
    if row['VIF'] >= 10:
        flag = "❌ 다중공선성 심각"
    elif row['VIF'] >= 5:
        flag = "⚠️ 검토 필요"
    else:
        flag = "✅ 정상"
    print(f"  {row['피처']:<28} {row['VIF']:>8.2f}  {flag}")

print()

high_vif = vif_df[vif_df['VIF'] >= 10]
if len(high_vif) > 0:
    print(f"VIF 10 이상 피처: {len(high_vif)}개 → 제거 검토 필요")
    print(high_vif['피처'].tolist())
else:
    print("VIF 10 이상 피처 없음 ✅")
print()

# ==========================================================
# EDA 5. 누수(Leakage) 의심 탐지
# ==========================================================

print("=" * 60)
print("EDA 5. Data Leakage 탐지")
print("=" * 60)

# 5-1. 단일 피처 AUROC 확인
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

print("단일 피처 AUROC (Leakage 탐지)")
print("-" * 50)

X_temp = train_raw[FEATURE_COLS].fillna(
    train_raw[FEATURE_COLS].median()
)
y_temp = train_raw['aki_label'].values

single_auroc = []

for col in FEATURE_COLS:
    x = X_temp[[col]].values
    try:
        lr = LogisticRegression(random_state=42)
        lr.fit(x, y_temp)
        proba = lr.predict_proba(x)[:, 1]
        auroc = roc_auc_score(y_temp, proba)
        single_auroc.append({
            '피처': col,
            'AUROC': round(auroc, 4)
        })
    except:
        pass

auroc_df = pd.DataFrame(single_auroc)\
    .sort_values('AUROC', ascending=False)

print(f"{'피처':<30} {'AUROC':>8}  {'판정'}")
print("-" * 55)

for _, row in auroc_df.iterrows():
    if row['AUROC'] >= 0.90:
        flag = "❌ Leakage 강력 의심"
    elif row['AUROC'] >= 0.80:
        flag = "⚠️ 요주의"
    elif row['AUROC'] >= 0.70:
        flag = "확인 권장"
    else:
        flag = "✅ 정상"
    print(f"  {row['피처']:<28} {row['AUROC']:>8.4f}  {flag}")

print()

max_auroc_row = auroc_df.iloc[0]
print(f"단일 피처 최고 AUROC: "
      f"{max_auroc_row['피처']} ({max_auroc_row['AUROC']:.4f})")

if max_auroc_row['AUROC'] >= 0.90:
    print("❌ Data Leakage 의심 → 즉시 해당 피처 확인")
elif max_auroc_row['AUROC'] >= 0.80:
    print("⚠️ 요주의 피처 존재 → 임상적 의미 재확인")
else:
    print("✅ 단일 피처 AUROC 정상 범위")
print()

# 5-2. 메타 컬럼 포함 여부 최종 확인
print("=" * 60)
print("메타 컬럼 포함 여부 최종 확인")
print("=" * 60)

DANGER_COLS = [
    'aki_label', 'aki_stage',
    'aki_onset_time', 'prediction_cutoff'
]

for col in DANGER_COLS:
    if col in FEATURE_COLS:
        print(f"❌ {col} 피처에 포함됨 → 즉시 제거 필요")
    else:
        print(f"✅ {col} 피처에 미포함")

print()

# 5-3. 전체 피처로 간단한 모델 성능 확인
print("=" * 60)
print("전체 피처 Logistic Regression AUROC 확인")
print("=" * 60)

X_all = train_raw[FEATURE_COLS].fillna(
    train_raw[FEATURE_COLS].median()
)
y_all = train_raw['aki_label'].values

from sklearn.preprocessing import StandardScaler

scaler_temp = StandardScaler()
X_all_scaled = scaler_temp.fit_transform(X_all)

lr_all = LogisticRegression(
    max_iter=1000, random_state=42
)
lr_all.fit(X_all_scaled, y_all)
proba_all = lr_all.predict_proba(X_all_scaled)[:, 1]
auroc_all  = roc_auc_score(y_all, proba_all)

print(f"전체 피처 AUROC: {auroc_all:.4f}")

if auroc_all >= 0.99:
    print("❌ AUROC 0.99 이상 → Data Leakage 강력 의심")
    print("   피처 목록 즉시 재확인 필요")
elif auroc_all >= 0.90:
    print("⚠️ AUROC 0.90 이상 → 요주의")
    print("   상관관계 높은 피처 재확인 권장")
else:
    print("✅ AUROC 정상 범위 → Leakage 없음")
print()

# ==========================================================
# EDA 6-1. 결측 패턴 시각화
# ==========================================================

print("=" * 60)
print("EDA 6-1. 결측 패턴 시각화")
print("=" * 60)

# 결측 패턴은 이상치 처리 후 스케일링 전 데이터 기준
# train_missing_processed.csv 사용
train_missing = pd.read_csv("data/train_missing_processed.csv")

# 결측률 10% 이상 피처만 선택
high_missing = train_missing[FEATURE_COLS]\
    .isnull().mean()
high_missing = high_missing[high_missing > 0.1]\
    .sort_values(ascending=False)

print(f"결측률 10% 이상 피처: {len(high_missing)}개")
for col, pct in high_missing.items():
    print(f"  {col:<30} {pct*100:.1f}%")
print()

if len(high_missing) == 0:
    print("결측값이 모두 처리된 상태입니다")
    print("→ 결측 패턴 시각화는 이상치 처리 직후 데이터")
    print("  (train_outlier_processed.csv) 기준으로 확인")
    print()

    # 이상치 처리 후 데이터로 재시도
    train_outlier = pd.read_csv(
        "data/train_outlier_processed.csv"
    )
    high_missing2 = train_outlier[FEATURE_COLS]\
        .isnull().mean()
    high_missing2 = high_missing2[
        high_missing2 > 0.1
    ].sort_values(ascending=False)

    print(f"이상치 처리 후 결측률 10% 이상: {len(high_missing2)}개")
    for col, pct in high_missing2.items():
        print(f"  {col:<30} {pct*100:.1f}%")
    print()

    # 히트맵 데이터
    missing_source = train_outlier
    missing_cols   = high_missing2.index
else:
    missing_source = train_missing
    missing_cols   = high_missing.index

# 히트맵 생성
if len(missing_cols) > 0:
    fig, ax = plt.subplots(figsize=(14, 8))
    missing_data = missing_source[missing_cols].isnull()

    sns.heatmap(
        missing_data.sample(500, random_state=42).T,
        cmap=['#185FA5', '#E24B4A'],
        cbar=False,
        ax=ax,
        yticklabels=True,
        xticklabels=False
    )
    ax.set_title('결측 패턴 (파란색=값 있음, 빨간색=결측)\n'
                 '샘플 500명 기준',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('피처')
    ax.set_xlabel('환자 (샘플)')
    plt.tight_layout()
    plt.savefig('outputs/eda_missing_pattern.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("결측 패턴 그래프 저장: outputs/eda_missing_pattern.png")
else:
    print("결측값 없음 → 결측 패턴 그래프 생성 생략")
print()

# ==========================================================
# EDA 7. 인구통계 비교
# ==========================================================

print("=" * 60)
print("EDA 7. 인구통계 비교")
print("=" * 60)

# 7-1. 연령별 AKI 발생률
print("연령별 AKI 발생률")
print("-" * 40)

train_raw['age_group'] = pd.cut(
    train_raw['age'],
    bins=[18, 40, 60, 70, 80, 120],
    labels=['18-40', '41-60', '61-70', '71-80', '80+']
)

age_aki = train_raw.groupby('age_group', observed=True)\
    .agg(
        total=('aki_label', 'count'),
        aki=('aki_label', 'sum')
    )
age_aki['aki_rate'] = (
    age_aki['aki'] / age_aki['total'] * 100
).round(1)

print(age_aki.to_string())
print()

# 7-2. 성별 AKI 발생률
print("성별 AKI 발생률")
print("-" * 40)

gender_aki = train_raw.groupby('gender')\
    .agg(
        total=('aki_label', 'count'),
        aki=('aki_label', 'sum')
    )
gender_aki['aki_rate'] = (
    gender_aki['aki'] / gender_aki['total'] * 100
).round(1)

print(gender_aki.to_string())
print()

# 7-3. 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 연령별
ax1 = axes[0]
bars = ax1.bar(
    age_aki.index,
    age_aki['aki_rate'],
    color='#185FA5', alpha=0.8, edgecolor='white'
)
ax1.set_title('연령대별 AKI 발생률', fontsize=12,
              fontweight='bold')
ax1.set_xlabel('연령대')
ax1.set_ylabel('AKI 발생률 (%)')
for bar, (_, row) in zip(bars, age_aki.iterrows()):
    ax1.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.3,
        f"{row['aki_rate']}%",
        ha='center', fontsize=10, fontweight='bold'
    )
ax1.set_ylim(0, max(age_aki['aki_rate']) * 1.2)

# 성별
ax2 = axes[1]
gender_labels = gender_aki.index.tolist()
bars2 = ax2.bar(
    gender_labels,
    gender_aki['aki_rate'],
    color=['#4A90D9', '#E24B4A'],
    alpha=0.8, edgecolor='white', width=0.4
)
ax2.set_title('성별 AKI 발생률', fontsize=12,
              fontweight='bold')
ax2.set_xlabel('성별')
ax2.set_ylabel('AKI 발생률 (%)')
for bar, rate in zip(bars2, gender_aki['aki_rate']):
    ax2.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.3,
        f"{rate}%",
        ha='center', fontsize=11, fontweight='bold'
    )
ax2.set_ylim(0, max(gender_aki['aki_rate']) * 1.2)

plt.suptitle('인구통계별 AKI 발생률 (Train 기준)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/eda_demographics.png',
            dpi=150, bbox_inches='tight')
plt.show()

print("인구통계 그래프 저장: outputs/eda_demographics.png")
print()

# ==========================================================
# EDA 최종 요약 저장
# ==========================================================

print("=" * 60)
print("EDA 최종 요약")
print("=" * 60)

import pickle

eda_summary = {

    # 왜도 분석 결과
    'transform_candidates': {
        'log1p': [
            'urine_output_6h',
            'vasopressor_hours',
            'lactate_max',
            'lactate_mean',
        ],
        'sqrt': [
            'creatinine_delta',
            'norepi_dose_max',
            'map_below65_hours',
        ],
        'binary_skip': [
            'vasopressor_flag',
            'oliguria_flag',
        ]
    },

    # 제거 권장 피처
    'drop_candidates': {
        'urine_ml_kg_hr': 'urine_output_sum과 상관계수 1.0'
    },

    # Leakage 탐지 결과
    'leakage_check': {
        'max_single_auroc'    : 0.6158,
        'full_model_auroc'    : 0.8924,
        'meta_col_included'   : False,
        'result'              : '이상 없음'
    },

    # 인구통계 인사이트
    'demographics': {
        'age_highest_aki' : '71-80세 (31.5%)',
        'gender_diff'     : '남성 27.0% vs 여성 23.2%',
    },

    # 모델 학습 전 최종 확인사항
    'action_items': [
        'urine_ml_kg_hr 제거 (urine_output_sum과 동일)',
        'log1p 변환: urine_output_6h, vasopressor_hours, '
        'lactate_max, lactate_mean',
        'sqrt 변환: creatinine_delta, norepi_dose_max, '
        'map_below65_hours',
        'VIF는 XGBoost에서 문제 없음 → 제거 불필요',
    ]
}

with open('outputs/eda_summary.pkl', 'wb') as f:
    pickle.dump(eda_summary, f)

print("eda_summary.pkl 저장 완료")
print()

print("=" * 60)
print("모델 학습 전 액션 아이템")
print("=" * 60)
for i, item in enumerate(eda_summary['action_items'], 1):
    print(f"  {i}. {item}")
print()
print("EDA 완료 ✅")