# ==========================================================
# Patient-Level Stratified Split
#
# 목적:
#   1. 동일 환자(subject_id)가 Train/Valid/Test에
#      동시에 포함되는 것을 방지
#   2. AKI 발생 비율을 각 데이터셋에서 최대한 동일하게 유지
#   3. Patient-level Leakage 방지
#
# 최종 분할 비율
#   - Train : 70%
#   - Valid : 15%
#   - Test  : 15%
#
# 분할 기준
#   - Grouping      : subject_id
#   - Stratification: patient-level AKI label
#
# 출력 파일
#   - train.csv
#   - valid.csv
#   - test.csv
# ==========================================================

import pandas as pd
from sklearn.model_selection import train_test_split

# 재현 가능한 결과를 위해 random seed 고정
RANDOM_STATE = 42


# ==========================================================
# 1. 데이터 불러오기
# ==========================================================

df = pd.read_csv("final_features_48h.csv")

print("=" * 60)
print("원본 데이터셋 정보")
print("=" * 60)

print(f"전체 샘플 수      : {len(df):,}")
print(f"고유 환자 수      : {df['subject_id'].nunique():,}")
print(f"전체 AKI 발생률   : {df['aki_label'].mean():.4f}")
print()


# ==========================================================
# 2. 환자 수준(Patient-level) AKI Label 생성
# ==========================================================
# 동일 환자의 stay 중 하나라도 AKI가 발생하면
# 해당 환자는 AKI 환자로 간주
#
# 이 patient_aki는 오직 데이터 분할을 위한 용도이며
# 실제 모델 학습에는 사용되지 않음
#
# ==========================================================

patient_df = (
    df.groupby("subject_id", as_index=False)
      .agg(patient_aki=("aki_label", "max"))
)

print("=" * 60)
print("환자 수준 데이터 정보")
print("=" * 60)

print(f"고유 환자 수           : {len(patient_df):,}")
print(f"환자 수준 AKI 발생률   : {patient_df['patient_aki'].mean():.4f}")
print()


# ==========================================================
# 3. Train(70%) / Temp(30%) 분할
# ==========================================================
#
# Stratification 기준:
#   patient_aki
#
# 목적:
#   Train 세트와 Temp 세트의 AKI 비율을
#   최대한 비슷하게 유지
#
# ==========================================================

train_patients, temp_patients = train_test_split(
    patient_df,
    test_size=0.30,
    stratify=patient_df["patient_aki"],
    random_state=RANDOM_STATE
)


# ==========================================================
# 4. Temp(30%) → Valid(15%) / Test(15%)
# ==========================================================
#
# Temp 데이터셋을 다시 절반씩 나누어
# Validation과 Test 세트 생성
#
# 결과:
#   Train = 70%
#   Valid = 15%
#   Test  = 15%
#
# ==========================================================

valid_patients, test_patients = train_test_split(
    temp_patients,
    test_size=0.50,
    stratify=temp_patients["patient_aki"],
    random_state=RANDOM_STATE
)


# ==========================================================
# 5. 각 데이터셋에 포함될 환자 ID 추출
# ==========================================================
#
# 이후 원본 데이터(stay 단위)에서
# 해당 환자의 모든 stay를 가져오기 위해 사용
#
# ==========================================================

train_ids = set(train_patients["subject_id"])
valid_ids = set(valid_patients["subject_id"])
test_ids = set(test_patients["subject_id"])


# ==========================================================
# 6. 원본 데이터에서 실제 Train/Valid/Test 생성
# ==========================================================
#
# 중요:
# 여기서는 patient_df를 사용하는 것이 아니라
# 원본 stay-level 데이터를 사용함
#
# 따라서 동일 환자의 모든 stay가
# 하나의 데이터셋에만 포함됨
#
# ==========================================================

train_df = df[df["subject_id"].isin(train_ids)].copy()

valid_df = df[df["subject_id"].isin(valid_ids)].copy()

test_df = df[df["subject_id"].isin(test_ids)].copy()


# ==========================================================
# 7. CSV 파일 저장
# ==========================================================

train_df.to_csv("train.csv", index=False)
valid_df.to_csv("valid.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("=" * 60)
print("CSV 저장 완료")
print("=" * 60)
print("train.csv")
print("valid.csv")
print("test.csv")
print()


# ==========================================================
# 8. 데이터셋 요약 함수
# ==========================================================
#
# 각 데이터셋의
#   - 샘플 수
#   - 환자 수
#   - AKI 비율
# 확인
#
# ==========================================================

def summarize(name, data):

    print(f"[{name}]")

    print(f"샘플 수     : {len(data):,}")

    print(f"고유 환자 수 : {data['subject_id'].nunique():,}")

    print(f"AKI 발생률  : {data['aki_label'].mean():.4f}")

    print()


# ==========================================================
# 9. 최종 분할 결과 확인
# ==========================================================

print("=" * 60)
print("최종 데이터셋 요약")
print("=" * 60)

summarize("TRAIN", train_df)
summarize("VALID", valid_df)
summarize("TEST", test_df)


# ==========================================================
# 10. 환자 비율 확인
# ==========================================================
#
# 실제 환자 기준으로
# 70/15/15 비율이 잘 유지되는지 확인
#
# ==========================================================

total_patients = len(patient_df)

print("=" * 60)
print("환자 기준 분할 비율")
print("=" * 60)

print(f"Train : {len(train_patients)/total_patients:.3%}")
print(f"Valid : {len(valid_patients)/total_patients:.3%}")
print(f"Test  : {len(test_patients)/total_patients:.3%}")

print()


# ==========================================================
# 11. Patient Leakage 검사
# ==========================================================
#
# 동일 환자가 여러 데이터셋에
# 포함되어 있으면 안 됨
#
# 정상 결과:
#   Train ∩ Valid = 0
#   Train ∩ Test  = 0
#   Valid ∩ Test  = 0
#
# ==========================================================

train_valid_overlap = train_ids & valid_ids
train_test_overlap = train_ids & test_ids
valid_test_overlap = valid_ids & test_ids

print("=" * 60)
print("환자 중복 검사")
print("=" * 60)

print(f"Train ∩ Valid : {len(train_valid_overlap)}")
print(f"Train ∩ Test  : {len(train_test_overlap)}")
print(f"Valid ∩ Test  : {len(valid_test_overlap)}")

print()

if (
    len(train_valid_overlap) == 0
    and len(train_test_overlap) == 0
    and len(valid_test_overlap) == 0
):
    print("✅ Patient Leakage 없음")
else:
    print("❌ Patient Leakage 발견")

print()


# ==========================================================
# 12. Stay-level AKI 비율 확인
# ==========================================================
#
# 실제 모델 학습에 사용되는 데이터 기준
# AKI 비율 확인
#
# ==========================================================

print("=" * 60)
print("Stay-level AKI 분포")
print("=" * 60)

distribution = pd.DataFrame({
    "Dataset": ["All", "Train", "Valid", "Test"],
    "AKI_Rate": [
        df["aki_label"].mean(),
        train_df["aki_label"].mean(),
        valid_df["aki_label"].mean(),
        test_df["aki_label"].mean()
    ]
})

print(distribution)

print()


# ==========================================================
# 13. Patient-level AKI 비율 확인
# ==========================================================
#
# Stratification이 제대로 적용되었는지 확인
#
# ==========================================================

train_patient_rate = train_patients["patient_aki"].mean()
valid_patient_rate = valid_patients["patient_aki"].mean()
test_patient_rate = test_patients["patient_aki"].mean()

print("=" * 60)
print("Patient-level AKI 분포")
print("=" * 60)

print(f"Train : {train_patient_rate:.4f}")
print(f"Valid : {valid_patient_rate:.4f}")
print(f"Test  : {test_patient_rate:.4f}")

print()
print("데이터 분할 완료")