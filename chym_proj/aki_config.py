# 모델링 전체에서 공통으로 사용하는 설정값을 한 곳에 모아둔 파일입니다.
#
# 왜 필요한가?
# - 여러 파일에서 같은 변수명, 저장 경로, 랜덤 시드 등을 반복해서 쓰면 관리가 어려워집니다.
# - 나중에 feature 컬럼이 바뀌거나 모델 저장 경로를 바꿀 때 이 파일만 수정하면 됩니다.

# 사용할 feature(입력 변수)
# 현재는 임시 변수명입니다.
# 최종 data.csv가 완성되면 여기에 최종 feature 컬럼명을 넣으면 됩니다.
# 단, 현재 aki_utils.py의 time_based_split()은 숫자형 feature를 자동 선택하도록 되어 있어
# 이 FEATURES는 주로 "명시적으로 관리하고 싶은 변수 목록" 또는 load_data()용으로 남겨둔 설정입니다.
FEATURES = ['creatinine', 'bun', 'urine_output', 'mean_bp']
# 타겟 변수 (예측할 값)
# → AKI 발생 여부 (0 or 1)
TARGET = 'aki_label'

# 데이터 분할 비율
# TEST_SIZE: 최종 평가용 test set 비율
# VALID_SIZE: train 안에서 validation set을 추가로 나눌 때 쓰는 비율입니다.
# 현재 main 학습 흐름은 time_based_split()을 사용하므로, 이 값은 split_data()를 쓸 때 주로 사용됩니다.
# 데이터 분할 비율 (20% test)
TEST_SIZE = 0.2
VALID_SIZE = 0.2   # train에서 추가 분리

# 랜덤 시드
# 랜덤성이 들어가는 모델 학습/분할에서 결과를 재현 가능하게 만들기 위한 값입니다.
# 랜덤 고정 (재현성 확보)
RANDOM_STATE = 42


# 모델 저장 경로
MODEL_PATH = "aki_xgb_model.pkl"
# threshold 저장 경로
THRESHOLD_PATH = "aki_best_threshold.pkl"
# train/valid/test로 나눈 데이터 저장 경로
# evaluation, threshold tuning, SHAP 파일에서 동일한 데이터 분할을 불러오기 위해 저장합니다.
DATA_SPLIT_PATH = "aki_data_split.pkl"

# 병렬 처리 (CPU 최대 사용)
N_JOBS = -1
# 교차검증 fold 수
# 5겹 교차검증은 학습 데이터를 5등분해서 여러 번 검증하는 방식입니다.
CV_FOLDS = 5
# GridSearch 평가 기준
SCORING = "roc_auc"