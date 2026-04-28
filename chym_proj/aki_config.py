# 모델링 전체에서 공통으로 사용하는 설정값을 한 곳에 모아둔 파일입니다.
#
# 왜 필요한가?
# - 여러 파일에서 같은 변수명, 저장 경로, 랜덤 시드 등을 반복해서 쓰면 관리가 어려워집니다.
# - 나중에 feature 컬럼이 바뀌거나 모델 저장 경로를 바꿀 때 이 파일만 수정하면 됩니다.

# 사용할 feature(입력 변수)
# 현재는 임시 변수명입니다.
# 최종 data.csv가 완성되면 여기에 최종 feature 컬럼명을 넣으면 됩니다.
# 단, 현재 aki_utils.py의 time_based_split()은 숫자형 feature를 자동 선택하도록 되어 있어
# 타겟 변수 (예측할 값)
TARGET = 'aki_label'
# 랜덤 시드 (재현성 유지)
RANDOM_STATE = 42
# 모델 및 결과 저장 경로
MODEL_PATH = "aki_xgb_model.pkl"
THRESHOLD_PATH = "aki_best_threshold.pkl"
DATA_SPLIT_PATH = "aki_data_split.pkl"
# 병렬 처리
N_JOBS = -1
# Optuna 반복 횟수
N_TRIALS = 50
# 입력 데이터
DATA_PATH = "final_features_48h.csv"