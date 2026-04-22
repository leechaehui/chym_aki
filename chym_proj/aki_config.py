# 설정 파일
# 컬럼 바뀌면 여기서 수정
# 다른 코드에서 import해서 사용됨

# 사용할 feature(입력 변수)
FEATURES = ['creatinine', 'bun', 'urine_output', 'mean_bp']
# 타겟 변수 (예측할 값)
# → AKI 발생 여부 (0 or 1)
TARGET = 'aki_label'

# 데이터 분할 비율 (20% test)
TEST_SIZE = 0.2
VALID_SIZE = 0.2   # train에서 추가 분리
# 랜덤 고정 (재현성 확보)
RANDOM_STATE = 42

# 모델 저장 경로
MODEL_PATH = "aki_xgb_model.pkl"
# threshold 저장 경로
THRESHOLD_PATH = "aki_best_threshold.pkl"
# 데이터 분할 결과 저장 경로
DATA_SPLIT_PATH = "aki_data_split.pkl"

# 병렬 처리 (CPU 최대 사용)
N_JOBS = -1
# 교차검증 fold 수 (5겹)
CV_FOLDS = 5
# GridSearch 평가 기준
SCORING = "roc_auc"