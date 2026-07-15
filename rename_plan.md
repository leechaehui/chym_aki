# AKI 한글 폴더/파일 이름 영문 일괄 변경 계획서

이 문서는 `chym_aki/AKI/` 디렉터리 하위의 한글 폴더 및 파일명을 영문으로 일괄 변경하고, 백엔드 및 테스트 코드 내의 경로 참조를 함께 수정하기 위한 계획입니다.

## 폴더/파일명 변경 매핑 (Mapping)

### 1. AKI 하위 주요 디렉터리
* `EDA 및 변수변환` ➔ `eda_and_transform`
* `데이터 분할` ➔ `data_split`
* `레이블` ➔ `labels`
* `모델링` ➔ `modeling`
* `변수변환` ➔ `transform`
* `전처리` ➔ `preprocessing`
* `코호트` ➔ `cohort`
* `피처` ➔ `feature`

### 2. 모델링 (`modeling/`) 하위 디렉터리
* `modeling/다중분류` ➔ `modeling/multiclass`
* `modeling/이진분류` ➔ `modeling/binary`
* `modeling/이진분류/1단계_Logistic Regression` ➔ `modeling/binary/stage1_logistic_regression`
* `modeling/이진분류/1단계_Random Forest` ➔ `modeling/binary/stage1_random_forest`
* `modeling/이진분류/1단계_XGBoost` ➔ `modeling/binary/stage1_xgboost`
* `modeling/다중분류/2단계_CatBoost` ➔ `modeling/multiclass/stage2_catboost`
* `modeling/다중분류/2단계_LGBM` ➔ `modeling/multiclass/stage2_lgbm`
* `modeling/다중분류/2단계_XGBoost` ➔ `modeling/multiclass/stage2_xgboost`

### 3. 전처리 (`preprocessing/`) 하위 디렉터리
* `preprocessing/전처리(권미정)` ➔ `preprocessing/preprocess_gwon`
* `preprocessing/전처리(권미정)/이상치` ➔ `preprocessing/preprocess_gwon/outliers`
* `preprocessing/전처리(권미정)/클래스 불균형` ➔ `preprocessing/preprocess_gwon/class_imbalance`
* `preprocessing/전처리(이채희)` ➔ `preprocessing/preprocess_lee`
* `preprocessing/전처리_최종코드파일` ➔ `preprocessing/preprocess_final`
* `preprocessing/전처리_최종코드파일/data(데이터분할)` ➔ `preprocessing/preprocess_final/data_split`
* `preprocessing/최종 데이터셋_전처리 완료된` ➔ `preprocessing/final_dataset`

### 4. 모델링 (`modeling/`) 하위 파일명 변경
* `modeling/multiclass/stage2_catboost/CatBoost_2단계(변수제거).py` ➔ `CatBoost_stage2_exclude_features.py`
* `modeling/multiclass/stage2_catboost/CatBoost_2단계(변수포함).py` ➔ `CatBoost_stage2_include_features.py`
* `modeling/multiclass/stage2_catboost/CatBoost_2단계(변수포함,threshold조정).py` ➔ `CatBoost_stage2_include_features_threshold_adj.py`
* `modeling/multiclass/stage2_xgboost/XGBoost_2단계(변수제거,class weight).py` ➔ `XGBoost_stage2_exclude_features_class_weight.py`
* `modeling/multiclass/stage2_xgboost/XGBoost_2단계(변수제거,SMOTE).py` ➔ `XGBoost_stage2_exclude_features_smote.py`
* `modeling/multiclass/stage2_xgboost/XGBoost_2단계(변수포함,class weight).py` ➔ `XGBoost_stage2_include_features_class_weight.py`
* `modeling/multiclass/stage2_xgboost/XGBoost_2단계(변수포함,class weight,클래스비율변경).py` ➔ `XGBoost_stage2_include_features_class_weight_ratio_adj.py`
* `modeling/multiclass/stage2_xgboost/XGBoost_2단계(변수포함,SMOTE).py` ➔ `XGBoost_stage2_include_features_smote.py`
* `modeling/binary/stage1_xgboost/XGBoost_1단계.py` ➔ `XGBoost_stage1.py`

---

## 코드 및 문서 수정 내역

### 1. 백엔드 및 테스트 코드 경로 수정
* **`tests/vv_runners/run_aki_validation.py`**
  * `DATA` 경로 ➔ `preprocessing/final_dataset/test_final.csv`
* **`backend/services/icu_monitor_service.py`**
  * `FEATURE_CSV_DIR` 경로 ➔ `preprocessing/final_dataset`
* **`backend/services/model_metrics_service.py`**
  * `TEST_CSV` 경로 ➔ `preprocessing/final_dataset/test_final.csv`
* **`backend/services/aki_feature_transform.py`**
  * `_TRANSFORM_INFO` 경로 ➔ `transform/transform_info.pkl`
* **`backend/ml_models/train_aki_models.py`**
  * `DATA_DIR` 경로 ➔ `preprocessing/final_dataset`
  * `AKI_MODELS_DIR` 경로 ➔ `modeling/models`
* **`backend/db/load_final_features_48h.py`**
  * `CSV_PATH` 경로 ➔ `feature/final_features_48h.csv`

### 2. 프로젝트 문서 정보 수정
* **`project_summary.md`**: 폴더 구조 설명 내 한글 디렉터리명을 영문으로 변경
* **`backend/README.md`**: `modeling/final_aki_model.py` 참조 경로 수정

---

## 검증 방안
1. **한글 파일/디렉터리 잔재 확인**: 스크립트를 통해 `AKI/` 하위에 한글이 포함된 경로가 존재하는지 검사
2. **Validation Runner 실행**: `run_aki_validation.py`를 실행하여 모델 파이프라인의 입출력 정상 작동 여부 및 리포트 재생성 확인
