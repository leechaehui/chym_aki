"""
================================================================================
train_drug_xgboost.py
================================================================================
목적  : MIMIC-IV 기반 ICU 급성 신장 손상(AKI) XGBoost 예측 모델
입력  : 03_features_drug_(트랙C) (03번 SQL 산출물)
출력  : aki_xgb_output/ 폴더 (모델, 파라미터, 성능 차트, SHAP)

실행법:
  python 04_train_xgb.py            # Optuna 100회 튜닝 포함 (30분~2시간)
  python 04_train_xgb.py --no-tune  # 기본 파라미터로 빠른 실행 (5분)

XGBoost 전용 설계 원칙:
  1. 결측치를 채우지 않음   → XGBoost가 NaN 분기 방향을 직접 학습
  2. 스케일링 하지 않음     → 트리 분기는 rank 기반이므로 절댓값 무관
  3. scale_pos_weight 사용 → SMOTE 대신 손실 함수 가중치로 불균형 대응
  4. subject_id 단위 분할  → stay_id 단위 분할 시 환자 정보 누수 발생
  5. eval_metric="aucpr"   → AUROC는 불균형 데이터에서 낙관적, AUPRC가 더 엄격
================================================================================
"""

# ── 표준 라이브러리 ─────────────────────────────────────────────────────────
import argparse          # --no-tune 같은 CLI 인자 파싱
import warnings          # Optuna 실험적 기능 경고 억제
import json              # best_params.json, summary.json 저장
from pathlib import Path # 파일 경로를 문자열 대신 객체로 관리 (OS 독립)

# ── 데이터 처리 ─────────────────────────────────────────────────────────────
import numpy as np       # 배열 연산, 난수 생성(default_rng), linspace
import pandas as pd      # DataFrame 조작, SQL 결과 수신, pd.cut(버킷 변환)
from sqlalchemy import create_engine, text # PostgreSQL 연결 엔진 생성 (DB URL → engine 객체)
from config import DB_URL

engine = create_engine(DB_URL)
# DB 연결 함수
def db_connection():
    try:
        # 2. DB 연결
        with engine.connect() as conn:
            print("DB 연결 성공!")
    except Exception as e:
        print(f"연결 실패: {e}")

# ── 모델 ────────────────────────────────────────────────────────────────────
import xgboost as xgb    # XGBClassifier: 트리 앙상블 기반 이진 분류기

# ── 해석 가능성 ─────────────────────────────────────────────────────────────
import shap              # SHAP TreeExplainer: 각 피처가 예측에 기여한 정도 계산

# ── 하이퍼파라미터 최적화 ────────────────────────────────────────────────────
import optuna            # Bayesian 최적화 (TPE 알고리즘으로 AUPRC 최대화)

# ── 시각화 ──────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")    # 화면 없는 서버 환경에서도 이미지 파일로 저장 가능하게
import matplotlib.pyplot as plt

# ── 평가 지표·교차 검증 ──────────────────────────────────────────────────────
from sklearn.model_selection import StratifiedGroupKFold
# StratifiedGroupKFold:
#   - Group  → subject_id 단위로 묶어서 분할 (같은 환자가 Train/Val 양쪽에 못 들어감)
#   - Strat  → 각 fold에서 AKI 양성 비율을 균등하게 유지
#   일반 StratifiedKFold는 Group 조건을 지원하지 않음 → 환자 누수 발생

from sklearn.metrics import (
    roc_auc_score,           # AUROC: 전반적 판별력 (0~1, 높을수록 좋음)
    average_precision_score, # AUPRC: Precision-Recall 곡선 아래 면적 (불균형 데이터 기준)
    classification_report,   # Precision / Recall / F1 / Support 출력
    confusion_matrix,        # TP / FP / FN / TN 2×2 행렬
    RocCurveDisplay,         # ROC 곡선 시각화 (sklearn 내장)
    PrecisionRecallDisplay,  # PR 곡선 시각화
)
from sklearn.calibration import calibration_curve
# calibration_curve: 예측 확률이 실제 발생률과 얼마나 일치하는지 확인
# 잘 보정된 모델은 대각선(y=x)에 가까운 곡선을 그림

from scipy.special import expit  # sigmoid 함수 (필요 시 로짓 변환에 사용)

# ── Optuna 로그 레벨 조정 ────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)
# WARNING 이하 로그 억제 → 진행 바(tqdm)만 출력되게


# ================================================================================
# 섹션 0. 전역 설정
# ================================================================================

# DB 연결 문자열 — 실행 전 반드시 본인 환경에 맞게 수정
# 형식: "postgresql+psycopg2://유저:비밀번호@호스트:포트/DB명"

# ── 피처 컬럼 목록 ────────────────────────────────────────────────────────────
# 이 리스트를 수정하면 전처리·학습·SHAP 해석 전체에 자동 반영됨
# 피처 추가·삭제 시 이 곳만 수정하면 됨
FEATURE_COLS = [

    # ── [그룹 1] ICU 약물 노출 (처방 여부 + 노출 시간/용량) ──────────────────
    "vancomycin_rx",             # 반코마이신 처방 여부 (0/1)
    "vancomycin_exposure_hours", # 반코마이신 누적 노출 시간 (연속형)
    "piptazo_rx",                # 피페라실린/타조박탐(Zosyn) 처방 여부
    "aminoglycoside_rx",         # 아미노글리코사이드계 항생제 처방 여부
    "amphotericin_b_rx",         # 암포테리신B(항진균제) 처방 여부
    "carbapenem_rx",             # 카바페넴계 항생제 처방 여부
    "ketorolac_rx",              # 케토롤락(NSAIDs) 처방 여부
    "nsaid_any_rx",              # NSAIDs 계열 전체 처방 여부 (케토롤락 포함)
    "ace_inhibitor_rx",          # ACE 억제제(리시노프릴 등) 처방 여부
    "arb_rx",                    # ARB(로사르탄 등) 처방 여부
    "acei_arb_any_rx",           # ACEi 또는 ARB 중 하나라도 처방 여부
    "furosemide_rx",             # 푸로세미드(이뇨제) 처방 여부
    "furosemide_cumulative_mg",  # 푸로세미드 누적 투여 용량 (mg, 연속형)
    "tacrolimus_rx",             # 타크롤리무스(면역억제제) 처방 여부
    "cyclosporine_rx",           # 사이클로스포린(면역억제제) 처방 여부
    "metformin_rx",              # 메트포르민(당뇨약) 처방 여부
    "ppi_rx",                    # PPI(위산억제제) 처방 여부

    # ── [그룹 2] 약물 조합 위험도 (병용 투여의 신독성 시너지) ─────────────────
    "vanco_piptazo_combo",       # 반코마이신 + Pip/Tazo 병용 (고위험 조합)
    "vanco_aminogly_combo",      # 반코마이신 + 아미노글리코사이드 병용
    "vanco_carbapenem_combo",    # 반코마이신 + 카바페넴 병용
    "nsaid_acei_combo",          # NSAIDs + ACEi/ARB 병용 (신혈류 이중 차단)
    "triple_whammy",             # NSAIDs + ACEi/ARB + 이뇨제 3중 병용 (극고위험)
    "diuretic_overload_flag",    # 이뇨제 과다 + 신독성 항생제 동시 (플래그)
    "metformin_risk_flag",       # 메트포르민 + 신독성 약물 동시 (플래그)
    "nephrotoxic_burden_score",  # 누적 신독성 부담 점수 (0~8, 높을수록 위험)
    "drug_risk_score",           # 약물 처방 위험 종합 점수 (0~5)

    # ── [그룹 3] ED(응급실) 초기 상태 ────────────────────────────────────────
    "ed_los_hours",              # 응급실 체류 시간 (연속형, 단위: 시간)
    "direct_icu_admit_flag",     # 응급실 경유 없이 직접 ICU 입원 여부 (0/1)
    "triage_sbp",                # 트리아지 수축기혈압 (mmHg, 결측 가능)
    "triage_hr",                 # 트리아지 심박수 (bpm, 결측 가능)
    "triage_temp",               # 트리아지 체온 (°C, 결측 가능)
    "triage_o2sat",              # 트리아지 산소포화도 (%, 결측 가능)
    "triage_ktas",               # 트리아지 중증도 (1~5, 낮을수록 중증)
    "triage_critical_flag",      # 트리아지 중증(1~2단계) 여부 (0/1)
    "nsaid_preadmission_flag",   # 입원 전 NSAIDs 복용 이력 (0/1)
    "acei_arb_preadmission_flag",# 입원 전 ACEi/ARB 복용 이력 (0/1)
    "metformin_preadmission_flag",# 입원 전 메트포르민 복용 이력 (0/1)
    "diuretic_preadmission_flag",# 입원 전 이뇨제 복용 이력 (0/1)
    "immunosuppressant_preadmission_flag", # 입원 전 면역억제제 복용 이력 (0/1)

    # ── [그룹 4] IV 수액 및 조영제 노출 ────────────────────────────────────────
    "iv_fluid_total_ml",         # ICU 내 총 IV 수액량 (mL, 연속형)
    "iv_fluid_per_kg",           # 체중당 IV 수액량 (mL/kg, 연속형)
    "contrast_exposure_flag",    # 조영제 노출 여부 (CT 촬영 등, 0/1)
    "weight_kg",                 # 체중 (kg, 연속형, 결측 가능)

    # ── [그룹 5] 파생 복합 지표 ──────────────────────────────────────────────
    "total_nephrotoxic_burden",  # ICU 약물 + 입원 전 복용 통합 신독성 부담
    "contrast_nsaid_combo",      # 조영제 × NSAIDs 병용 (CI-AKI 위험, 0/1)

    # ── [그룹 6] 결측 지시자 (전처리에서 생성) ───────────────────────────────
    "missing_triage_flag",       # triage_sbp가 NULL이면 1 (응급실 미경유 표시)
]

# ── 타깃·그룹·학습 설정 ────────────────────────────────────────────────────
TARGET_COL    = "aki_label"      # 예측 대상 컬럼: 0=Non-AKI, 1=AKI
GROUP_COL     = "subject_id"     # 환자 단위 그룹핑 기준 (데이터 누수 방지 핵심)
N_FOLDS       = 5                # Cross-Validation 폴드 수
OPTUNA_TRIALS = 100              # Optuna 하이퍼파라미터 탐색 횟수
OUTPUT_DIR    = Path("./aki_xgb_output")  # 모든 결과물 저장 폴더
OUTPUT_DIR.mkdir(exist_ok=True)  # 폴더 없으면 자동 생성, 있으면 그대로 사용


# ================================================================================
# 섹션 1. 데이터 로드
# ================================================================================

def load_data(db_url: str) -> pd.DataFrame:
    """
    PostgreSQL에서 AKI 피처 테이블을 로드한다.

    Parameters
    ----------
    db_url : str
        SQLAlchemy 연결 문자열
        예) "postgresql+psycopg2://user:pw@localhost:5432/mimic4"

    Returns
    -------
    pd.DataFrame
        stay_id 단위 원시 피처 DataFrame
        (aki_label IS NOT NULL인 행만 포함)
    """


    # SELECT에 명시된 컬럼만 메모리에 로드 → 불필요한 컬럼 제외로 메모리 절약
    # WHERE aki_label IS NOT NULL → Python이 아닌 DB에서 필터링 (속도 대폭 향상)
    query = """
        SELECT
            -- 메타 컬럼 (모델 피처로는 사용하지 않음, 분할·검증용)
            stay_id,
            subject_id,
            aki_label,
            aki_stage,           -- 전처리에서 드롭 (타깃 누수)
            is_pseudo_cutoff,    -- AKI 미발생군의 가상 cutoff 여부 표시
            hours_to_aki,        -- 전처리에서 드롭 (극심한 타깃 누수)

            -- ICU 약물 노출
            vancomycin_rx, vancomycin_exposure_hours,
            piptazo_rx, aminoglycoside_rx, amphotericin_b_rx, carbapenem_rx,
            ketorolac_rx, nsaid_any_rx,
            ace_inhibitor_rx, arb_rx, acei_arb_any_rx,
            furosemide_rx, furosemide_cumulative_mg,
            tacrolimus_rx, cyclosporine_rx, metformin_rx, ppi_rx,

            -- 약물 조합 위험도
            vanco_piptazo_combo, vanco_aminogly_combo, vanco_carbapenem_combo,
            nsaid_acei_combo, triple_whammy, diuretic_overload_flag,
            metformin_risk_flag, nephrotoxic_burden_score, drug_risk_score,

            -- ED 초기 상태
            ed_los_hours, direct_icu_admit_flag,
            triage_sbp, triage_hr, triage_temp, triage_o2sat,
            triage_ktas, triage_critical_flag,
            nsaid_preadmission_flag, acei_arb_preadmission_flag,
            metformin_preadmission_flag, diuretic_preadmission_flag,
            immunosuppressant_preadmission_flag,

            -- IV 수액·조영제
            iv_fluid_total_ml, iv_fluid_per_kg, contrast_exposure_flag,
            weight_kg, weight_source,   -- weight_source는 전처리에서 드롭

            -- 파생 복합 지표
            total_nephrotoxic_burden, contrast_nsaid_combo

        FROM aki_project.aki_drug_exposure_features
        WHERE aki_label IS NOT NULL
        -- aki_label이 NULL인 행은 학습에도 평가에도 쓸 수 없으므로 DB에서 제거
        -- Python에서 dropna() 하는 것보다 DB 단 필터가 훨씬 빠름
    """

    df = pd.read_sql(query, engine)

    # 기본 통계 출력: 로드된 행 수와 AKI 양성 비율 확인
    print(
        f"[load] {len(df):,}행 로드 완료  |  "
        f"AKI 양성: {df[TARGET_COL].mean():.1%}  |  "
        f"stay 수: {df['stay_id'].nunique():,}  |  "
        f"환자 수: {df[GROUP_COL].nunique():,}"
    )
    return df


# ================================================================================
# 섹션 2. 전처리 (XGBoost 특화)
# ================================================================================

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    XGBoost에 최적화된 전처리를 수행한다.

    XGBoost 전처리 원칙:
      - 결측치를 채우지 않는다 → XGBoost가 NaN을 만나면 학습 과정에서
        Left/Right 중 어느 방향이 손실을 더 줄이는지 자동으로 결정함
      - 스케일링하지 않는다   → 트리 분기는 "값 > X?" 형태이므로
        절댓값 크기나 분포 모양이 성능에 영향을 주지 않음
      - 이상치만 처리한다     → 임상적으로 불가능한 값(SBP=0 등)과
        극단값(99th %ile 초과)만 제거/대체

    Parameters
    ----------
    df : pd.DataFrame
        load_data()에서 받은 원시 DataFrame

    Returns
    -------
    pd.DataFrame
        전처리 완료 DataFrame
        (missing_triage_flag, vanco_duration_cat, furosemide_dose_cat 컬럼 추가)
    """
    df = df.copy()  # 원본 DataFrame 불변 원칙 (뷰가 아닌 독립 복사본 생성)

    # ── 2-A. 결측 지시자 변수 (Missing Indicator) ──────────────────────────────
    # triage_sbp가 NULL인 경우: 응급실을 경유하지 않고 직접 ICU에 입원한 환자
    # 이 NULL은 "데이터 없음"이 아니라 "응급실에 안 갔음"이라는 임상적 의미를 가짐
    # XGBoost는 NaN을 처리할 수 있지만, 이 구조적 패턴을 명시적으로 표시해두면
    # 모델이 해당 패턴과 AKI 위험의 관계를 더 명확히 학습할 수 있음
    df["missing_triage_flag"] = df["triage_sbp"].isna().astype(int)
    # 결과: 응급실 경유 없이 ICU 입원 → 1, 응급실 거침 → 0

    # ── 2-B. 임상 경계 클리핑 (생물학적으로 불가능한 이상치 제거) ──────────────
    # 데이터 입력 오류(SBP=0, HR=500 등)만 제거하고,
    # 임상적으로 가능한 극값(SBP=60 = 저혈압쇼크)은 건드리지 않음
    # 이유: SBP=60은 중요한 임상 신호 → 제거하면 정보 손실
    clips = {
        "triage_sbp":   (30, 300),  # 수축기혈압: 30~300 mmHg (생리적 한계)
        "triage_hr":    (20, 300),  # 심박수: 20~300 bpm
        "triage_temp":  (25,  45),  # 체온: 25~45°C (저체온~고열 범위)
        "triage_o2sat": (50, 100),  # 산소포화도: 50~100%
        "weight_kg":    (20, 300),  # 체중: 20~300 kg (성인 기준)
    }
    for col, (lo, hi) in clips.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
            # clip(lo, hi): lo 미만 → lo로, hi 초과 → hi로 대체

    # ── 2-C. 우편향 연속형 변수 Winsorize (99th percentile cap) ──────────────
    # 극단값(상위 1%)을 99번째 백분위값으로 대체
    # 예) furosemide 누적 용량이 비정상적으로 큰 환자 몇 명이 있을 때,
    #     그 값이 수백 개의 트리에서 잘못된 분기를 만들어낼 수 있음
    #
    # 왜 log1p나 StandardScaler 대신 Winsorize인가?
    #   - XGBoost는 rank 기반 분기 → 절댓값 크기 무관 → log1p 효과 없음
    #   - StandardScaler도 트리에는 영향 없음 → 생략
    #   - 오직 극단값 1%만 잘라내는 것으로 충분
    skewed_cols = [
        "furosemide_cumulative_mg", # 이뇨제 누적 용량 (mg)
        "iv_fluid_total_ml",        # 총 IV 수액량 (mL)
        "iv_fluid_per_kg",          # 체중당 수액량 (mL/kg)
        "vancomycin_exposure_hours",# 반코마이신 노출 시간 (h)
        "ed_los_hours",             # 응급실 체류 시간 (h)
    ]
    for col in skewed_cols:
        if col in df.columns:
            cap = df[col].quantile(0.99)  # 전체 데이터 기준 99th percentile 계산
            df[col] = df[col].clip(upper=cap)
            # cap 초과 값 → cap으로 대체 (아래는 건드리지 않음)

    # ── 2-D. 임상 임계값 기반 버킷 피처 생성 ──────────────────────────────────
    # 연속형 수치를 임상적으로 의미 있는 구간으로 나눠 새 범주형 피처 추가
    #
    # 왜 이 작업이 XGBoost에서 유리한가?
    #   - XGBoost가 스스로 최적 분기점을 찾지만,
    #     임상 지식 기반의 구간(예: vancomycin 48h 초과 = 고위험)을
    #     미리 명시해두면 탐색 공간이 줄고 해석 가능성도 높아짐
    #   - 원본 연속형 변수도 그대로 유지 → 모델이 둘 중 더 유용한 것 선택
    #   - 전처리 후 자동으로 FEATURE_COLS에 포함됨 (_cat 접미사 감지)

    if "vancomycin_exposure_hours" in df.columns:
        df["vanco_duration_cat"] = pd.cut(
            df["vancomycin_exposure_hours"],
            bins=[-1, 0, 24, 48, 9999],  # 경계값 정의
            labels=[0, 1, 2, 3]          # 0=미처방, 1=24h이내, 2=24~48h, 3=48h초과
        ).astype(float)
        # float으로 변환: XGBoost가 category 타입보다 float을 더 안정적으로 처리

    if "furosemide_cumulative_mg" in df.columns:
        df["furosemide_dose_cat"] = pd.cut(
            df["furosemide_cumulative_mg"],
            bins=[-1, 0, 80, 200, 9999], # 임상적 기준: 0/저용량(80mg)/중용량(200mg)/고용량
            labels=[0, 1, 2, 3]          # 0=미처방, 1=저, 2=중, 3=고
        ).astype(float)

    # ── 2-E. 타깃 누수 위험 컬럼 제거 ───────────────────────────────────────────
    # 이 컬럼들은 타깃(AKI 발생) 정보를 직접 담고 있어
    # 피처로 사용하면 예측 시점에 알 수 없는 정보가 모델에 들어가게 됨
    drop_cols = [
        "hours_to_aki",  # AKI 발생까지의 시간 → 이미 AKI가 일어났다는 정보 포함 (극심한 누수)
        "aki_stage",     # AKI 중증도 Stage 1/2/3 → 타깃과 직결된 파생 변수
        "weight_source", # 체중 측정 방법 텍스트 → 예측에 무관한 메타 정보
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    # if c in df.columns: 컬럼이 없어도 오류 발생하지 않도록 안전하게 처리

    return df

# ================================================================================
# 섹션 3. 피처 목록 확정
# ================================================================================

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    전처리 완료 DataFrame에서 실제로 사용할 피처 컬럼 목록을 반환한다.

    - FEATURE_COLS에 명시된 컬럼 중 DataFrame에 실제로 존재하는 것만 포함
    - 전처리에서 새로 생성된 _cat 버킷 피처(vanco_duration_cat 등)를 자동 추가
    - 타깃·그룹·메타 컬럼은 제외

    Parameters
    ----------
    df : pd.DataFrame
        preprocess() 완료 후의 DataFrame

    Returns
    -------
    list[str]
        최종 피처 컬럼명 리스트
    """
    # 전처리에서 새로 생성된 _cat 피처를 자동으로 감지
    # (FEATURE_COLS에는 없지만 DataFrame에 있는 _cat 컬럼)
    extra_cat = [
        c for c in df.columns
        if c.endswith("_cat") and c not in FEATURE_COLS
    ]

    # FEATURE_COLS + 버킷 피처를 합친 후:
    #   1. 실제 DataFrame에 존재하는 컬럼만 남김 (SQL에 없는 컬럼 안전 처리)
    #   2. 타깃·그룹·메타 컬럼 제외
    exclude = {TARGET_COL, GROUP_COL, "stay_id", "is_pseudo_cutoff"}
    cols = [
        c for c in FEATURE_COLS + extra_cat
        if c in df.columns and c not in exclude
    ]

    return cols

# ================================================================================
# 섹션 4. scale_pos_weight 계산
# ================================================================================

def calc_spw(y: pd.Series) -> float:
    """
    클래스 불균형 보정을 위한 scale_pos_weight를 계산한다.

    XGBoost에서 scale_pos_weight는 양성(AKI) 샘플의 손실 기여도를 확대함.
    SMOTE처럼 데이터를 합성·복제하는 게 아니라 손실 함수 내부 가중치만 조정하므로
    데이터 변형 없이 불균형 대응 가능.

    Parameters
    ----------
    y : pd.Series
        Train set의 타깃 레이블 (0/1)
        반드시 Train set 기준으로만 계산해야 함 (Test 비율 개입 금지)

    Returns
    -------
    float
        scale_pos_weight 값 = 음성 수 / 양성 수
        예) 음성 7000, 양성 3000 → 2.333
    """
    neg = (y == 0).sum()  # Non-AKI 환자 수
    pos = (y == 1).sum()  # AKI 환자 수
    spw = neg / pos       # 비율 계산

    print(
        f"[spw] 음성(Non-AKI) {neg:,}명  |  "
        f"양성(AKI) {pos:,}명  |  "
        f"scale_pos_weight = {spw:.3f}"
    )
    return spw

# ================================================================================
# 섹션 5. Optuna 목표 함수 (CV AUPRC 최대화)
# ================================================================================

def make_objective(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    spw: float
):
    """
    Optuna가 각 Trial마다 호출할 목표 함수를 생성해 반환한다.
    클로저(closure) 패턴: X, y, groups, spw를 내부에서 참조.

    Parameters
    ----------
    X      : 피처 DataFrame (Train set 전체)
    y      : 타깃 Series (Train set 전체)
    groups : subject_id Series (CV 분할 기준)
    spw    : scale_pos_weight 값

    Returns
    -------
    Callable
        Optuna Trial 객체를 받아 float(CV AUPRC)을 반환하는 함수
    """
    # StratifiedGroupKFold: 환자 단위 분리 + 양성 비율 균등화 동시 충족
    # - 일반 KFold:           랜덤 분할 → 같은 환자가 Train/Val 양쪽에 들어갈 수 있음
    # - StratifiedKFold:      양성 비율은 맞추지만 환자 단위 분리 불가
    # - StratifiedGroupKFold: 두 조건 동시 충족 → 이 프로젝트에서 유일한 정답
    cv = StratifiedGroupKFold(n_splits=N_FOLDS)

    def objective(trial: optuna.Trial) -> float:
        """
        하이퍼파라미터 하나의 조합을 받아 5-fold CV 평균 AUPRC를 반환.
        Optuna는 이 값을 최대화하는 방향으로 다음 파라미터를 탐색함.
        """
        params = {
            # ── 고정 파라미터 (탐색하지 않음) ────────────────────────────────
            "objective":        "binary:logistic",
            # binary:logistic: 이진 분류, sigmoid 출력 → 0~1 확률 반환

            "eval_metric":      "aucpr",
            # aucpr: AUPRC(Precision-Recall AUC) 기준으로 early stopping 판단
            # AUROC 대신 AUPRC를 쓰는 이유:
            #   AUROC는 불균형 데이터에서 낙관적으로 나옴
            #   AKI 양성이 30%라도 AUROC는 높게 나올 수 있음
            #   AUPRC는 양성 예측 정밀도를 함께 보므로 더 엄격한 기준

            "tree_method":      "hist",
            # hist: 히스토그램 기반 고속 학습 알고리즘 (GPU 없어도 빠름)
            # exact 방법보다 메모리 효율적이고 대용량 데이터에 적합

            "scale_pos_weight": spw,
            # 양성(AKI) 샘플의 손실 가중치 → 불균형 보정

            "random_state":     42,
            # 재현성 보장: 같은 파라미터로 실행하면 항상 같은 결과

            "n_estimators":     1000,
            # 최대 트리 수. early_stopping이 실제 최적 개수를 자동 결정함.
            # 충분히 크게 잡아야 early stopping이 효과를 발휘함

            "verbosity":        0,
            # 0: 학습 로그 출력 없음 (Optuna 진행 바와 겹치지 않게)

            # ── Optuna가 탐색하는 파라미터 ────────────────────────────────────

            "max_depth": trial.suggest_int("max_depth", 3, 7),
            # 트리 최대 깊이.
            # 작을수록: 단순한 모델 (언더피팅↑, 과적합↓)
            # 클수록:   복잡한 패턴 학습 가능하나 과적합 위험 증가
            # AKI처럼 복잡한 임상 데이터는 5~6이 적절한 경우 많음

            "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
            # 리프 노드를 만들기 위한 최소 샘플 가중치 합.
            # 클수록: 분기를 덜 함 → 과적합 방지
            # 작을수록: 소수 케이스도 분기 → 과적합 위험
            # 데이터가 수천 행이면 10~30이 안정적

            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            # 각 트리 학습 시 사용할 샘플 비율.
            # 0.8이면 매 트리마다 전체 데이터의 80%를 랜덤 선택
            # → 트리 간 다양성 확보, 과적합 방지 (랜덤 포레스트 아이디어)

            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            # 각 트리 학습 시 사용할 피처 비율.
            # 0.7이면 전체 피처 중 70%를 랜덤 선택 (트리마다 다른 피처 조합)
            # → 특정 강한 피처에 모든 트리가 의존하는 것 방지

            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            # 학습률(eta). 각 트리의 예측 기여도를 얼마나 반영할지.
            # 작을수록: 천천히 학습, 더 많은 트리 필요하지만 안정적
            # log=True: 로그 스케일로 탐색 (0.005~0.1 구간을 균등하게 탐색)
            # early_stopping과 함께 쓰면 0.01~0.05도 실용적

            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            # L1 정규화 (Lasso 효과).
            # 피처의 가중치를 0으로 만드는 경향 → 불필요한 피처 자동 제거
            # 피처 수가 많을 때 희소성(sparsity) 유도에 유리

            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            # L2 정규화 (Ridge 효과).
            # 가중치 크기를 전체적으로 줄임 → 과적합 방지
            # XGBoost 기본값은 1, 일반적으로 reg_alpha보다 더 영향력 큼

            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            # 분기(split)를 수행하기 위한 최소 손실 감소량.
            # 0이면 손실이 조금이라도 줄면 분기 허용
            # 클수록 분기 조건이 더 까다로워짐 → 보수적인 트리 구조
        }

        # ── 5-fold CV 수행 ────────────────────────────────────────────────────
        scores = []
        for tr_idx, val_idx in cv.split(X, y, groups=groups):
            # cv.split에 groups=groups 전달 → subject_id 단위로 분할
            # 같은 subject_id의 모든 stay가 같은 fold에 배정됨

            X_tr,  X_val  = X.iloc[tr_idx],  X.iloc[val_idx]
            y_tr,  y_val  = y.iloc[tr_idx],  y.iloc[val_idx]

            model = xgb.XGBClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                # eval_set: 매 라운드마다 Val AUPRC를 계산해 early stopping 기준으로 사용

                early_stopping_rounds=50,
                # Val AUPRC가 50라운드 연속으로 개선되지 않으면 학습 중단
                # → n_estimators=1000을 다 쓰지 않아도 됨, 과적합 방지

                verbose=False,
                # Optuna Trial마다 학습 로그가 출력되면 너무 많아지므로 억제
            )

            # predict_proba(X_val): (n_samples, 2) 배열 반환
            # [:, 1]: 양성(AKI) 클래스의 확률만 추출
            prob = model.predict_proba(X_val)[:, 1]

            # AUPRC 계산: Precision-Recall 곡선 아래 면적
            # average_precision_score = AUPRC의 근사값
            scores.append(average_precision_score(y_val, prob))

        # 5개 fold의 평균 AUPRC 반환 → Optuna가 이 값을 최대화
        return float(np.mean(scores))

    return objective

# ================================================================================
# 섹션 6. 최종 모델 학습
# ================================================================================

def train_final(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val:   pd.DataFrame, y_val:   pd.Series,
    best_params: dict,
    spw: float,
) -> xgb.XGBClassifier:
    """
    Optuna 최적 파라미터로 전체 Train set에서 최종 모델을 학습한다.

    Parameters
    ----------
    X_train, y_train : 학습 데이터
    X_val,   y_val   : early stopping 기준 Validation 데이터
    best_params      : Optuna가 탐색한 최적 하이퍼파라미터 딕셔너리
    spw              : scale_pos_weight 값

    Returns
    -------
    xgb.XGBClassifier
        학습 완료된 모델 (model.best_iteration에 실제 사용된 트리 수 저장됨)
    """
    params = {
        "objective":        "binary:logistic",
        "eval_metric":      "aucpr",
        "tree_method":      "hist",
        "scale_pos_weight": spw,
        "random_state":     42,

        "n_estimators":     2000,
        # Optuna CV에서는 1000이었지만, 최종 학습은 데이터가 더 많으므로 2000으로 확대
        # early_stopping_rounds=50이 적절한 시점에서 자동으로 멈춤

        **best_params,
        # Optuna 탐색 결과 최적 파라미터를 덮어씌움
        # 예) {"max_depth": 5, "learning_rate": 0.03, ...}
    }

    model = xgb.XGBClassifier(**params, verbosity=0)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        # Val set AUPRC를 매 라운드마다 계산하여 early stopping 기준으로 활용

        early_stopping_rounds=50,
        # Val AUPRC가 50라운드 동안 개선되지 않으면 학습 중단
        # → model.best_iteration에 실제 최적 트리 수가 저장됨

        verbose=100,
        # 100 라운드마다 현재 Val AUPRC 출력
        # 예) [100] validation_0-aucpr:0.62347
    )

    print(f"[train] 최적 트리 수(best_iteration): {model.best_iteration}")
    # 예) n_estimators=2000 설정해도 실제로는 437번째 트리에서 최적이었다면
    # model.best_iteration = 437
    # predict_proba는 자동으로 437번째 트리까지만 사용

    return model

# ================================================================================
# 섹션 7. 평가
# ================================================================================

def evaluate(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict:
    """
    Test set에서 최종 모델 성능을 측정하고 출력한다.

    Parameters
    ----------
    model     : 학습 완료된 XGBClassifier
    X_test    : Test set 피처
    y_test    : Test set 타깃
    threshold : 이진 분류 임계값 (find_optimal_threshold에서 결정된 값)

    Returns
    -------
    dict
        {"auroc": float, "auprc": float, "prob": ndarray}
    """
    # predict_proba 반환값: shape (n_samples, 2)
    # [:, 0]: Non-AKI 확률, [:, 1]: AKI 확률
    prob = model.predict_proba(X_test)[:, 1]

    # threshold 적용: 확률 >= threshold면 AKI(1), 미만이면 Non-AKI(0)
    pred = (prob >= threshold).astype(int)

    # AUROC: ROC 곡선 아래 면적. 0.5=무작위, 1.0=완벽한 분류
    auroc = roc_auc_score(y_test, prob)

    # AUPRC: PR 곡선 아래 면적. 불균형 데이터에서 AUROC보다 더 엄격한 기준
    # 기준선 = 양성 발생률(예: AKI 30%이면 기준선 = 0.30)
    auprc = average_precision_score(y_test, prob)

    print(f"\n{'='*55}")
    print(f"  AUROC  : {auroc:.4f}")
    print(f"  AUPRC  : {auprc:.4f}")
    print(f"  임계값 : {threshold:.2f}")
    print()

    # classification_report: Precision / Recall / F1 / Support를 클래스별 출력
    # Precision(PPV): 양성 예측 중 실제 양성 비율 → 알람 정확도
    # Recall(Sensitivity): 실제 양성 중 맞게 예측한 비율 → 놓치는 환자 비율
    print(classification_report(
        y_test, pred,
        target_names=["Non-AKI(0)", "AKI(1)"],
        digits=3
    ))

    # Confusion matrix: [[TN FP], [FN TP]]
    # TN: 정상 → 정상 예측 (올바름)
    # FP: 정상 → AKI 예측 (거짓 알람)
    # FN: AKI → 정상 예측 (놓침, 임상적으로 가장 위험)
    # TP: AKI → AKI 예측 (올바름)
    cm = confusion_matrix(y_test, pred)
    print(f"  Confusion Matrix:\n  TN={cm[0,0]:,}  FP={cm[0,1]:,}\n  FN={cm[1,0]:,}  TP={cm[1,1]:,}")
    print(f"{'='*55}\n")

    return {"auroc": auroc, "auprc": auprc, "prob": prob}


# ================================================================================
# 섹션 8. 임계값 최적화
# ================================================================================

def find_optimal_threshold(
    y_true: pd.Series,
    prob:   np.ndarray,
    strategy: str = "f1",
) -> float:
    """
    이진 분류 임계값(threshold)을 Validation set 기준으로 최적화한다.

    기본 임계값 0.5를 쓰지 않는 이유:
      - 0.5는 양성:음성 = 1:1인 균형 데이터를 가정
      - AKI처럼 불균형 데이터에서 모델은 양성 확률을 전반적으로 낮게 예측
      - 0.5로 자르면 AKI 환자를 많이 놓침 (실제 최적값은 0.3~0.4인 경우 흔함)

    Parameters
    ----------
    y_true   : Validation set 실제 레이블
    prob     : Validation set AKI 예측 확률 (0~1)
    strategy : "f1"          → F1 점수 최대화 (기본값)
               "sensitivity" → Sensitivity >= 80% 조건 하에 PPV 최대화

    Returns
    -------
    float
        최적 임계값 (0.10 ~ 0.90 범위)
    """
    from sklearn.metrics import f1_score, recall_score

    # 0.1부터 0.9까지 0.01 간격으로 81개 임계값 후보 생성
    thresholds = np.linspace(0.1, 0.9, 81)

    if strategy == "f1":
        # F1 = 2 × Precision × Recall / (Precision + Recall)
        # Precision과 Recall의 조화평균 → 둘 다 균형 있게 높은 임계값 선택
        scores = [
            f1_score(y_true, prob >= t, zero_division=0)
            for t in thresholds
        ]

    else:  # strategy == "sensitivity"
        # 임상 적용 시나리오:
        # "AKI 환자의 최소 80%는 반드시 잡아야 한다 (Recall >= 0.80)"
        # "그 조건 안에서 알람 정확도(PPV)를 최대한 높인다"
        # → 거짓 알람을 줄이면서도 중요한 케이스를 놓치지 않는 균형점
        scores = [
            (recall_score(y_true, prob >= t, zero_division=0) >= 0.80)
            * (y_true[prob >= t].mean() if (prob >= t).any() else 0.0)
            # recall >= 0.80 조건 충족 시: PPV(양성 예측 정확도) 반환
            # 조건 불충족 시: 0 반환 → 해당 임계값 후보 탈락
            for t in thresholds
        ]

    best_t = float(thresholds[np.argmax(scores)])
    print(
        f"[threshold] 전략={strategy}  "
        f"최적 임계값={best_t:.2f}  "
        f"점수={max(scores):.4f}"
    )
    return best_t


# ================================================================================
# 섹션 9. 시각화
# ================================================================================

def plot_results(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    prob:   np.ndarray,
    feature_cols: list[str],
) -> None:
    """
    Test set 성능 차트 4종을 하나의 이미지로 저장한다.

    저장 위치: OUTPUT_DIR / "performance_plots.png"
    차트 구성:
      [상단 좌] ROC Curve        — FPR vs TPR, AUROC 포함
      [상단 우] PR Curve         — Recall vs Precision, AUPRC 포함
      [하단 좌] Calibration Curve — 예측 확률 vs 실제 발생률
      [하단 우] Feature Importance (gain) — 상위 20개 피처
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── 9-A. ROC Curve ────────────────────────────────────────────────────────
    # x축: False Positive Rate (1-특이도), y축: True Positive Rate (민감도)
    # 대각선(y=x)은 무작위 분류기 기준선
    # 곡선이 좌상단에 가까울수록 좋은 모델
    RocCurveDisplay.from_predictions(y_test, prob, ax=axes[0, 0])
    axes[0, 0].set_title("ROC Curve (AUROC)")
    axes[0, 0].plot([0, 1], [0, 1], "k--", alpha=0.3)  # 기준선 추가

    # ── 9-B. Precision-Recall Curve ───────────────────────────────────────────
    # x축: Recall (민감도), y축: Precision (양성 예측도)
    # 기준선 = 양성 발생률 (예: AKI 30%이면 y=0.30 수평선)
    # 불균형 데이터에서 ROC보다 더 엄격한 성능 평가 기준
    PrecisionRecallDisplay.from_predictions(y_test, prob, ax=axes[0, 1])
    axes[0, 1].set_title("Precision-Recall Curve (AUPRC)")

    # ── 9-C. Calibration Curve ────────────────────────────────────────────────
    # 모델이 "AKI 확률 0.7"이라고 예측했을 때 실제로 70%가 AKI인지 확인
    # 대각선(y=x)에 가까울수록 잘 보정된(calibrated) 모델
    # 위로 치우치면: 과신(overconfident), 아래로 치우치면: 과소신(underconfident)
    frac_pos, mean_pred = calibration_curve(y_test, prob, n_bins=10)
    axes[1, 0].plot(mean_pred, frac_pos, "s-", label="XGBoost", color="royalblue")
    axes[1, 0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="완벽한 보정")
    axes[1, 0].set_xlabel("예측 확률 (mean predicted probability)")
    axes[1, 0].set_ylabel("실제 양성 비율 (fraction of positives)")
    axes[1, 0].set_title("Calibration Curve")
    axes[1, 0].legend()

    # ── 9-D. Feature Importance (Gain 기준) ───────────────────────────────────
    # Gain: 해당 피처로 분기할 때 평균적으로 손실이 얼마나 감소하는가
    # 높을수록 그 피처가 예측에 더 중요하게 기여함
    # (참고: 'weight'=분기 횟수, 'cover'=샘플 커버리지도 있지만 Gain이 가장 직관적)
    imp = pd.Series(
        model.get_booster().get_score(importance_type="gain")
    )
    imp = imp.sort_values(ascending=True).tail(20)  # 상위 20개만
    imp.plot(kind="barh", ax=axes[1, 1], color="steelblue")
    axes[1, 1].set_title("Feature Importance — Gain (top 20)")
    axes[1, 1].set_xlabel("Gain")

    plt.suptitle("AKI XGBoost — Test Set Performance", fontsize=14, y=1.01)
    plt.tight_layout()

    save_path = OUTPUT_DIR / "performance_plots.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] 저장 완료 → {save_path}")


def plot_shap(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    feature_cols: list[str],
) -> None:
    """
    SHAP TreeExplainer로 피처별 예측 기여도를 시각화하고 저장한다.

    SHAP(SHapley Additive exPlanations) 개념:
      각 환자-피처 조합에 대해 "이 피처가 해당 환자의 AKI 예측 확률을
      기준값 대비 얼마나 올리거나 내렸는가"를 수치로 계산
      → 양수: AKI 위험 증가에 기여
      → 음수: AKI 위험 감소에 기여

    TreeExplainer vs KernelExplainer:
      TreeExplainer: XGBoost 트리 구조를 직접 분석 → 매우 빠름
      KernelExplainer: 모델 무관, 범용적이지만 수백~수천 배 느림
      → 항상 TreeExplainer 사용

    저장 파일:
      shap_beeswarm.png : 환자×피처 분포 시각화 (해석에 가장 유용)
      shap_bar.png      : 피처별 평균 |SHAP| 랭킹
      shap_values.csv   : 환자별 원본 SHAP 값 (개별 케이스 분석용)
    """
    print("[shap] SHAP 값 계산 중... (환자 수에 따라 1~5분 소요)")

    # TreeExplainer 초기화: XGBoost 모델의 트리 구조를 분석
    explainer = shap.TreeExplainer(model)

    # SHAP 값 계산: shape (n_patients, n_features)
    # SHAP이 느리면 X_test.sample(1000, random_state=42)로 샘플링 후 계산
    # 1000명 샘플이면 전체 패턴을 충분히 반영
    shap_values = explainer(X_test)

    # ── Beeswarm Plot ────────────────────────────────────────────────────────
    # 각 점 = 환자 1명
    # x축 = SHAP 값 (양수 → AKI 위험 증가, 음수 → 위험 감소)
    # 색상 = 피처값 크기 (빨강 = 높음, 파랑 = 낮음)
    # → "vancomycin_exposure_hours가 높을 때(빨강) 오른쪽(AKI 위험 증가)에 분포"
    #    같은 임상적 해석을 한눈에 파악 가능
    fig1, _ = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=25, show=False)
    plt.tight_layout()
    fig1.savefig(OUTPUT_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # ── Bar Plot ─────────────────────────────────────────────────────────────
    # 피처별 평균 |SHAP| = 해당 피처의 전반적인 중요도
    # (방향 무관, 크기만 봄)
    # XGBoost 내장 feature importance(gain)보다 더 신뢰할 수 있는 지표
    fig2, _ = plt.subplots(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=25, show=False)
    plt.tight_layout()
    fig2.savefig(OUTPUT_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # ── SHAP 값 CSV 저장 ─────────────────────────────────────────────────────
    # 환자별 × 피처별 SHAP 값 원본 저장
    # 나중에 특정 환자의 예측 근거를 개별 분석할 때 사용
    # 예) 왜 이 환자의 AKI 위험이 높게 나왔는가? → 해당 환자 행 조회
    shap_df = pd.DataFrame(shap_values.values, columns=feature_cols)
    shap_df.to_csv(OUTPUT_DIR / "shap_values.csv", index=False)

    print(
        f"[shap] 저장 완료 →\n"
        f"  {OUTPUT_DIR}/shap_beeswarm.png\n"
        f"  {OUTPUT_DIR}/shap_bar.png\n"
        f"  {OUTPUT_DIR}/shap_values.csv"
    )


# ================================================================================
# 섹션 10. 메인 실행 함수
# ================================================================================

def main(tune: bool = True) -> None:
    """
    전체 파이프라인을 순서대로 실행한다.

    Parameters
    ----------
    tune : True  → Optuna 100회 탐색 (기본, 30분~2시간)
           False → 기본 파라미터 사용 (빠른 테스트용, 5분)
    """

    # ── 10-A. 데이터 로드 및 전처리 ─────────────────────────────────────────
    print("\n" + "="*55)
    print("  [STEP 1] 데이터 로드")
    print("="*55)
    df = load_data(DB_URL)

    print("\n" + "="*55)
    print("  [STEP 2] 전처리")
    print("="*55)
    df = preprocess(df)

    # 피처 목록 확정 (버킷 피처 자동 포함)
    feature_cols = get_feature_cols(df)
    print(f"\n[features] 총 {len(feature_cols)}개 피처 확정")
    print(f"  {feature_cols}\n")

    # X, y, groups 분리
    X      = df[feature_cols]   # 피처 행렬
    y      = df[TARGET_COL]     # 타깃 벡터 (0/1)
    groups = df[GROUP_COL]      # 환자 ID 벡터 (CV 분할 기준)

    # ── 10-B. subject_id 기준 Train / Test 분할 (8:2) ────────────────────────
    print("="*55)
    print("  [STEP 3] Train/Test 분할 (subject_id 기준)")
    print("="*55)

    # 주의: stay_id가 아닌 subject_id 단위로 분할해야 함
    # 한 환자가 여러 번 입원(stay_id 여러 개)한 경우,
    # stay_id 단위로 분할하면 같은 환자의 다른 입원이 Train/Test에 동시에 들어감
    # → 모델이 Test에서 해당 환자 정보를 이미 학습한 상태 = 데이터 누수

    unique_subjects = df[GROUP_COL].unique()          # 고유 환자 ID 목록
    rng = np.random.default_rng(42)                   # seed 고정 → 재현성 보장
    rng.shuffle(unique_subjects)                      # 환자 단위로 무작위 섞기

    split_idx   = int(len(unique_subjects) * 0.80)   # 80% 지점
    train_subs  = set(unique_subjects[:split_idx])   # 환자 80% → Train
    test_subs   = set(unique_subjects[split_idx:])   # 환자 20% → Test

    # 마스크 생성: 각 stay가 Train/Test 중 어디에 속하는지
    train_mask = df[GROUP_COL].isin(train_subs)
    test_mask  = df[GROUP_COL].isin(test_subs)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]
    groups_train     = groups[train_mask]

    print(
        f"[split] Train: {len(X_train):,}행  "
        f"(AKI {y_train.mean():.1%})  |  "
        f"Test: {len(X_test):,}행  "
        f"(AKI {y_test.mean():.1%})"
    )

    # ── 10-C. scale_pos_weight 계산 (Train 기준) ────────────────────────────
    # 반드시 Train set 확정 후에 호출해야 함
    # Test 데이터의 비율이 반영되면 정보 누수
    print("\n" + "="*55)
    print("  [STEP 4] scale_pos_weight 계산")
    print("="*55)
    spw = calc_spw(y_train)

    # ── 10-D. Optuna 하이퍼파라미터 탐색 ────────────────────────────────────
    print("\n" + "="*55)
    print(f"  [STEP 5] 하이퍼파라미터 탐색 (tune={tune})")
    print("="*55)

    if tune:
        print(f"[optuna] {OPTUNA_TRIALS}회 탐색 시작 (목표: AUPRC 최대화)")
        print(f"[optuna] TPE(Bayesian) 알고리즘 사용 — 이전 Trial 결과로 다음 탐색 범위 좁힘")

        study = optuna.create_study(
            direction="maximize",                        # AUPRC 최대화
            sampler=optuna.samplers.TPESampler(seed=42), # 베이지안 최적화 (TPE)
            # TPE: Tree-structured Parzen Estimator
            # 이전 Trial 결과를 바탕으로 "좋은 파라미터 구간"을 추정해 집중 탐색
            # 무작위 Grid Search보다 수렴 속도가 훨씬 빠름
        )
        study.optimize(
            make_objective(X_train, y_train, groups_train, spw),
            n_trials=OPTUNA_TRIALS,   # Trial 횟수 (100회 = 약 30분~2시간)
            show_progress_bar=True,   # tqdm 진행 바 표시
        )

        best_params = study.best_params
        print(f"\n[optuna] 탐색 완료!")
        print(f"[optuna] 최적 AUPRC: {study.best_value:.4f}")
        print(f"[optuna] 최적 파라미터:\n{json.dumps(best_params, indent=2)}")

        # 나중에 재현 실험 시 사용할 수 있도록 파일로 저장
        with open(OUTPUT_DIR / "best_params.json", "w") as f:
            json.dump(best_params, f, indent=2, ensure_ascii=False)

    else:
        # --no-tune 옵션: 빠른 테스트용 기본 파라미터
        # 실제 연구에서는 반드시 Optuna 탐색 후 best_params를 사용할 것
        best_params = {
            "max_depth":        5,
            "min_child_weight": 10,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "learning_rate":    0.05,
            "reg_alpha":        0.1,
            "reg_lambda":       1.0,
            "gamma":            0.0,
        }
        print("[optuna] 건너뜀 — 기본 파라미터 사용 (빠른 테스트 모드)")

    # ── 10-E. 최종 모델용 Validation set 분리 ───────────────────────────────
    # Train 환자 80% 중에서 다시 20%를 Validation으로 분리
    # 이 Val set은 두 가지 용도:
    #   1. train_final()의 early_stopping 기준
    #   2. find_optimal_threshold()의 임계값 탐색 기준
    # Test set은 최종 평가 단계까지 일절 사용하지 않음

    val_subs_arr = np.array(sorted(train_subs))   # 정렬 후 배열로 변환
    rng2 = np.random.default_rng(0)               # 별도 시드 (Train 분할과 독립)
    rng2.shuffle(val_subs_arr)
    val_cut     = int(len(val_subs_arr) * 0.2)   # Train 환자 중 20%
    val_subs_set = set(val_subs_arr[:val_cut])

    # 마스크: val_subs_set에 속하면 Val, 아니면 최종 Train
    val_mask_tr   = df[GROUP_COL].isin(val_subs_set) & train_mask
    tr_mask_final = ~df[GROUP_COL].isin(val_subs_set) & train_mask

    X_tr_f,  y_tr_f  = X[tr_mask_final], y[tr_mask_final]
    X_val_f, y_val_f = X[val_mask_tr],   y[val_mask_tr]

    print(
        f"\n[val split] 최종 Train: {len(X_tr_f):,}행  |  "
        f"Val: {len(X_val_f):,}행"
    )

    # ── 10-F. 최종 모델 학습 ─────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  [STEP 6] 최종 모델 학습")
    print("="*55)
    model = train_final(X_tr_f, y_tr_f, X_val_f, y_val_f, best_params, spw)

    # 모델 저장: .ubj 형식 (XGBoost 네이티브 바이너리)
    # 불러올 때: model = xgb.XGBClassifier(); model.load_model("aki_xgb_model.ubj")
    model_path = OUTPUT_DIR / "aki_xgb_model.ubj"
    model.save_model(str(model_path))
    print(f"[train] 모델 저장 완료 → {model_path}")

    # ── 10-G. 임계값 최적화 (Validation 기준) ────────────────────────────────
    print("\n" + "="*55)
    print("  [STEP 7] 임계값 최적화")
    print("="*55)
    prob_val      = model.predict_proba(X_val_f)[:, 1]
    opt_threshold = find_optimal_threshold(y_val_f, prob_val, strategy="f1")
    # strategy="f1" → F1 점수 최대화 임계값
    # strategy="sensitivity" → Recall>=0.80 조건 하 PPV 최대화

    # ── 10-H. Test set 최종 평가 ─────────────────────────────────────────────
    print("\n" + "="*55)
    print("  [STEP 8] Test Set 최종 평가")
    print("="*55)
    print("[주의] 이 단계는 모든 설계 결정이 끝난 후 단 한 번만 실행")
    result = evaluate(model, X_test, y_test, threshold=opt_threshold)

    # ── 10-I. 시각화 저장 ────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  [STEP 9] 시각화 저장")
    print("="*55)
    plot_results(model, X_test, y_test, result["prob"], feature_cols)
    plot_shap(model, X_test, feature_cols)

    # ── 10-J. 실험 요약 저장 (summary.json) ─────────────────────────────────
    summary = {
        "n_features":       len(feature_cols),        # 사용 피처 수
        "feature_cols":     feature_cols,             # 피처 이름 목록
        "n_train":          int(len(X_train)),        # Train 행 수 (Val 포함)
        "n_test":           int(len(X_test)),         # Test 행 수
        "aki_rate_train":   float(y_train.mean()),    # Train AKI 양성 비율
        "aki_rate_test":    float(y_test.mean()),     # Test AKI 양성 비율
        "scale_pos_weight": float(spw),              # 클래스 불균형 보정 가중치
        "best_iteration":   int(model.best_iteration),# 실제 사용된 트리 수
        "threshold":        float(opt_threshold),     # 적용된 분류 임계값
        "auroc":            float(result["auroc"]),   # Test AUROC
        "auprc":            float(result["auprc"]),   # Test AUPRC
        "best_params":      best_params,              # 최적 하이퍼파라미터 전체
    }
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── 최종 결과 출력 ────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  완료! 모든 결과물이 저장되었습니다.")
    print("="*55)
    print(f"  AUROC  : {result['auroc']:.4f}")
    print(f"  AUPRC  : {result['auprc']:.4f}")
    print(f"  임계값 : {opt_threshold:.2f}")
    print(f"  저장위치: {OUTPUT_DIR.resolve()}/")
    print(f"    ├── aki_xgb_model.ubj      ← 모델 파일")
    print(f"    ├── best_params.json       ← 최적 파라미터")
    print(f"    ├── summary.json           ← 실험 전체 요약")
    print(f"    ├── performance_plots.png  ← ROC/PR/Calibration/FI")
    print(f"    ├── shap_beeswarm.png      ← SHAP 분포 (해석)")
    print(f"    ├── shap_bar.png           ← SHAP 랭킹")
    print(f"    └── shap_values.csv        ← 환자별 SHAP 원본")
    print("="*55 + "\n")

# ================================================================================
# 진입점
# ================================================================================

if __name__ == "__main__":
    # argparse: 커맨드라인 인자 파싱
    parser = argparse.ArgumentParser(
        description="AKI 예측 XGBoost 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python train_drug_xgboost.py            # Optuna 100회 탐색 포함 전체 실행
  python train_drug_xgboost.py --no-tune  # 기본 파라미터로 빠른 테스트
        """,
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",        # 플래그 인자: 값 없이 --no-tune만 쓰면 True
        help="Optuna 탐색 건너뛰고 기본 파라미터 사용 (빠른 테스트용)",
    )
    args = parser.parse_args()

    # tune=True(기본) 또는 tune=False(--no-tune 입력 시)로 main 호출
    main(tune=not args.no_tune)