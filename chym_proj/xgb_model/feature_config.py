"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
xgb_model/feature_config.py  —  피처 목록 & 전처리 설정 중앙 관리
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

역할:
  train.py (학습)과 inference.py (추론) 양쪽에서 import하는 단일 진실 소스.
  피처 추가·제거는 이 파일만 수정하면 학습과 추론 모두에 반영된다.

설계 원칙:
  ① 피처를 화면(SCR)·트랙(A~D)별로 그룹화해 해석·디버깅을 쉽게 한다.
  ② 이상치 클리핑 범위는 임상 전문가 검토를 거쳐 설정한다.
  ③ 범주형 변수의 인코딩 매핑은 학습·추론에서 완전히 동일해야 한다.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ─────────────────────────────────────────────────────────────────────────────
# 타겟 및 제외 컬럼
# ─────────────────────────────────────────────────────────────────────────────

TARGET = "aki_label"   # 0: 비AKI, 1: AKI 발생

# 식별자·레이블·누수 위험 컬럼 — 피처에서 반드시 제외
EXCLUDE_COLS = [
    "stay_id", "subject_id", "hadm_id",
    "aki_label", "aki_stage", "aki_onset_time",
    "prediction_cutoff", "effective_cutoff",
    "hours_to_aki",           # 타겟 관련 직접 누수
    "is_pseudo_cutoff",       # 메타 정보
    "icu_intime", "icu_outtime",
    "hospital_expire_flag",   # 사망 여부 — 누수 아니지만 임상 해석 복잡
    "competed_with_death",    # 학습 전 이미 필터링됨
    "hours_to_death",
    "death_time",
]


# ─────────────────────────────────────────────────────────────────────────────
# Track A  —  혈액검사 (SCR-04 검사 결과 탭)
# ─────────────────────────────────────────────────────────────────────────────
FEAT_LAB = [
    # 크레아티닌: AKI 판정의 핵심. cr_delta ≥ 0.3 → KDIGO Stage 1
    "cr_min", "cr_max", "cr_mean", "cr_delta",
    # BUN: bun_cr_ratio > 20 → 신전성 AKI, < 10 → 신성 AKI
    "bun_max", "bun_mean", "bun_cr_ratio",
    # 전해질·산염기
    "potassium_max", "potassium_mean",
    "bicarbonate_min", "bicarbonate_mean",
    # 헤모글로빈: < 7 → 신장 산소 공급 부족
    "hemoglobin_min", "hemoglobin_mean",
    # 젖산: > 2.0 → 조직 저산소 → 허혈성 AKI
    "lactate_max", "lactate_mean",
    # eGFR CKD-EPI 2021 파생 (SCR-04·06 표시, SCR-06 +20점 기준)
    "egfr_ckdepi",
]

# Track A  —  활력징후
FEAT_VITAL = [
    "hr_min", "hr_max", "hr_mean",
    "rr_max", "rr_mean",
    "vital_map_min", "vital_map_mean",
    "sbp_min", "sbp_mean",
    "temp_min", "temp_mean",
    "spo2_min", "spo2_mean",
    "vital_shock_index",
]

# Track A  —  소변량·수액 균형
FEAT_URINE = [
    "urine_6h", "urine_24h", "urine_total", "urine_zero_ratio",
    "fluid_24h", "fluid_total",
    "fluid_balance_24h", "fluid_balance_total",
]

# Track A  —  결측 지시변수
# "측정 안 함" 자체가 임상 정보 (젖산은 쇼크 의심 시에만 측정)
FEAT_MISSING_A = [
    "cr_missing", "lactate_missing",
    "urine_missing", "map_missing",
]


# ─────────────────────────────────────────────────────────────────────────────
# Track B  —  허혈성 피처 (SCR-05 카디오 필터)
# ─────────────────────────────────────────────────────────────────────────────
FEAT_ISCHEMIC = [
    "isch_map_mean", "isch_map_min",
    # map_below65_hours: SCR-05 좌측 카드, SCR-06 허혈 +15점 기준
    "map_below65_hours", "flag_ischemia_over120min",
    "isch_shock_index",
    # vasopressor_flag: SCR-05 IRI 배너 경고 트리거
    "vasopressor_flag",
    "isch_lactate_max", "isch_hemo_min",
]

FEAT_MISSING_B = [
    "shock_index_missing", "isch_lactate_missing", "hemoglobin_missing",
]


# ─────────────────────────────────────────────────────────────────────────────
# Track C  —  신독성 약물 (SCR-03 처방 약물 관리)
# ─────────────────────────────────────────────────────────────────────────────
FEAT_DRUG = [
    # 단일 약물 노출 여부
    "vancomycin_rx", "vancomycin_exposure_hours",
    "piptazo_rx", "aminoglycoside_rx", "amphotericin_b_rx", "carbapenem_rx",
    "ketorolac_rx", "nsaid_any_rx",
    "ace_inhibitor_rx", "arb_rx", "acei_arb_any_rx",
    "furosemide_rx", "furosemide_cumulative_mg",
    "tacrolimus_rx", "cyclosporine_rx",
    "metformin_rx", "ppi_rx",
    # 약물 조합 위험도 (SCR-03 팝업 경고 근거)
    "vanco_piptazo_combo",    # AKI 3.7배↑
    "vanco_aminogly_combo",   # 신독성 상가
    "vanco_carbapenem_combo",
    "nsaid_acei_combo",
    "triple_whammy",          # NSAIDs+ACEi/ARB+이뇨제
    "diuretic_overload_flag",
    # 종합 부담 점수 (SCR-03 AI 배너)
    "nephrotoxic_burden_score",
    "drug_risk_score",
]


# ─────────────────────────────────────────────────────────────────────────────
# Track D  —  방사선 NLP (별도 임포트 후 사용)
# ─────────────────────────────────────────────────────────────────────────────
FEAT_NLP = [
    "kw_hydronephrosis", "kw_aki_mention",
    "kw_edema", "kw_ascites",
]


# ─────────────────────────────────────────────────────────────────────────────
# SCR-06  —  규칙 기반 점수 (XGBoost 입력 피처로도 사용)
# ─────────────────────────────────────────────────────────────────────────────
FEAT_RULE = [
    # 개별 기여 점수 (flag × weight)
    "score_cr", "score_bun", "score_egfr", "score_ischemia", "score_map",
    # 규칙 총점 (SCR-06 대형 숫자의 기반값)
    "rule_based_score",
    # 총 신독성 부담 지수
    "total_nephrotoxic_burden",
]


# ─────────────────────────────────────────────────────────────────────────────
# 인구통계
# ─────────────────────────────────────────────────────────────────────────────
FEAT_DEMO = [
    "age",
    "gender",           # 범주형 → 인코딩 필요
    "first_careunit",   # 범주형 → 인코딩 필요
    "icu_los_hours",
]


# ─────────────────────────────────────────────────────────────────────────────
# 전체 피처 통합 (학습·추론 공통)
# ─────────────────────────────────────────────────────────────────────────────
ALL_FEATURES: list[str] = (
    FEAT_LAB
    + FEAT_VITAL
    + FEAT_URINE
    + FEAT_MISSING_A
    + FEAT_ISCHEMIC
    + FEAT_MISSING_B
    + FEAT_DRUG
    + FEAT_RULE
    + FEAT_DEMO
)
# NLP 피처 (Track D — 03_features_nlp_(트랙D).sql 실행 후 활성화)
# cdss_master_features에 NLP 컬럼이 추가된 경우 주석 해제
FEAT_NLP_PRESET = [
    # 사전 추출 키워드 (nlp_keyword_features.csv)
    "kw_oliguria", "kw_anuria", "kw_edema", "kw_hydronephrosis",
    "kw_aki_mention", "kw_renal_abnormal", "kw_fluid_overload",
    # 원문 추출 키워드 (radiology_nlp_text.csv → 정규표현식)
    "kw_pulmonary_edema", "kw_pleural_effusion", "kw_ascites",
    "kw_contrast_agent", "kw_rad_hydronephrosis", "kw_renal_calculus",
    "kw_cardiomegaly", "kw_foley_catheter", "kw_rad_aki_mention",
    # 집계·결측
    "nlp_keyword_score", "rad_report_count",
    "nlp_missing", "rad_text_missing",
    # 복합 플래그
    "nlp_direct_renal_flag", "nlp_fluid_burden_flag",
]
# Track D NLP 피처 — 03_features_nlp_(트랙D).sql 실행 후 master_features에 컬럼 추가됨
# cdss_master_features에 NLP 컬럼이 없으면 preprocessing.py의 select_and_order_features()가
# 자동으로 해당 컬럼을 NaN으로 채우므로 SQL 실행 전에도 학습 가능 (NLP 피처 = 전부 NaN)
ALL_FEATURES += FEAT_NLP_PRESET

# 범주형 피처 목록 (LabelEncoder 적용 대상)
FEAT_CAT: list[str] = ["gender", "first_careunit"]


# ─────────────────────────────────────────────────────────────────────────────
# 이상치 클리핑 범위
# 생리적으로 불가능한 값 또는 측정 오류를 제거한다.
# 범위 근거: MIMIC-IV 99th percentile + 임상 전문가 검토
# ─────────────────────────────────────────────────────────────────────────────
CLIP_RULES: dict[str, tuple[float, float]] = {
    # 혈액검사
    "cr_min":                   (0,    30),   # mg/dL
    "cr_max":                   (0,    30),
    "cr_delta":                 (0,    20),
    "bun_max":                  (0,   300),   # mg/dL
    "bun_cr_ratio":             (0,   100),
    "potassium_max":            (1,    10),   # mEq/L
    "bicarbonate_min":          (0,    45),   # mEq/L
    "hemoglobin_min":           (0,    25),   # g/dL
    "lactate_max":              (0,    30),   # mmol/L
    "egfr_ckdepi":              (0,   200),   # mL/min/1.73m²
    # 활력징후
    "vital_map_min":            (20,  200),   # mmHg
    "vital_map_mean":           (20,  200),
    "sbp_min":                  (40,  300),
    "hr_min":                   (20,  250),   # bpm
    "hr_max":                   (20,  250),
    "vital_shock_index":        (0,     5),
    # 허혈성 피처
    "map_below65_hours":        (0,    72),   # 최대 72h
    "isch_shock_index":         (0,     5),
    "isch_lactate_max":         (0,    30),
    # 약물
    "vancomycin_exposure_hours":(0,   720),   # 최대 30일
    "furosemide_cumulative_mg": (0,  5000),   # mg
    # 규칙 점수
    "rule_based_score":         (0,   100),
    "total_nephrotoxic_burden": (0,    20),
    "icu_los_hours":            (0,  1000),
}


# ─────────────────────────────────────────────────────────────────────────────
# 범주형 인코딩 고정 매핑
# 학습·추론에서 완전히 동일한 값을 사용해야 한다.
# label_encoders.pkl이 없을 때의 fallback 딕셔너리.
# ─────────────────────────────────────────────────────────────────────────────
GENDER_MAP: dict[str, int] = {
    "M": 0, "F": 1, "Unknown": -1,
}

CAREUNIT_MAP: dict[str, int] = {
    "Medical Intensive Care Unit (MICU)":                0,
    "Surgical Intensive Care Unit (SICU)":               1,
    "Cardiac Vascular Intensive Care Unit (CVICU)":      2,
    "Medical/Surgical Intensive Care Unit (MICU/SICU)":  3,
    "Coronary Care Unit (CCU)":                          4,
    "Neuro Surgical Intensive Care Unit (Neuro SICU)":   5,
    "Trauma SICU (TSICU)":                               6,
    "Neuro Intermediate":                                7,
    "Unknown":                                          -1,
}

STATIC_ENCODERS: dict[str, dict[str, int]] = {
    "gender":        GENDER_MAP,
    "first_careunit":CAREUNIT_MAP,
}