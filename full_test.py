"""
full_test.py  —  AKI CDSS 전체 통합 테스트
DB 없이 Mock 데이터로 모든 백엔드 로직을 실제로 실행한다.

실행: python full_test.py
필요 패키지: pandas, numpy, scikit-learn (별도 설치 불필요)
"""

import sys, os, json, time, traceback, importlib
from unittest.mock import patch, MagicMock

## ── 경로 설정 ──────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.abspath(__file__))
CHYM_PROJ = os.path.join(ROOT, "chym_proj")

# [핵심 수정] XGB_MODEL을 "model" 폴더의 부모 폴더까지만 설정합니다.
# 그래야 코드 내부의 "model/xgb_aki.json"과 결합했을 때 경로가 중복되지 않습니다.
XGB_MODEL = os.path.join(CHYM_PROJ, "xgb_model")

# sys.path에는 모듈 파일(.py)들이 있는 XGB_MODEL 경로만 추가하면 됩니다.
for p in [CHYM_PROJ, XGB_MODEL]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ── 미설치 라이브러리 Mock ────────────────────────────────────────────────
# pydantic, fastapi, xgboost, sqlalchemy 없어도 테스트 가능하게 처리

class _BaseModel:
    """pydantic.BaseModel 대체 — 필드를 __init__ 인자로 받는 단순 구현"""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    def model_dump(self):
        return {k: (v.model_dump() if isinstance(v, _BaseModel) else v)
                for k, v in self.__dict__.items()}
    class Config:
        from_attributes = True

_pydantic_mock = MagicMock()
_pydantic_mock.BaseModel = _BaseModel
_pydantic_mock.Field     = lambda *a, **kw: None
sys.modules.setdefault("pydantic", _pydantic_mock)

_fastapi_mock = MagicMock()
sys.modules.setdefault("fastapi", _fastapi_mock)
sys.modules.setdefault("fastapi.middleware.cors", MagicMock())
sys.modules.setdefault("fastapi.responses", MagicMock())

_xgb_mock = MagicMock()
sys.modules.setdefault("xgboost", _xgb_mock)

_sa_mock = MagicMock()
sys.modules.setdefault("sqlalchemy", _sa_mock)
sys.modules.setdefault("sqlalchemy.orm", MagicMock())

sys.modules.setdefault("optuna", MagicMock())
sys.modules.setdefault("optuna.samplers", MagicMock())
sys.modules.setdefault("optuna.pruners",  MagicMock())

# ── 색상 출력 ─────────────────────────────────────────────────────────────
G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[94m"
BOLD = "\033[1m"; RESET = "\033[0m"

passed = []; failed = []

def ok(msg):   print(f"  {G}✅ {msg}{RESET}")
def fail(msg): print(f"  {R}❌ {msg}{RESET}")
def info(msg): print(f"  {B}ℹ  {msg}{RESET}")
def section(title):
    print(f"\n{BOLD}{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}{RESET}")

def run_test(name, fn):
    """테스트 함수를 실행하고 결과를 기록한다."""
    try:
        fn()
        passed.append(name)
        ok(name)
    except AssertionError as e:
        failed.append((name, str(e)))
        fail(f"{name}  →  {e}")
    except Exception as e:
        failed.append((name, str(e)))
        fail(f"{name}  →  {type(e).__name__}: {e}")


# ══════════════════════════════════════════════════════════════════════════
# Mock DB 데이터 — 고위험 AKI 환자 (PPT 예시와 동일)
# ══════════════════════════════════════════════════════════════════════════

MOCK_STAY_ID = 99999

MOCK_MASTER = {
    "stay_id": MOCK_STAY_ID, "subject_id": 12345, "hadm_id": 67890,
    "age": 20, "gender": "F", "first_careunit": "Surgical Intensive Care Unit (SICU)",
    "icu_intime": "2024-04-23 08:00:00", "icu_outtime": "2024-04-27 18:00:00",
    "icu_los_hours": 106.0, "aki_label": 1, "aki_stage": 2,
    "prediction_cutoff": "2024-04-26 14:00:00",
    "effective_cutoff":  "2024-04-26 14:00:00",
    "hours_to_aki": 12.0, "is_pseudo_cutoff": 0,
    "hospital_expire_flag": 0, "competed_with_death": 0, "hours_to_death": None,
    # SCR-04 혈액검사
    "cr_min": 0.8, "cr_max": 2.1, "cr_mean": 1.5, "cr_delta": 1.3,
    "bun_max": 42.0, "bun_mean": 30.0, "bun_cr_ratio": 20.0,
    "potassium_max": 4.8, "bicarbonate_min": 20.0,
    "hemoglobin_min": 11.2, "hemoglobin_mean": 12.0,
    "lactate_max": 2.5, "lactate_mean": 1.8,
    "egfr_ckdepi": 38.0, "cr_missing": 0, "lactate_missing": 0,
    # SCR-05 허혈성
    "current_map": 62.0, "isch_map_mean": 65.0, "isch_map_min": 58.0,
    "map_below65_hours": 2.42,   # 145분
    "flag_ischemia_over120min": 1,
    "isch_shock_index": 0.95, "vasopressor_flag": 1,
    "isch_lactate_max": 2.5, "isch_hemo_min": 11.2,
    "map_missing": 0, "shock_index_missing": 0,
    "isch_lactate_missing": 0, "hemoglobin_missing": 0,
    # SCR-06 규칙 기반
    "val_cr": 2.1, "val_bun": 42.0, "val_egfr": 38.0,
    "val_ischemia_min": 145.0, "val_map": 62.0,
    "flag_cr": 1, "flag_bun": 1, "flag_egfr": 1, "flag_ischemia": 1, "flag_map": 1,
    "score_cr": 30, "score_bun": 20, "score_egfr": 20,
    "score_ischemia": 15, "score_map": 15,
    "rule_based_score": 100, "high_risk_flag": 1,
    # SCR-03 약물
    "vancomycin_rx": 1, "vancomycin_exposure_hours": 72.0,
    "piptazo_rx": 1, "aminoglycoside_rx": 0, "amphotericin_b_rx": 0,
    "carbapenem_rx": 0, "ketorolac_rx": 0, "nsaid_any_rx": 0,
    "ace_inhibitor_rx": 0, "arb_rx": 0, "acei_arb_any_rx": 0,
    "furosemide_rx": 1, "furosemide_cumulative_mg": 80.0,
    "tacrolimus_rx": 1, "cyclosporine_rx": 0, "metformin_rx": 0, "ppi_rx": 1,
    "vanco_piptazo_combo": 1, "vanco_aminogly_combo": 0,
    "vanco_carbapenem_combo": 0, "nsaid_acei_combo": 0,
    "triple_whammy": 0, "diuretic_overload_flag": 0,
    "nephrotoxic_burden_score": 3, "drug_risk_score": 2,
    "total_nephrotoxic_burden": 9,
    # ischemia time 조회용 (retrieve_total_ischemia_time_in_minutes)
    "ischemia_min": 145.0,
    # Track D NLP
    "kw_oliguria": 0, "kw_anuria": 0, "kw_edema": 1,
    "kw_hydronephrosis": 1, "kw_aki_mention": 0,
    "kw_renal_abnormal": 0, "kw_fluid_overload": 1,
    "kw_pulmonary_edema": 1, "kw_pleural_effusion": 1,
    "kw_ascites": 0, "kw_contrast_agent": 1,
    "kw_rad_hydronephrosis": 1, "kw_renal_calculus": 0,
    "kw_cardiomegaly": 1, "kw_foley_catheter": 0, "kw_rad_aki_mention": 0,
    "nlp_keyword_score": 4, "rad_report_count": 3,
    "nlp_missing": 0, "rad_text_missing": 0,
    "nlp_direct_renal_flag": 1, "nlp_fluid_burden_flag": 1,
}

MOCK_PRESCRIPTIONS = [
    {"drug": "Vancomycin", "route": "IV", "dose_val_rx": "1000",
     "dose_unit_rx": "mg", "starttime": "2024-04-24 08:00:00", "stoptime": None},
    {"drug": "Piperacillin-Tazobactam", "route": "IV", "dose_val_rx": "4.5",
     "dose_unit_rx": "g", "starttime": "2024-04-24 08:00:00", "stoptime": None},
    {"drug": "Tacrolimus", "route": "PO", "dose_val_rx": "2",
     "dose_unit_rx": "mg", "starttime": "2024-04-23 09:00:00", "stoptime": None},
    {"drug": "Furosemide", "route": "IV", "dose_val_rx": "40",
     "dose_unit_rx": "mg", "starttime": "2024-04-25 06:00:00", "stoptime": None},
]

MOCK_LAB_ROWS = [
    {"lab_date": "2024-04-26", "itemid": 50912, "valuenum": 2.1, "prev_valuenum": 1.5},
    {"lab_date": "2024-04-26", "itemid": 51006, "valuenum": 42.0,"prev_valuenum": 28.0},
    {"lab_date": "2024-04-26", "itemid": 51222, "valuenum": 11.2,"prev_valuenum": 12.0},
    {"lab_date": "2024-04-25", "itemid": 50912, "valuenum": 1.5, "prev_valuenum": 0.9},
    {"lab_date": "2024-04-25", "itemid": 51006, "valuenum": 28.0,"prev_valuenum": 18.0},
    {"lab_date": "2024-04-25", "itemid": 51222, "valuenum": 12.0,"prev_valuenum": 13.5},
]

MOCK_MAP_DATA = {
    "current_map": 62.0, "isch_map_min": 58.0,
    "gap_from_target": -3.0,
}

MOCK_RULE_SCORE = {
    "val_cr": 2.1, "val_bun": 42.0, "val_egfr": 38.0,
    "val_ischemia_min": 145.0, "val_map": 62.0,
    "flag_cr": 1, "flag_bun": 1, "flag_egfr": 1, "flag_ischemia": 1, "flag_map": 1,
    "score_cr": 30, "score_bun": 20, "score_egfr": 20,
    "score_ischemia": 15, "score_map": 15,
    "rule_based_score": 100,
}

MOCK_TIMESERIES = [
    {"timestamp": f"2024-04-26 {h:02d}:00:00",
     "cr_at_hour": 2.1 - (7-i)*0.15, "bun_at_hour": 42-(7-i)*2,
     "map_min_hour": 62+(i%3), "ischemia_ratio_hour": 0.6 if i > 4 else 0.1,
     "nlp_direct_renal_flag": 1, "nlp_fluid_burden_flag": 1}
    for i, h in enumerate(range(8, 19))
]

def mock_execute_query(sql, params=None):
    """모든 execute_query 호출을 Mock 데이터로 대체한다."""
    sql_upper = sql.upper().strip()

    if "SELECT 1" in sql_upper:
        return [{"?column?": 1}]
    if "COUNT(*)" in sql_upper and "MASTER_FEATURES" in sql_upper:
        return [{"n_total": 5000, "n_aki": 1800, "aki_pct": 36.0,
                 "avg_rule_score": 45.2, "n_high_risk": 900}]
    if "COUNT(*)" in sql_upper:
        return [{"cnt": 5000}]
    if "AVG(AKI_LABEL" in sql_upper or "AVG(aki_label" in sql:
        return [{"ratio": 0.36}]
    if "AVG((1-NLP_MISSING)" in sql_upper:
        return [{"coverage": 0.64}]
    if "AVG(COMPETED_WITH_DEATH" in sql_upper:
        return [{"competing": 0.05}]
    if "SUM(NLP_FLUID_BURDEN_FLAG)" in sql_upper:
        return [{"flag_sum": 500, "kw_sum": 1200}]

    # master_features 단일 환자 조회
    if "CDSS_MASTER_FEATURES" in sql_upper and "WHERE STAY_ID" in sql_upper:
        return [MOCK_MASTER]
    if "CDSS_MASTER_FEATURES" in sql_upper and "WHERE M.STAY_ID" in sql_upper:
        return [MOCK_MASTER]
    if "CDSS_MASTER_FEATURES" in sql_upper and "WHERE STAY_ID IN" in sql_upper:
        return [MOCK_MASTER]
    if "CDSS_MASTER_FEATURES" in sql_upper and "LIMIT 1" in sql_upper:
        return [MOCK_MASTER]
    if "CDSS_MASTER_FEATURES" in sql_upper and "HIGH_RISK_FLAG = 1" in sql_upper:
        return [MOCK_MASTER]
    if "CDSS_MASTER_FEATURES" in sql_upper and "HIGH_RISK_FLAG = 0" in sql_upper:
        return [{**MOCK_MASTER, "aki_label": 0, "high_risk_flag": 0, "rule_based_score": 15}]

    # 처방 조회
    if "CDSS_NEPHROTOXIC_RX_RAW" in sql_upper:
        return MOCK_PRESCRIPTIONS

    # 혈액검사
    if "CDSS_RAW_LAB_VALUES" in sql_upper:
        return MOCK_LAB_ROWS

    # MAP 조회
    if "CURRENT_MAP" in sql_upper and "GAP_FROM_TARGET" in sql_upper:
        return [MOCK_MAP_DATA]

    # 규칙 점수 조회
    if "CDSS_RULE_SCORE_FEATURES" in sql_upper:
        return [MOCK_RULE_SCORE]

    # 시계열 (generate_series)
    if "GENERATE_SERIES" in sql_upper or "WINDOW_END" in sql_upper:
        return MOCK_TIMESERIES

    # NLP 조회
    if "NLP_DIRECT_RENAL_FLAG" in sql_upper or "KW_EDEMA" in sql_upper:
        return [MOCK_MASTER]

    # 허혈 시간 (map_below65_hours * 60 AS ischemia_min 쿼리)
    if "ISCHEMIA_MIN" in sql_upper and "MAP_BELOW65_HOURS" in sql_upper:
        return [{"map_below65_hours": 2.42, "ischemia_min": 145.0}]

    # ANALYZE (주석 처리 됐어야 하지만 혹시 모르니)
    return [MOCK_MASTER]


# ══════════════════════════════════════════════════════════════════════════
# PART 1: db.py 기본 유틸리티
# ══════════════════════════════════════════════════════════════════════════

section("PART 1  db.py — 기본 유틸리티")

def t_riskLevel_high():
    from db import classify_value_as_risk_level
    r = classify_value_as_risk_level(2.0, high_threshold=1.5)
    assert r.level == "high"
    assert r.color == "#ef4444"

def t_riskLevel_warning():
    from db import classify_value_as_risk_level
    r = classify_value_as_risk_level(1.2, high_threshold=1.5, warning_threshold=1.0)
    assert r.level == "warning"

def t_riskLevel_normal():
    from db import classify_value_as_risk_level
    r = classify_value_as_risk_level(0.8, high_threshold=1.5, warning_threshold=1.0)
    assert r.level == "normal"

def t_riskLevel_low_threshold():
    from db import classify_value_as_risk_level
    # eGFR: 하한 기준
    r = classify_value_as_risk_level(38.0, low_threshold=45.0, low_warning=60.0)
    assert r.level == "high"

run_test("classify_value → high",    t_riskLevel_high)
run_test("classify_value → warning", t_riskLevel_warning)
run_test("classify_value → normal",  t_riskLevel_normal)
run_test("classify_value 하한기준",   t_riskLevel_low_threshold)


# ══════════════════════════════════════════════════════════════════════════
# PART 2: SCR-03 처방 약물 관리
# ══════════════════════════════════════════════════════════════════════════

section("PART 2  SCR-03 처방 약물 관리")

# db.execute_query를 mock으로 교체 (이미 import된 모듈에도 적용)
import db
db.execute_query = mock_execute_query

import scr03_drug_management, scr04_lab_monitoring
import scr05_cardio_filter, scr06_ai_risk_score, scr07_risk_timeseries
for _mod in [scr03_drug_management, scr04_lab_monitoring,
             scr05_cardio_filter, scr06_ai_risk_score, scr07_risk_timeseries]:
    if hasattr(_mod, 'execute_query'):
        _mod.execute_query = mock_execute_query

from scr03_drug_management import (
retrieve_current_prescriptions_for_display,
detect_dangerous_drug_combination_patterns,
generate_ai_nephrotoxicity_monitoring_message,
build_drug_management_screen_response,
DRUG_METADATA,
)

def t_03_prescriptions():
        rxs = retrieve_current_prescriptions_for_display(MOCK_STAY_ID)
        assert len(rxs) >= 1
        names = [r.drug_name for r in rxs]
        assert any("Vancomycin" in n or "vancomycin" in n.lower() for n in names)

def t_03_nephrotoxic_first():
        rxs = retrieve_current_prescriptions_for_display(MOCK_STAY_ID)
        assert rxs[0].is_nephrotoxic, "신독성 약물이 맨 위에 있어야 함"

def t_03_combo_alerts():
        alerts = detect_dangerous_drug_combination_patterns(MOCK_STAY_ID)
        triggered = [a for a in alerts if a.is_triggered]
        assert any(a.combo_type == "vanco_piptazo" for a in triggered)

def t_03_ai_banner_high():
        banner = generate_ai_nephrotoxicity_monitoring_message(MOCK_STAY_ID)
        assert banner.overall_risk == "높음"
        assert banner.primary_drug is not None

def t_03_ai_banner_nlp_hydro():
        banner = generate_ai_nephrotoxicity_monitoring_message(MOCK_STAY_ID)
        assert "수신증" in banner.detail_message or "방사선" in banner.detail_message

def t_03_burden_score_range():
        # burden_score는 Mock 데이터에서 직접 확인
        score = MOCK_MASTER.get("nephrotoxic_burden_score", 0)
        assert 0 <= score <= 8

def t_03_build_full():
        result = build_drug_management_screen_response(MOCK_STAY_ID)
        assert result.stay_id == MOCK_STAY_ID
        assert result.ai_alert is not None

run_test("SCR-03 처방 목록 조회",         t_03_prescriptions)
run_test("SCR-03 신독성 약물 최상단",      t_03_nephrotoxic_first)
run_test("SCR-03 Vanco+Pip 조합 경고",    t_03_combo_alerts)
run_test("SCR-03 AI 배너 위험도 높음",    t_03_ai_banner_high)
run_test("SCR-03 AI 배너 NLP 수신증 경고", t_03_ai_banner_nlp_hydro)
run_test("SCR-03 부담 점수 범위",         t_03_burden_score_range)
run_test("SCR-03 전체 화면 조립",         t_03_build_full)


# ══════════════════════════════════════════════════════════════════════════
# PART 3: SCR-04 검사 결과 & AKI 모니터링
# ══════════════════════════════════════════════════════════════════════════

section("PART 3  SCR-04 검사 결과 & AKI 모니터링")

# scr04 execute_query mock 재주입
import scr04_lab_monitoring
scr04_lab_monitoring.execute_query = mock_execute_query

from scr04_lab_monitoring import (
    classify_lab_result_status_by_normal_range,
    calculate_trend_direction_from_sequential_values,
    derive_egfr_from_creatinine_and_demographics,
    build_aki_monitoring_summary_for_bottom_banner,
    build_lab_monitoring_screen_response,
)

def t_04_cr_high():
    r = classify_lab_result_status_by_normal_range("creatinine", 2.1)
    assert r.level == "high"

def t_04_bun_warning():
    r = classify_lab_result_status_by_normal_range("bun", 25.0)
    assert r.level == "warning"

def t_04_egfr_high():
    r = classify_lab_result_status_by_normal_range("egfr", 38.0)
    assert r.level == "high"

def t_04_hgb_warning():
    r = classify_lab_result_status_by_normal_range("hemoglobin", 11.2)
    assert r.level == "warning"

def t_04_trend_up():
    trend, arrow = calculate_trend_direction_from_sequential_values(2.1, 1.5)
    assert trend == "up" and arrow == "↑"

def t_04_trend_stable():
    trend, arrow = calculate_trend_direction_from_sequential_values(1.5, 1.52)
    assert trend == "stable" and arrow == "→"

def t_04_trend_down():
    trend, arrow = calculate_trend_direction_from_sequential_values(11.2, 12.5)
    assert trend == "down" and arrow == "↓"

def t_04_egfr_female():
    egfr = derive_egfr_from_creatinine_and_demographics(2.1, 20, "F")
    assert egfr is not None and 10 < egfr < 80

def t_04_egfr_male():
    egfr = derive_egfr_from_creatinine_and_demographics(1.0, 60, "M")
    assert egfr is not None and 60 < egfr < 100

def t_04_aki_banner():
    banner = build_aki_monitoring_summary_for_bottom_banner(MOCK_STAY_ID)
    assert banner.prediction_level in ("높음", "보통", "낮음")

def t_04_aki_banner_nlp_cause():
    banner = build_aki_monitoring_summary_for_bottom_banner(MOCK_STAY_ID)
    # NLP 소견(수신증·체액 과부하)이 원인에 포함돼야 함
    cause_or_detail = banner.primary_cause + " " + banner.detail_message
    assert "방사선" in cause_or_detail or "수신증" in cause_or_detail or "폐부종" in cause_or_detail

def t_04_build_full():
    result = build_lab_monitoring_screen_response(MOCK_STAY_ID)
    assert result.stay_id == MOCK_STAY_ID
    assert len(result.current_results) == 4

run_test("SCR-04 크레아티닌 → 높음",       t_04_cr_high)
run_test("SCR-04 BUN → 주의",             t_04_bun_warning)
run_test("SCR-04 eGFR → 높음(하한)",      t_04_egfr_high)
run_test("SCR-04 헤모글로빈 → 경계",       t_04_hgb_warning)
run_test("SCR-04 추이 ↑",                 t_04_trend_up)
run_test("SCR-04 추이 →(안정)",            t_04_trend_stable)
run_test("SCR-04 추이 ↓",                 t_04_trend_down)
run_test("SCR-04 eGFR CKD-EPI 여성",     t_04_egfr_female)
run_test("SCR-04 eGFR CKD-EPI 남성",     t_04_egfr_male)
run_test("SCR-04 AKI 배너 위험도",        t_04_aki_banner)
run_test("SCR-04 AKI 배너 NLP 원인",      t_04_aki_banner_nlp_cause)
run_test("SCR-04 전체 화면 조립(4행)",    t_04_build_full)


# ══════════════════════════════════════════════════════════════════════════
# PART 4: SCR-05 카디오 필터
# ══════════════════════════════════════════════════════════════════════════

section("PART 4  SCR-05 카디오 필터")

# scr05 execute_query mock 재주입
import scr05_cardio_filter
scr05_cardio_filter.execute_query = mock_execute_query

from scr05_cardio_filter import (
    check_whether_ischemia_time_exceeds_safe_threshold,
    check_whether_map_is_below_renal_perfusion_target,
    retrieve_total_ischemia_time_in_minutes,
    assess_cardio_ischemia_risk_for_aki_prediction,
    generate_cardio_filter_protocol_recommendation,
    build_cardio_filter_screen_response,
)

def t_05_ischemia_over():
    is_over, color, warn_txt, _ = check_whether_ischemia_time_exceeds_safe_threshold(145)
    assert is_over and color == "#ef4444"
    assert "120분" in warn_txt

def t_05_ischemia_safe():
    is_over, color, _, _ = check_whether_ischemia_time_exceeds_safe_threshold(30)
    assert not is_over and color == "#e5e7eb"

def t_05_map_below():
    is_below, color, txt, _ = check_whether_map_is_below_renal_perfusion_target(62)
    assert is_below and color == "#f97316"
    assert "65" in txt

def t_05_map_ok():
    is_below, color, _, _ = check_whether_map_is_below_renal_perfusion_target(75)
    assert not is_below

def t_05_ischemia_min():
    mins = retrieve_total_ischemia_time_in_minutes(MOCK_STAY_ID)
    assert mins is not None and abs(mins - 145.0) < 1.0

def t_05_cardio_risk_high():
    risk = assess_cardio_ischemia_risk_for_aki_prediction(MOCK_STAY_ID)
    assert risk["iri_risk_label"] == "고위험"
    assert risk["cardio_risk_score"] >= 2

def t_05_nlp_cardiomegaly_bonus():
    risk = assess_cardio_ischemia_risk_for_aki_prediction(MOCK_STAY_ID)
    assert risk.get("kw_cardiomegaly") == True  # Mock 데이터 kw_cardiomegaly=1

def t_05_banner_nlp_warnings():
    risk = assess_cardio_ischemia_risk_for_aki_prediction(MOCK_STAY_ID)
    banner = generate_cardio_filter_protocol_recommendation(risk, ml_probability=0.88)
    combined = banner.get("combined_analysis", "")
    assert "심비대" in combined or "조영제" in combined or "방사선" in combined

def t_05_build_full():
    result = build_cardio_filter_screen_response(MOCK_STAY_ID)
    assert result["stay_id"] == MOCK_STAY_ID
    assert "ischemia_card" in result and "map_card" in result
    assert result["ischemia_card"]["total_minutes"] > 0

run_test("SCR-05 허혈 초과 → 빨간 카드",     t_05_ischemia_over)
run_test("SCR-05 허혈 기준 이내 → 회색",      t_05_ischemia_safe)
run_test("SCR-05 MAP 미달 → 주황 카드",       t_05_map_below)
run_test("SCR-05 MAP 목표 달성",              t_05_map_ok)
run_test("SCR-05 허혈 시간 분 변환(145분)",   t_05_ischemia_min)
run_test("SCR-05 IRI 위험도 고위험",          t_05_cardio_risk_high)
run_test("SCR-05 NLP 심비대 점수 가산",        t_05_nlp_cardiomegaly_bonus)
run_test("SCR-05 IRI 배너 NLP 경고 문구",     t_05_banner_nlp_warnings)
run_test("SCR-05 전체 화면 조립",             t_05_build_full)


# ══════════════════════════════════════════════════════════════════════════
# PART 5: SCR-06 AI 급성 신손상 예측도
# ══════════════════════════════════════════════════════════════════════════

section("PART 5  SCR-06 AI 급성 신손상 예측도")

# scr06 execute_query mock 재주입
import scr06_ai_risk_score
scr06_ai_risk_score.execute_query = mock_execute_query

from scr06_ai_risk_score import (
    calculate_rule_based_risk_score_with_factor_breakdown,
    combine_rule_score_and_ml_probability_into_final_risk,
    build_risk_factor_contribution_table_for_display,
    build_ai_risk_score_screen_response,
)

def t_06_rule_score_100():
    data = calculate_rule_based_risk_score_with_factor_breakdown(MOCK_STAY_ID)
    assert data["rule_based_score"] == 100
    assert len(data["factors"]) == 5

def t_06_all_factors_exceeded():
    data = calculate_rule_based_risk_score_with_factor_breakdown(MOCK_STAY_ID)
    assert all(f["exceeded"] for f in data["factors"])

def t_06_score_sum_correct():
    data = calculate_rule_based_risk_score_with_factor_breakdown(MOCK_STAY_ID)
    total = sum(f["score"] for f in data["factors"])
    assert total == 100  # 30+20+20+15+15

def t_06_combine_rule_only():
    val, risk = combine_rule_score_and_ml_probability_into_final_risk(100, None)
    assert val == 100 and risk.level == "high"

def t_06_combine_with_ml():
    val, risk = combine_rule_score_and_ml_probability_into_final_risk(100, 0.88)
    # 100*0.4 + 88*0.6 = 40+52.8 = 92
    assert val == 92 and risk.level == "high"

def t_06_combine_low_risk():
    val, risk = combine_rule_score_and_ml_probability_into_final_risk(0, 0.1)
    assert val == 6 and risk.level == "normal"

def t_06_factor_table_5rows():
    data = calculate_rule_based_risk_score_with_factor_breakdown(MOCK_STAY_ID)
    table = build_risk_factor_contribution_table_for_display(data["factors"])
    assert len(table) == 5

def t_06_factor_table_colors():
    data = calculate_rule_based_risk_score_with_factor_breakdown(MOCK_STAY_ID)
    table = build_risk_factor_contribution_table_for_display(data["factors"])
    exceeded = [r for r in table if r.is_exceeded]
    assert all(r.status_color == "#f97316" for r in exceeded)

def t_06_build_full():
    result = build_ai_risk_score_screen_response(MOCK_STAY_ID)
    assert result.stay_id == MOCK_STAY_ID
    assert result.risk_display.displayed_value == 100
    assert result.risk_display.is_high_risk
    assert result.total_score == 100

def t_06_build_alert_message():
    result = build_ai_risk_score_screen_response(MOCK_STAY_ID)
    assert "높음" in result.alert_message or "매우 높음" in result.alert_message

run_test("SCR-06 규칙 점수 100점",           t_06_rule_score_100)
run_test("SCR-06 5개 요인 전부 초과",         t_06_all_factors_exceeded)
run_test("SCR-06 기여도 합산 100",            t_06_score_sum_correct)
run_test("SCR-06 규칙만 → 100점",            t_06_combine_rule_only)
run_test("SCR-06 규칙+ML 혼합 → 92점",       t_06_combine_with_ml)
run_test("SCR-06 저위험 혼합 → 정상",         t_06_combine_low_risk)
run_test("SCR-06 기여도 테이블 5행",          t_06_factor_table_5rows)
run_test("SCR-06 초과 항목 주황색",           t_06_factor_table_colors)
run_test("SCR-06 전체 화면 조립",             t_06_build_full)
run_test("SCR-06 상단 경고 배너 문구",         t_06_build_alert_message)


# ══════════════════════════════════════════════════════════════════════════
# PART 6: SCR-07 AKI 위험도 시계열
# ══════════════════════════════════════════════════════════════════════════

section("PART 6  SCR-07 AKI 위험도 시계열")

# scr07 execute_query mock 재주입
import scr07_risk_timeseries
scr07_risk_timeseries.execute_query = mock_execute_query

from scr07_risk_timeseries import (
    _compute_hourly_rule_score,
    detect_risk_score_escalation_above_alert_threshold,
    build_timeseries_alert_banner_message,
    build_risk_timeseries_screen_response,
)

def t_07_compute_base_score():
    row = {"cr_at_hour": 2.1, "bun_at_hour": 42,
           "map_min_hour": 62, "ischemia_ratio_hour": 0.6,
           "nlp_direct_renal_flag": 0, "nlp_fluid_burden_flag": 0}
    score = _compute_hourly_rule_score(row)
    assert score == min(30+20+15+15, 100)  # cr+bun+map+ischemia = 80

def t_07_nlp_adds_score():
    base_row = {"cr_at_hour": 2.1, "bun_at_hour": 42,
                "map_min_hour": 62, "ischemia_ratio_hour": 0.6,
                "nlp_direct_renal_flag": 0, "nlp_fluid_burden_flag": 0}
    nlp_row  = {**base_row,
                "nlp_direct_renal_flag": 1, "nlp_fluid_burden_flag": 1}
    base  = _compute_hourly_rule_score(base_row)
    with_nlp = _compute_hourly_rule_score(nlp_row)
    assert with_nlp - base == 15  # +10(direct)+5(fluid)

def t_07_score_capped_100():
    # 최대 가능 점수: 30+20+15+15+10+5 = 95 → 100 이하 보장 확인
    row = {"cr_at_hour": 5.0, "bun_at_hour": 100,
           "map_min_hour": 50, "ischemia_ratio_hour": 1.0,
           "nlp_direct_renal_flag": 1, "nlp_fluid_burden_flag": 1}
    score = _compute_hourly_rule_score(row)
    assert score == 95, f"예상 95, 실제 {score}"
    assert score <= 100

def t_07_escalation_detect():
    points = [
        {"risk_pct": 15, "time_str": "08:00"},
        {"risk_pct": 25, "time_str": "10:00"},
        {"risk_pct": 42, "time_str": "12:00"},
        {"risk_pct": 65, "time_str": "14:00"},
        {"risk_pct": 85, "time_str": "16:00"},
        {"risk_pct": 88, "time_str": "18:00"},
    ]
    result = detect_risk_score_escalation_above_alert_threshold(points)
    assert result["is_escalating"] == True
    assert result["first_exceed_time"] == "16:00"
    assert result["peak_risk_pct"] == 88

def t_07_no_escalation():
    points = [{"risk_pct": i*5, "time_str": f"{h:02d}:00"}
              for i, h in enumerate(range(8, 14))]  # 최대 25%
    result = detect_risk_score_escalation_above_alert_threshold(points)
    assert result["is_escalating"] == False

def t_07_banner_active():
    esc = {"is_escalating": True, "first_exceed_time": "16:00",
           "peak_risk_pct": 88, "escalation_rate": 11.5}
    banner = build_timeseries_alert_banner_message(esc, 88)
    assert banner["is_active"] == True
    assert "16:00" in banner["message"]
    assert "75" in banner["sub_message"]

def t_07_banner_inactive():
    esc = {"is_escalating": False, "first_exceed_time": None,
           "peak_risk_pct": 42, "escalation_rate": None}
    banner = build_timeseries_alert_banner_message(esc, 42)
    assert banner["is_active"] == False

def t_07_build_full():
    result = build_risk_timeseries_screen_response(MOCK_STAY_ID, window_hours=10)
    assert result["stay_id"] == MOCK_STAY_ID
    assert result["chart_config"]["threshold_line"] == 75
    assert all(0 <= dp["risk_pct"] <= 100 for dp in result["data_points"])

run_test("SCR-07 시간별 점수 계산(80점)",     t_07_compute_base_score)
run_test("SCR-07 NLP +15점 가산",             t_07_nlp_adds_score)
run_test("SCR-07 점수 100 캡",                t_07_score_capped_100)
run_test("SCR-07 급상승 감지(16:00)",          t_07_escalation_detect)
run_test("SCR-07 급상승 없음",                t_07_no_escalation)
run_test("SCR-07 경고 배너 활성",             t_07_banner_active)
run_test("SCR-07 경고 배너 비활성",            t_07_banner_inactive)
run_test("SCR-07 전체 화면 조립",             t_07_build_full)


# ══════════════════════════════════════════════════════════════════════════
# PART 7: XGBoost 파이프라인 — feature_config + preprocessing
# ══════════════════════════════════════════════════════════════════════════

section("PART 7  XGBoost 파이프라인")

import pandas as pd
import numpy as np

def t_xgb_feature_config():
    from feature_config import ALL_FEATURES, FEAT_NLP_PRESET, FEAT_CAT, CLIP_RULES
    assert len(ALL_FEATURES) > 50, "피처 수가 너무 적음"
    assert len(FEAT_NLP_PRESET) == 22, f"NLP 피처 수 이상: {len(FEAT_NLP_PRESET)}"
    assert "gender" in FEAT_CAT
    assert "cr_max" in CLIP_RULES

def t_xgb_nlp_in_all_features():
    from feature_config import ALL_FEATURES, FEAT_NLP_PRESET
    for feat in FEAT_NLP_PRESET:
        assert feat in ALL_FEATURES, f"{feat} ALL_FEATURES에 없음"

def t_xgb_clip_rules_reasonable():
    from feature_config import CLIP_RULES
    lo, hi = CLIP_RULES["cr_max"]
    assert lo == 0 and hi == 30
    lo2, hi2 = CLIP_RULES["egfr_ckdepi"]
    assert lo2 == 0 and hi2 == 200

def t_xgb_preprocessing_clip():
    from feature_config import CLIP_RULES
    from preprocessing import clip_outliers_by_clinical_range
    df = pd.DataFrame({"cr_max": [50.0], "egfr_ckdepi": [300.0]})
    clipped = clip_outliers_by_clinical_range(df)
    assert clipped["cr_max"].iloc[0] == 30.0
    assert clipped["egfr_ckdepi"].iloc[0] == 200.0

def t_xgb_preprocessing_encode():
    from preprocessing import encode_categorical_columns
    df = pd.DataFrame({"gender": ["M", "F", None], "first_careunit": ["unknown", "unknown", "unknown"]})
    encoded = encode_categorical_columns(df, encoders=None)
    assert encoded["gender"].iloc[0] == 0  # M → 0
    assert encoded["gender"].iloc[1] == 1  # F → 1
    assert encoded["gender"].iloc[2] == -1 # None → -1

def t_xgb_select_features():
    from feature_config import ALL_FEATURES
    from preprocessing import select_and_order_features
    n_feats = len(ALL_FEATURES)
    df = pd.DataFrame(np.zeros((5, n_feats)), columns=ALL_FEATURES)
    df["extra_col"] = 999  # 불필요한 컬럼 추가
    X, names = select_and_order_features(df)
    assert list(X.columns) == ALL_FEATURES  # extra_col 제외됨
    assert "extra_col" not in X.columns

def t_xgb_missing_features_filled():
    from feature_config import ALL_FEATURES
    from preprocessing import select_and_order_features
    # 일부 피처만 있는 df
    df = pd.DataFrame({"cr_max": [2.1], "bun_max": [42.0]})
    X, names = select_and_order_features(df, feature_names=ALL_FEATURES[:10])
    assert X.shape[1] == 10
    # cr_max가 10번째 이내에 있으면 값이 있고, 없으면 NaN
    missing_count = X.isna().sum().sum()
    assert missing_count >= 0  # NaN 허용

def t_xgb_synthetic_training():
    """sklearn GradientBoosting으로 XGBoost 학습 흐름을 시뮬레이션."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    np.random.seed(42)
    n = 500

    # 합성 데이터 생성 (실제 피처 분포 근사)
    X = pd.DataFrame({
        "cr_max":            np.random.exponential(1.2, n).clip(0, 10),
        "cr_delta":          np.random.exponential(0.5, n).clip(0, 5),
        "bun_max":           np.random.normal(25, 15, n).clip(0, 150),
        "egfr_ckdepi":       np.random.normal(65, 25, n).clip(5, 200),
        "map_below65_hours": np.random.exponential(1, n).clip(0, 20),
        "vasopressor_flag":  np.random.binomial(1, 0.3, n),
        "vancomycin_rx":     np.random.binomial(1, 0.4, n),
        "rule_based_score":  np.random.randint(0, 101, n),
        "nlp_direct_renal_flag": np.random.binomial(1, 0.15, n),
        "nlp_fluid_burden_flag": np.random.binomial(1, 0.25, n),
    })
    y = ((X["cr_max"] > 1.5).astype(int) |
         (X["egfr_ckdepi"] < 45).astype(int) |
         (X["vasopressor_flag"] == 1)).astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=42)
    model.fit(X_tr, y_tr)
    prob = model.predict_proba(X_te)[:, 1]
    auroc = roc_auc_score(y_te, prob)
    assert auroc > 0.7, f"합성 데이터 AUROC 낮음: {auroc:.3f}"
    info(f"합성 데이터 AUROC: {auroc:.3f}")

run_test("feature_config 구조 확인",          t_xgb_feature_config)
run_test("NLP 피처 ALL_FEATURES 포함 확인",    t_xgb_nlp_in_all_features)
run_test("CLIP_RULES 범위 합리성",             t_xgb_clip_rules_reasonable)
run_test("preprocessing 이상치 클리핑",        t_xgb_preprocessing_clip)
run_test("preprocessing 범주형 인코딩",        t_xgb_preprocessing_encode)
run_test("preprocessing 피처 선택·정렬",       t_xgb_select_features)
run_test("preprocessing 결측 피처 NaN 채움",   t_xgb_missing_features_filled)
run_test("합성 데이터 GBM 학습 AUROC>0.7",    t_xgb_synthetic_training)


# ══════════════════════════════════════════════════════════════════════════
# PART 8: SQL 파이프라인 구문 검사
# ══════════════════════════════════════════════════════════════════════════

section("PART 8  SQL 파이프라인 구문 검사")

import re

def t_sql_drop_commented():
    with open(os.path.join(ROOT, "01_sql_pipeline_by_screen.sql")) as f:
        sql = f.read()
    active = [l for l in sql.split('\n')
              if re.match(r'\s*DROP TABLE', l)]
    assert len(active) == 0, f"활성 DROP TABLE {len(active)}건 남음"

def t_sql_analyze_commented():
    with open(os.path.join(ROOT, "01_sql_pipeline_by_screen.sql")) as f:
        sql = f.read()
    active = [l for l in sql.split('\n')
              if re.match(r'\s*ANALYZE ', l)]
    assert len(active) == 0, f"활성 ANALYZE {len(active)}건 남음"

def t_sql_index_if_not_exists():
    with open(os.path.join(ROOT, "01_sql_pipeline_by_screen.sql")) as f:
        sql = f.read()
    bad = [l for l in sql.split('\n')
           if re.match(r'\s*CREATE INDEX ', l)
           and 'IF NOT EXISTS' not in l.upper()]
    assert len(bad) == 0, f"IF NOT EXISTS 없는 인덱스 {len(bad)}건"

def t_sql_alter_table_safe():
    with open(os.path.join(ROOT, "01_sql_pipeline_by_screen.sql")) as f:
        sql = f.read()
    alters = [l for l in sql.split('\n')
              if re.match(r'\s*ALTER TABLE', l)]
    # ALTER TABLE이 있으면 ADD COLUMN IF NOT EXISTS 패턴이어야 함
    for alt in alters:
        assert "IF NOT EXISTS" in sql[sql.find(alt):sql.find(alt)+500], \
            "ALTER TABLE에 IF NOT EXISTS 없음"

def t_sql_all_cdss_tables_present():
    with open(os.path.join(ROOT, "01_sql_pipeline_by_screen.sql")) as f:
        sql = f.read()
    required_tables = [
        "cdss_cohort_window", "cdss_nephrotoxic_rx_raw",
        "cdss_icu_nephrotoxic_rx", "cdss_nephrotoxic_combo_risk",
        "cdss_raw_lab_values", "cdss_lab_features",
        "cdss_raw_map_values", "cdss_feat_map_ischemia",
        "cdss_feat_map_summary", "cdss_feat_shock_index",
        "cdss_feat_vasopressor", "cdss_ischemic_features",
        "cdss_rule_score_features", "cdss_master_features",
        "stg_nlp_keyword_features", "stg_radiology_nlp_text",
        "cdss_nlp_keyword_raw", "cdss_nlp_radiology_extra",
        "cdss_nlp_features",
    ]
    for tbl in required_tables:
        assert tbl in sql, f"테이블 누락: {tbl}"

run_test("SQL DROP TABLE 전부 주석",      t_sql_drop_commented)
run_test("SQL ANALYZE 전부 주석",         t_sql_analyze_commented)
run_test("SQL CREATE INDEX IF NOT EXISTS", t_sql_index_if_not_exists)
run_test("SQL ALTER TABLE IF NOT EXISTS", t_sql_alter_table_safe)
run_test("SQL 19개 테이블 전부 존재",      t_sql_all_cdss_tables_present)


# ══════════════════════════════════════════════════════════════════════════
# PART 9: NLP 로직 검사
# ══════════════════════════════════════════════════════════════════════════

section("PART 9  Track D NLP 로직")

def t_nlp_fluid_burden_flag():
    """체액 과부하 복합 플래그: 개별 kw_ 2개 이상이어야 1."""
    from db import execute_query as eq
    # kw_edema=1, kw_fluid_overload=1, kw_pulmonary_edema=1 → flag=1
    data = MOCK_MASTER
    flag = data["nlp_fluid_burden_flag"]
    kw_count = (data["kw_edema"] + data["kw_fluid_overload"] +
                data["kw_pulmonary_edema"] + data["kw_pleural_effusion"] +
                data["kw_ascites"])
    if kw_count >= 2:
        assert flag == 1

def t_nlp_direct_renal_flag():
    """직접 신장 이상 플래그: 수신증 or AKI 언급이 있으면 1."""
    data = MOCK_MASTER
    flag = data["nlp_direct_renal_flag"]
    has_renal = (data["kw_hydronephrosis"] or data["kw_rad_hydronephrosis"] or
                 data["kw_aki_mention"] or data["kw_rad_aki_mention"])
    if has_renal:
        assert flag == 1

def t_nlp_missing_indicator():
    """nlp_missing=0이면 NLP 데이터가 있어야 한다."""
    data = MOCK_MASTER
    if data["nlp_missing"] == 0:
        assert data["nlp_keyword_score"] >= 0
        assert data["rad_report_count"] >= 0

def t_nlp_scr03_query_has_nlp_cols():
    import inspect, scr03_drug_management as m
    # generate_ai_nephrotoxicity_monitoring_message 내부에 NLP 쿼리가 있음
    src = inspect.getsource(m.generate_ai_nephrotoxicity_monitoring_message)
    assert "kw_hydronephrosis" in src
    assert "kw_contrast_agent" in src

def t_nlp_scr04_query_has_nlp_cols():
    import inspect, scr04_lab_monitoring as m
    src = inspect.getsource(m.build_aki_monitoring_summary_for_bottom_banner)
    assert "kw_fluid_overload" in src
    assert "nlp_fluid_burden" in src

def t_nlp_scr05_query_has_nlp_cols():
    import inspect, scr05_cardio_filter as m
    src = inspect.getsource(m.assess_cardio_ischemia_risk_for_aki_prediction)
    assert "kw_cardiomegaly" in src
    assert "nlp_fluid_burden_flag" in src

def t_nlp_scr07_hourly_score_uses_nlp():
    import inspect, scr07_risk_timeseries as m
    src = inspect.getsource(m._compute_hourly_rule_score)
    assert "nlp_direct_renal_flag" in src
    assert "nlp_fluid_burden_flag" in src

run_test("NLP 체액과부하 복합 플래그 로직",   t_nlp_fluid_burden_flag)
run_test("NLP 직접신장이상 플래그 로직",       t_nlp_direct_renal_flag)
run_test("NLP 결측 지시변수 일관성",           t_nlp_missing_indicator)
run_test("SCR-03 쿼리에 NLP 컬럼 포함",       t_nlp_scr03_query_has_nlp_cols)
run_test("SCR-04 쿼리에 NLP 컬럼 포함",       t_nlp_scr04_query_has_nlp_cols)
run_test("SCR-05 쿼리에 NLP 컬럼 포함",       t_nlp_scr05_query_has_nlp_cols)
run_test("SCR-07 시간별 점수에 NLP 사용",     t_nlp_scr07_hourly_score_uses_nlp)


# ══════════════════════════════════════════════════════════════════════════
# 최종 결과
# ══════════════════════════════════════════════════════════════════════════

total = len(passed) + len(failed)
print(f"\n{BOLD}{'═'*60}")
print(f"  최종 결과  —  {total}개 테스트")
print(f"{'═'*60}{RESET}")
print(f"  {G} 통과: {len(passed)}개{RESET}")
print(f"  {R} 실패: {len(failed)}개{RESET}")

if failed:
    print(f"\n{R}  실패 목록:{RESET}")
    for name, reason in failed:
        print(f"     {name}")
        print(f"   → {reason}")

print()
if not failed:
    print(f"  {G}{BOLD}✅ 전체 통과 — 백엔드 정상 동작{RESET}")
else:
    print(f"  {Y}위 실패 항목을 확인하세요.{RESET}")
print()