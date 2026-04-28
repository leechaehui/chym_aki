"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
scr06_ai_risk_score.py  —  SCR-06 AI 급성 신손상 예측도 화면 백엔드
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[담당 화면] SCR-06 AI 급성 신손상 예측도

[화면 레이아웃]
  ┌─────────────────────────────────────────────────────────────────┐
  │ ⚠ AI 급성 신손상 예측도 — 12시간 내 AKI 발생 확률 매우 높음   │
  │  ┌──────────────────────────────────┐                          │
  │  │           88  %                  │  ← XGBoost + 규칙 혼합  │
  │  │  규칙 기반 임상 위험도 지수       │  ← 빨간 배경(≥70점)    │
  │  └──────────────────────────────────┘                          │
  │  위험 요인      현재값    기준      기여도   상태               │
  │  크레아티닌     2.1 mg/dL > 1.5    +30점   기준 초과           │
  │  BUN           42 mg/dL  > 30     +20점   기준 초과           │
  │  eGFR          38 mL/min < 45     +20점   기준 초과           │
  │  허혈 시간      145분     > 120분  +15점   기준 초과           │
  │  MAP           62 mmHg   < 65     +15점   기준 초과           │
  │  합산: 100점 → 위험도 88%  (임계값 ≥ 70점: 고위험)            │
  └─────────────────────────────────────────────────────────────────┘

[파일 분리 구조]
  scr06_ai_risk_score.py   ← 현재 파일 (규칙 계산 + 화면 조립)
  xgb_model/inference.py   ← XGBoost 추론 엔진 (AKIInferenceEngine)
  xgb_model/feature_config.py, preprocessing.py  ← 학습·추론 공유 설정

[XGBoost 연동 방식]
  AKIInferenceEngine 싱글턴을 import해서 사용한다.
  모델 파일(model/xgb_aki.json)이 없으면 ModelNotReadyError를 잡아
  rule_based_score만 표시하는 fallback으로 자동 전환한다.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import logging
from typing import Optional

from pydantic import BaseModel
from db import execute_query, RiskLevel

# XGBoost 추론 엔진 import
# inference.py가 없거나 모델 미준비 시에도 서버가 기동되어야 하므로
# import 실패를 잡아서 is_ready=False 상태로 처리한다.
try:
    from xgb_model.inference import aki_engine, ModelNotReadyError, InferenceError
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    aki_engine = None
    class ModelNotReadyError(Exception): pass
    class InferenceError(Exception): pass

logger = logging.getLogger("aki_cdss.scr06")

HIGH_RISK_CUTOFF = 70   # 규칙 기반 점수 고위험 임계값

# 규칙 기반 5개 요인 메타 정보 (SCR-06 테이블 헤더)
RULE_META = {
    "creatinine":    {"label": "크레아티닌", "unit": "mg/dL", "op": "> 1.5",  "score": 30},
    "bun":           {"label": "BUN",        "unit": "mg/dL", "op": "> 30",   "score": 20},
    "egfr":          {"label": "eGFR",       "unit": "mL/min","op": "< 45",   "score": 20},
    "ischemia_time": {"label": "허혈 시간",  "unit": "분",    "op": "> 120분", "score": 15},
    "map":           {"label": "MAP",        "unit": "mmHg",  "op": "< 65",   "score": 15},
}


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic 모델
# ─────────────────────────────────────────────────────────────────────────────

class RiskFactorRow(BaseModel):
    """[SCR-06] 위험 요인 테이블 1행 — 5개 요인별 현재값·기준·기여도·상태."""
    factor_name:   str
    current_value: Optional[float]
    unit:          str
    threshold_op:  str
    contribution:  int
    is_exceeded:   bool
    status_label:  str    # "기준 초과" | "정상 범위"
    status_color:  str    # "#f97316" | "#6b7280"


class RiskScoreDisplay(BaseModel):
    """[SCR-06] 대형 숫자 박스 — 최종 위험도·등급·배경색."""
    displayed_value:   int           # 0~100 (화면 표시)
    display_unit:      str           # "%"
    rule_based_score:  int           # 규칙 기반 원점수
    ml_probability:    Optional[float]  # XGBoost 확률 (없으면 None)
    ml_missing_features: int = 0     # XGBoost 입력 중 결측 피처 수 (신뢰도 지표)
    is_high_risk:      bool
    risk_level:        RiskLevel
    score_explanation: str           # "XGBoost + 규칙 기반" vs "규칙 기반 단독"
    model_ready:       bool          # XGBoost 모델 준비 여부


class AIRiskScoreResponse(BaseModel):
    """[SCR-06] 전체 화면 응답."""
    stay_id:              int
    alert_message:        str
    risk_display:         RiskScoreDisplay
    risk_factor_table:    list[RiskFactorRow]
    total_score:          int
    score_interpretation: str
    upgrade_note:         str


# ─────────────────────────────────────────────────────────────────────────────
# 핵심 함수
# ─────────────────────────────────────────────────────────────────────────────

def calculate_rule_based_risk_score_with_factor_breakdown(stay_id: int) -> dict:
    """[SCR-06] 위험 요인 테이블 5행 원천 데이터 — 규칙 기반 점수 및 요인별 분해 조회

    cdss_rule_score_features 에서 각 요인의 현재값·초과 여부·기여 점수를 조회해
    SCR-06 위험 요인 테이블 5행 데이터를 구성한다.

    [화면 위치] SCR-06 중앙 테이블 (크레아티닌·BUN·eGFR·허혈시간·MAP 5행)
    [데이터 소스] cdss_rule_score_features (SQL STEP 4)

    Returns:
        {"rule_based_score": int, "factors": [{"key", "value", "score", "exceeded"}, ...]}
    """
    sql = """
        SELECT
            val_cr, val_bun, val_egfr, val_ischemia_min, val_map,
            flag_cr, flag_bun, flag_egfr, flag_ischemia, flag_map,
            score_cr, score_bun, score_egfr, score_ischemia, score_map,
            rule_based_score
        FROM aki_project.cdss_rule_score_features
        WHERE stay_id = :stay_id
    """
    rows = execute_query(sql, {"stay_id": stay_id})
    if not rows:
        return {"rule_based_score": 0, "factors": []}

    d = rows[0]
    # 5개 요인을 SCR-06 테이블 행 순서와 동일하게 구성
    factors = [
        {"key": "creatinine",    "value": d.get("val_cr"),          "score": int(d.get("score_cr")       or 0), "exceeded": bool(d.get("flag_cr",       0))},
        {"key": "bun",           "value": d.get("val_bun"),         "score": int(d.get("score_bun")      or 0), "exceeded": bool(d.get("flag_bun",      0))},
        {"key": "egfr",          "value": d.get("val_egfr"),        "score": int(d.get("score_egfr")     or 0), "exceeded": bool(d.get("flag_egfr",     0))},
        {"key": "ischemia_time", "value": d.get("val_ischemia_min"),"score": int(d.get("score_ischemia") or 0), "exceeded": bool(d.get("flag_ischemia", 0))},
        {"key": "map",           "value": d.get("val_map"),         "score": int(d.get("score_map")      or 0), "exceeded": bool(d.get("flag_map",      0))},
    ]
    return {"rule_based_score": int(d.get("rule_based_score") or 0), "factors": factors}


def predict_aki_probability_using_xgboost_model(stay_id: int) -> tuple[Optional[float], int]:
    """[SCR-06] 대형 숫자 ML 기여값 — AKIInferenceEngine으로 XGBoost 확률 예측

    AKIInferenceEngine 싱글턴에 예측을 위임한다.
    모델이 준비되지 않았거나 예측 실패 시 (None, 0) 을 반환해
    호출 측에서 규칙 기반 fallback을 사용할 수 있게 한다.

    [화면 위치] SCR-06 대형 숫자 박스의 ML 기여분 (규칙 40% + ML 60% 혼합)
    [모델 없을 때] None → rule_based_score 단독 표시 (SCR-06 "규칙 기반 단독" 레이블)

    Args:
        stay_id: ICU 체류 ID

    Returns:
        (AKI 확률 0~1 또는 None, 결측 피처 수)
    """
    if not XGB_AVAILABLE or aki_engine is None or not aki_engine.is_ready:
        logger.debug(f"XGBoost 미준비 — stay_id={stay_id} 규칙 기반 fallback")
        return None, 0

    try:
        result = aki_engine.predict_single(stay_id)
        return result.aki_probability, result.missing_features
    except ModelNotReadyError:
        logger.warning("ModelNotReadyError — 규칙 기반 fallback")
        return None, 0
    except InferenceError as e:
        logger.error(f"InferenceError stay_id={stay_id}: {e}")
        return None, 0


def combine_rule_score_and_ml_probability_into_final_risk(
    rule_score:     int,
    ml_probability: Optional[float],
) -> tuple[int, RiskLevel]:
    """[SCR-06] 대형 숫자 최종값 — 규칙(40%) + XGBoost(60%) 가중 평균

    XGBoost 모델이 있으면 두 값을 가중 평균해 최종 위험도를 산출하고,
    없으면 rule_based_score를 그대로 사용한다.

    [화면 위치] SCR-06 대형 숫자 박스 수치와 배경색
    [가중치 근거] 규칙 40% + ML 60%: ML이 더 정교하나 임상 규칙도 보존
    """
    if ml_probability is not None:
        ml_score    = ml_probability * 100
        final_value = int(rule_score * 0.4 + ml_score * 0.6)
    else:
        final_value = rule_score
    final_value = max(0, min(100, final_value))

    if final_value >= HIGH_RISK_CUTOFF:
        risk = RiskLevel(level="high",    label_kr="높음", color="#ef4444", border_color="#ef4444")
    elif final_value >= 40:
        risk = RiskLevel(level="warning", label_kr="보통", color="#f97316", border_color="#f97316")
    else:
        risk = RiskLevel(level="normal",  label_kr="낮음", color="#22c55e", border_color="#d1d5db")

    return final_value, risk


def build_risk_factor_contribution_table_for_display(
    factors: list[dict],
) -> list[RiskFactorRow]:
    """[SCR-06] 위험 요인 테이블 5행 조립 — 요인별 기여도 데이터를 화면 행으로 변환

    각 요인의 현재값·기준·기여 점수를 SCR-06 테이블 형식으로 변환한다.
    기여도 "+30점" 텍스트와 "기준 초과"(주황) / "정상 범위"(회색) 상태를 결정한다.

    [화면 위치] SCR-06 중앙 5행 테이블
      열: 위험 요인 | 현재값 | 기준 | 기여도 | 상태
    """
    rows = []
    for f in factors:
        meta = RULE_META.get(f["key"], {})
        rows.append(RiskFactorRow(
            factor_name   = meta.get("label", f["key"]),
            current_value = f.get("value"),
            unit          = meta.get("unit", ""),
            threshold_op  = meta.get("op", ""),
            contribution  = f.get("score", 0),
            is_exceeded   = f.get("exceeded", False),
            status_label  = "기준 초과" if f.get("exceeded") else "정상 범위",
            status_color  = "#f97316"   if f.get("exceeded") else "#6b7280",
        ))
    return rows


def build_ai_risk_score_screen_response(stay_id: int) -> AIRiskScoreResponse:
    """[SCR-06] AI 급성 신손상 예측도 화면 전체 데이터 조립

    규칙 기반 점수 계산 → XGBoost 추론 → 가중 혼합 → 화면 응답 조립 순서로 실행.
    main.py의 GET /api/scr06/risk/{stay_id} 엔드포인트가 이 함수를 호출한다.

    [반환 구조]
      risk_display.displayed_value  → 대형 숫자 박스 수치
      risk_factor_table             → 5행 위험 요인 테이블
      alert_message                 → 상단 빨간 배너 문구
      score_interpretation          → 하단 합산 설명 텍스트

    Args:
        stay_id: ICU 체류 ID

    Returns:
        AIRiskScoreResponse (전체 화면 데이터)
    """
    # ── 1. 규칙 기반 점수 ─────────────────────────────────────────────────
    rule_data    = calculate_rule_based_risk_score_with_factor_breakdown(stay_id)
    rule_score   = rule_data["rule_based_score"]
    factors      = rule_data["factors"]

    # ── 2. XGBoost 예측 ───────────────────────────────────────────────────
    ml_prob, missing_feats = predict_aki_probability_using_xgboost_model(stay_id)
    model_ready  = ml_prob is not None

    # ── 3. 최종 위험도 결정 ───────────────────────────────────────────────
    display_val, risk_level = combine_rule_score_and_ml_probability_into_final_risk(
        rule_score, ml_prob
    )

    # ── 4. 위험 요인 테이블 조립 ──────────────────────────────────────────
    factor_table = build_risk_factor_contribution_table_for_display(factors)

    # ── 5. 합산 점수 설명 문구 ────────────────────────────────────────────
    total_score  = sum(f.contribution for f in factor_table)
    if ml_prob is not None:
        score_interp = (
            f"규칙 합산 {total_score}점 + XGBoost {ml_prob*100:.0f}% → "
            f"혼합 위험도 {display_val}%  (고위험 기준 ≥ {HIGH_RISK_CUTOFF}점)"
        )
        score_explanation = f"XGBoost {ml_prob*100:.0f}% × 60% (NLP 포함) + 규칙 {rule_score}점 × 40%"
    else:
        score_interp = (
            f"합산: {total_score}점 → 위험도 {display_val}%  "
            f"(임계값 ≥ {HIGH_RISK_CUTOFF}점: 고위험)"
        )
        score_explanation = "규칙 기반 임상 위험도 지수 (0~100점) — XGBoost 모델 준비 중"

    # ── 6. 상단 경고 배너 문구 ────────────────────────────────────────────
    if display_val >= HIGH_RISK_CUTOFF:
        alert_msg = "⚠ AI 급성 신손상 예측도 — 12시간 내 AKI 발생 확률 매우 높음"
    elif display_val >= 40:
        alert_msg = "⚠ AI 급성 신손상 예측도 — 12시간 내 AKI 발생 가능성 있음"
    else:
        alert_msg = "AI 급성 신손상 예측도 — 현재 저위험 상태"

    # ── 7. 대형 숫자 박스 데이터 조립 ────────────────────────────────────
    risk_display = RiskScoreDisplay(
        displayed_value      = display_val,
        display_unit         = "%",
        rule_based_score     = rule_score,
        ml_probability       = ml_prob,
        ml_missing_features  = missing_feats,
        is_high_risk         = display_val >= HIGH_RISK_CUTOFF,
        risk_level           = risk_level,
        score_explanation    = score_explanation,
        model_ready          = model_ready,
    )

    return AIRiskScoreResponse(
        stay_id              = stay_id,
        alert_message        = alert_msg,
        risk_display         = risk_display,
        risk_factor_table    = factor_table,
        total_score          = total_score,
        score_interpretation = score_interp,
        upgrade_note         = (
            "✅ XGBoost 모델 적용 중 (Track D NLP 피처 포함)" if model_ready else
            "⏳ 모델 학습 후 XGBoost 자동 활성화 — train.py 실행 필요"
        ),
    )