from typing import Optional
from db import execute_query, RiskLevel

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# scr05_cardio_filter.py  —  SCR-05 카디오 필터 화면 백엔드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# [담당 화면] SCR-05 카디오 필터 — 허혈/재관류 손상(IRI) 모니터링
#
# [화면 레이아웃]
#   ┌──────────────────────────────────────────────────────────────────┐
#   │ ♥ 카디오 필터: 허혈/재관류 손상 (IRI) 모니터링                 │
#   │  ┌──────────────────────┐  ┌─────────────────────────────┐      │
#   │  │   총 허혈 시간        │  │  현재 MAP (평균동맥압)       │      │
#   │  │      145 분          │  │       62  mmHg              │      │
#   │  │ ⚠ 권장(120분) 초과   │  │ ⚠ 목표치(65mmHg) 미달       │      │
#   │  │ 신손상 위험 증가      │  │ 관류압 3mmHg 부족 — 즉시 확인│      │
#   │  └──────────────────────┘  └─────────────────────────────┘      │
#   │ ────────────────────────────────────────────────────────────     │
#   │ AI IRI 분석 — 88% | IRI 고위험 | 카디오 필터 프로토콜 즉시 적용  │
#   └──────────────────────────────────────────────────────────────────┘
#
# [파일 분리 가이드]
#   ├── scr05_cardio_filter.py   (현재) 허혈 시간·MAP·IRI 위험도 계산
#   └── scr05_cardio_schemas.py  Pydantic 모델 분리 (선택)
# ═══════════════════════════════════════════════════════════════════════════

MAP_TARGET  = 65    # mmHg, 신장 관류압 최소 목표치
ISCHEMIA_SAFE_MIN = 120  # 분, 권장 최대 허혈 시간


def retrieve_total_ischemia_time_in_minutes(stay_id: int) -> Optional[float]:
    """[SCR-05] 좌측 카드 "총 허혈 시간" — MAP < 65 mmHg 누적 시간(분) 조회

    SQL STEP 3-B에서 계산된 map_below65_hours(시간)를 분으로 변환해 반환한다.
    이 값이 SCR-05 좌측 카드의 대형 숫자("145 분")에 직접 표시된다.

    [화면 위치] SCR-05 좌측 카드 대형 숫자
    [경고 기준] 120분(2h) 초과 → 카드 테두리 빨간색 + "권장 초과" 경고 텍스트

    Returns: 총 허혈 시간 (분), 데이터 없으면 None
    """
    sql = """
        SELECT map_below65_hours,
               map_below65_hours * 60 AS ischemia_min
        FROM aki_project.cdss_master_features
        WHERE stay_id = :stay_id
    """
    rows = execute_query(sql, {"stay_id": stay_id})
    if not rows or rows[0].get("ischemia_min") is None:
        return None
    return float(rows[0]["ischemia_min"])


def retrieve_current_map_reading_and_target_gap(stay_id: int) -> dict:
    """[SCR-05] 우측 카드 "현재 MAP" — 최근 MAP 측정값 + 목표치 차이 조회

    SQL STEP 3-C(cdss_feat_map_summary)에서 DISTINCT ON으로 추출한 current_map을
    cdss_master_features를 통해 조회한다.

    [화면 위치] SCR-05 우측 카드 대형 숫자 + 목표 미달 여부
    [경고 기준] current_map < 65 → 카드 테두리 주황색 + "목표치 미달" 텍스트

    Returns: {"current_map", "isch_map_min", "gap_from_target"}
    """
    sql = """
        SELECT
            current_map,
            isch_map_min,
            current_map - :target AS gap_from_target
        FROM aki_project.cdss_master_features
        WHERE stay_id = :stay_id
    """
    rows = execute_query(sql, {"stay_id": stay_id, "target": MAP_TARGET})
    return rows[0] if rows else {}


def check_whether_ischemia_time_exceeds_safe_threshold(ischemia_min: Optional[float]):
    """[SCR-05] 좌측 카드 색상·경고 텍스트 — 허혈 시간 기준 초과 여부 판단

    허혈 시간을 120분(권장 최대)과 비교해 카드 색상과 경고 문구를 결정한다.

    [화면 위치] SCR-05 좌측 카드 테두리색·하단 경고 텍스트 2줄
    Returns: (is_over, card_color_hex, warning_text, sub_message)
    """
    if ischemia_min is None:
        return False, "#e5e7eb", "데이터 없음", "MAP 기록 확인 필요"
    if ischemia_min > ISCHEMIA_SAFE_MIN:
        return True, "#ef4444", f"권장({ISCHEMIA_SAFE_MIN}분) 초과", "신손상 위험 증가"
    if ischemia_min > ISCHEMIA_SAFE_MIN * 0.75:   # 90분~120분: 경계
        return False, "#f97316", f"권장 기준 {ISCHEMIA_SAFE_MIN}분 근접", "주의 모니터링"
    return False, "#e5e7eb", "기준 이내", "현재 안전 범위"


def check_whether_map_is_below_renal_perfusion_target(current_map: Optional[float]):
    """[SCR-05] 우측 카드 색상·경고 텍스트 — MAP 목표치 달성 여부 판단

    MAP을 65 mmHg(신장 관류압 최소 목표)와 비교해 카드 색상과 경고 문구를 결정한다.

    [화면 위치] SCR-05 우측 카드 테두리색·하단 경고 텍스트 2줄
    Returns: (is_below, card_color_hex, warning_text, sub_message)
    """
    if current_map is None:
        return False, "#e5e7eb", "데이터 없음", "MAP 측정 기록 없음"
    if current_map < MAP_TARGET:
        gap = MAP_TARGET - current_map
        return True, "#f97316", f"목표치({MAP_TARGET}mmHg) 미달", f"관류압 {gap:.0f}mmHg 부족 — 즉시 확인"
    if current_map < MAP_TARGET + 5:   # 65~70: 경계
        return False, "#fef3c7", "목표 최소 달성", "모니터링 강화 권고"
    return False, "#e5e7eb", "목표 달성", "신장 관류압 유지 중"


def assess_cardio_ischemia_risk_for_aki_prediction(stay_id: int) -> dict:
    """[SCR-05] 하단 AI IRI 배너 — 허혈 시간·MAP·승압제 조합으로 IRI 위험도 평가

    두 카드(허혈 시간·MAP)와 승압제 사용 여부를 조합해 IRI 위험 점수(0~3)를 산출한다.
    cardio_risk_score 2 이상 → "고위험" → 배너 자동 활성화.

    [화면 위치] SCR-05 하단 파란 배너 "IRI 고위험" 판정 근거
    Returns: {"cardio_risk_score", "iri_risk_label", "ischemia_min", "current_map", ...}
    """
    sql = """
        SELECT
            map_below65_hours * 60  AS ischemia_min,
            flag_ischemia_over120min,
            current_map, isch_map_min,
            vasopressor_flag,
            isch_shock_index        AS shock_index,
            rule_based_score, score_ischemia, score_map,
            -- Track D NLP: 방사선 판독 소견 — IRI 배너 경고 강화에 사용
            COALESCE(kw_cardiomegaly,       0) AS kw_cardiomegaly,
            COALESCE(kw_contrast_agent,     0) AS kw_contrast_agent,
            COALESCE(kw_pleural_effusion,   0) AS kw_pleural_effusion,
            COALESCE(kw_pulmonary_edema,    0) AS kw_pulmonary_edema,
            COALESCE(nlp_fluid_burden_flag, 0) AS nlp_fluid_burden_flag,
            COALESCE(nlp_missing,           1) AS nlp_missing
        FROM aki_project.cdss_master_features
        WHERE stay_id = :stay_id
    """
    rows = execute_query(sql, {"stay_id": stay_id})
    if not rows:
        return {}
    d             = rows[0]
    ischemia_over = bool(d.get("flag_ischemia_over120min", 0))
    map_below     = (d.get("current_map") or 999) < MAP_TARGET
    vaso_on       = bool(d.get("vasopressor_flag", 0))

    # Track D NLP 소견 — IRI 위험도 가중
    nlp_available   = not bool(d.get("nlp_missing", 1))
    kw_cardiomegaly = bool(d.get("kw_cardiomegaly", 0))   # 심비대 → 심신증후군
    kw_contrast     = bool(d.get("kw_contrast_agent", 0))  # 조영제 → CI-AKI
    kw_fluid_burden = bool(d.get("nlp_fluid_burden_flag", 0))  # 체액 과부하
    kw_pulm         = bool(d.get("kw_pulmonary_edema", 0))
    kw_pleural      = bool(d.get("kw_pleural_effusion", 0))

    # 카디오 위험 점수 (0~3 기본 + NLP 가중)
    score = int(ischemia_over) + int(map_below) + int(vaso_on)

    # NLP 소견이 있으면 cardio_risk_score에 추가 반영
    # 심비대: 심신증후군 위험 → 허혈 위험과 동일한 가중치
    # 체액 과부하: 심장 부담 → 신관류 감소
    nlp_cardio_bonus = 0
    if nlp_available:
        if kw_cardiomegaly:  nlp_cardio_bonus += 1
        if kw_fluid_burden:  nlp_cardio_bonus += 1

    total_score = min(score + nlp_cardio_bonus, 4)  # 최대 4점
    label = "고위험" if total_score >= 2 else ("중위험" if total_score == 1 else "저위험")

    return {
        "cardio_risk_score": total_score,
        "iri_risk_label":    label,
        "ischemia_min":      float(d.get("ischemia_min") or 0),
        "current_map":       d.get("current_map"),
        "vasopressor_on":    vaso_on,
        "shock_index":       d.get("shock_index"),
        "rule_based_score":  d.get("rule_based_score") or 0,
        # NLP 소견 (배너 메시지 생성에 전달)
        "nlp_available":     nlp_available,
        "kw_cardiomegaly":   kw_cardiomegaly,
        "kw_contrast":       kw_contrast,
        "kw_fluid_burden":   kw_fluid_burden,
        "kw_pulm":           kw_pulm,
        "kw_pleural":        kw_pleural,
    }


# ═══════════════════════════════════════════════════════════════════════════

def generate_cardio_filter_protocol_recommendation(
    cardio_risk: dict,
    ml_probability=None,
) -> dict:
    """[SCR-05] 하단 AI IRI 배너 메시지 생성

    허혈 시간 + MAP + NLP 소견을 종합해 SCR-05 하단 고정 배너 문구를 반환한다.

    [화면 위치] SCR-05 최하단 파란 고정 박스
      "AKI 예측 분석 — 88% | IRI 고위험 | 카디오 필터 프로토콜 즉시 적용 권고"
      "총 허혈 시간 145분 + MAP 62mmHg + [NLP 소견] 조합은 AKI 발생 가능성을 높입니다."
    """
    if not cardio_risk:
        return {"is_active": False, "message": "데이터 없음", "sub_message": ""}

    iri_label    = cardio_risk.get("iri_risk_label", "저위험")
    ischemia_min = cardio_risk.get("ischemia_min", 0)
    current_map  = cardio_risk.get("current_map")
    prob_pct     = round(ml_probability * 100, 0) if ml_probability else None

    # Track D NLP 경고 문구 조합
    nlp_available = cardio_risk.get("nlp_available", False)
    nlp_warnings  = []
    if nlp_available:
        if cardio_risk.get("kw_cardiomegaly"):
            nlp_warnings.append("심비대 소견 → 심신증후군 위험")
        if cardio_risk.get("kw_contrast"):
            nlp_warnings.append("조영제 투여 이력 → CI-AKI 추가 위험")
        if cardio_risk.get("kw_fluid_burden"):
            nlp_warnings.append("체액 과부하 복합 소견")

    if iri_label == "고위험":
        from db import RiskLevel
        risk   = RiskLevel(level="high", label_kr="높음",
                           color="#ef4444", border_color="#ef4444")
        action = "카디오 필터 프로토콜 즉시 적용 권고"
    elif iri_label == "중위험":
        from db import RiskLevel
        risk   = RiskLevel(level="warning", label_kr="보통",
                           color="#f97316", border_color="#f97316")
        action = "MAP 목표 달성 및 수액 보충 검토"
    else:
        from db import RiskLevel
        risk   = RiskLevel(level="normal", label_kr="낮음",
                           color="#22c55e", border_color="#d1d5db")
        action = "현재 모니터링 유지"

    map_str      = f"MAP {current_map:.0f}mmHg" if current_map else "MAP 측정 필요"
    ischemia_str = f"총 허혈 시간 {ischemia_min:.0f}분"

    combined = (
        f"{ischemia_str} + {map_str} 조합은 허혈/재관류 손상으로 인한 "
        f"급성신손상(AKI) 발생 가능성을 크게 높입니다."
        if iri_label == "고위험"
        else f"{ischemia_str}, {map_str} — 지속 모니터링이 필요합니다."
    )
    if nlp_warnings:
        combined += " | 방사선 판독: " + " · ".join(nlp_warnings)

    return {
        "is_active":           iri_label in ("고위험", "중위험"),
        "iri_risk_label":      iri_label,
        "risk_level":          risk,
        "risk_probability_pct":prob_pct,
        "combined_analysis":   combined,
        "protocol_action":     action,
    }

def build_cardio_filter_screen_response(
    stay_id: int,
    ml_probability=None,
) -> dict:
    """[SCR-05] 카디오 필터 화면 전체 데이터 조립

    SCR-05 화면 렌더링에 필요한 모든 데이터를 한 번에 반환한다.
    main.py의 GET /api/scr05/cardio/{stay_id} 엔드포인트가 이 함수를 호출한다.

    [반환 구조]
      ischemia_card   → 좌측 카드 (총 허혈 시간·경고)
      map_card        → 우측 카드 (현재 MAP·경고)
      cardio_banner   → 하단 AI IRI 배너 (NLP 경고 포함)
    """
    ischemia_min = retrieve_total_ischemia_time_in_minutes(stay_id)
    map_data     = retrieve_current_map_reading_and_target_gap(stay_id)
    cardio_risk  = assess_cardio_ischemia_risk_for_aki_prediction(stay_id)

    is_over, i_color, i_warn, i_sub = check_whether_ischemia_time_exceeds_safe_threshold(ischemia_min)
    current_map = map_data.get("current_map")
    is_below, m_color, m_warn, m_sub = check_whether_map_is_below_renal_perfusion_target(
        float(current_map) if current_map is not None else None
    )

    banner = generate_cardio_filter_protocol_recommendation(cardio_risk, ml_probability)

    return {
        "stay_id": stay_id,
        "ischemia_card": {
            "total_minutes":     ischemia_min or 0,
            "threshold_minutes": ISCHEMIA_SAFE_MIN,
            "is_over_threshold": is_over,
            "card_color":        i_color,
            "warning_text":      i_warn,
            "sub_message":       i_sub,
        },
        "map_card": {
            "current_map":     float(current_map) if current_map is not None else None,
            "target_map":      MAP_TARGET,
            "gap_from_target": map_data.get("gap_from_target"),
            "is_below_target": is_below,
            "card_color":      m_color,
            "warning_text":    m_warn,
            "sub_message":     m_sub,
        },
        "vasopressor_flag": cardio_risk.get("vasopressor_on", False),
        "shock_index":      cardio_risk.get("shock_index"),
        "cardio_aki_banner":banner,
    }