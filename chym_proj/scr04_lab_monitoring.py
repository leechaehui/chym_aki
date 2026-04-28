"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
scr04_lab_monitoring.py  —  SCR-04 검사 결과 & AKI 모니터링 화면 백엔드
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[담당 화면] SCR-04 검사 결과 & AKI 모니터링

[화면 레이아웃]
  ┌──────────────────────────────────────────────────────────────────────┐
  │  [04.24]  [04.25]  [◉ 04.26 오늘]                                  │
  │  검사항목      결과값        정상범위   상태      추이               │
  │  크레아티닌    2.1 mg/dL    < 1.0      높음  ↑  (빨간 행)          │
  │  BUN          42 mg/dL     7–20       높음  ↑  (빨간 행)          │
  │  eGFR         38 mL/min    > 60       저하  ↓  (주황 행)          │
  │  헤모글로빈    11.2 g/dL    12–16      경계  →  (회색 행)          │
  │ ─────────────────────────────────────────────────────────────────── │
  │  AKI 예측 — 높음  |  허혈 재관류 손상 의심  |  즉시 모니터링 권고  │
  └──────────────────────────────────────────────────────────────────────┘

[파일 분리 가이드]
  ├── scr04_lab_monitoring.py    (현재) 조회·분류·배너 생성
  └── scr04_lab_schemas.py       Pydantic 모델 분리 (선택)

[SQL 변경 사항 반영]
  - cdss_raw_lab_values 테이블명 반영 (테이블 rename)
  - cdss_master_features 에 hospital_expire_flag 컬럼 추가됨 → aki_banner에 영향 없음
  - eGFR 공식: SQL(STEP 4)과 동일한 CKD-EPI 2021 계수 사용 (백엔드 fallback용)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import datetime
from collections import defaultdict
from typing import Optional
from pydantic import BaseModel
from db import execute_query, classify_value_as_risk_level, RiskLevel


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic 모델
# ─────────────────────────────────────────────────────────────────────────────

class LabResultRow(BaseModel):
    """SCR-04 결과 테이블 1행. 화면 5열(항목·값·정상범위·상태·추이)에 대응."""
    item_name:    str
    item_key:     str            # creatinine | bun | egfr | hemoglobin
    value:        Optional[float]
    unit:         str
    normal_range: str
    status:       RiskLevel      # 빨강·주황·회색
    trend:        str            # up | down | stable | unknown
    trend_arrow:  str            # ↑ | ↓ | → | -
    prev_value:   Optional[float]

class DateTabData(BaseModel):
    """SCR-04 날짜 탭 1개 (04.24·04.25·04.26)."""
    date_str:    str             # "04.26" 형식
    is_today:    bool
    lab_results: list[LabResultRow]

class AKIMonitoringBanner(BaseModel):
    """SCR-04 하단 고정 AKI 예측 배너."""
    prediction_level: str
    risk_level:       RiskLevel
    primary_cause:    str
    recommendation:   str
    detail_message:   str

class LabMonitoringResponse(BaseModel):
    """SCR-04 전체 화면 응답."""
    stay_id:         int
    date_tabs:       list[DateTabData]
    current_results: list[LabResultRow]
    aki_banner:      AKIMonitoringBanner


# ─────────────────────────────────────────────────────────────────────────────
# 함수 구현
# ─────────────────────────────────────────────────────────────────────────────

def fetch_lab_results_with_date_tabs(stay_id: int) -> list[dict]:
    """[SCR-04] 날짜 탭 + 결과 테이블 — 최근 3일치 혈액검사 결과 조회

    SCR-04 날짜 탭(04.24·04.25·04.26) 전환에 필요한 일별 검사 결과와
    전일 대비 추이(↑↓→) 계산을 위한 전날 수치를 함께 조회한다.

    [화면 위치] SCR-04 상단 날짜 탭 + 중앙 결과 테이블 4행
    [데이터 소스] cdss_raw_lab_values (itemid 50912·51006·51222)
    [날짜 탭 생성 방식] DATE(charttime) 기준으로 최근 3일 그룹화

    주의: 같은 날 여러 번 검사한 경우 DISTINCT ON으로 가장 최신 값 1개만 사용.

    Args:
        stay_id: ICU 체류 ID

    Returns:
        [{"lab_date", "itemid", "valuenum", "prev_valuenum"}, ...]
    """
    sql = """
        WITH daily_labs AS (
            SELECT
                r.stay_id,
                DATE(r.charttime)  AS lab_date,
                r.itemid,
                r.valuenum,
                -- 전날 같은 항목 수치 → 추이 화살표 계산의 분모
                LAG(r.valuenum) OVER (
                    PARTITION BY r.stay_id, r.itemid
                    ORDER BY DATE(r.charttime)
                ) AS prev_valuenum
            FROM aki_project.cdss_raw_lab_values r
            WHERE r.stay_id = :stay_id
        ),
        -- 같은 날 여러 번 측정 시 가장 최신 1건만 사용
        latest_per_day AS (
            SELECT DISTINCT ON (lab_date, itemid)
                lab_date, itemid, valuenum, prev_valuenum
            FROM daily_labs
            ORDER BY lab_date, itemid, valuenum DESC
        )
        SELECT *
        FROM latest_per_day
        ORDER BY lab_date DESC
        LIMIT 12   -- 4항목 × 3일
    """
    return execute_query(sql, {"stay_id": stay_id})


def classify_lab_result_status_by_normal_range(item_key: str, value: float) -> RiskLevel:
    """[SCR-04] 결과 테이블 '상태' 열 — 수치를 정상범위와 비교해 색상 결정

    수치를 항목별 임계값과 비교해 높음(빨강)/경계(주황)/정상(회색)을 반환한다.
    반환된 RiskLevel.color 값이 행 배경색과 좌측 보더색에 직접 적용된다.

    [화면 위치] SCR-04 결과 테이블 "상태" 열 + 행 배경색
    [임계값 근거] KDIGO 2012 / 임상 일반 정상범위
      크레아티닌: >1.5 높음, >1.0 경계
      BUN:        >30  높음, >20  경계
      eGFR:       <45  높음, <60  경계   (하한 기준)
      헤모글로빈: <8   높음, <12  경계   (하한 기준)
    """
    thresholds = {
        "creatinine": dict(warning_threshold=1.0, high_threshold=1.5),
        "bun":        dict(warning_threshold=20.0, high_threshold=30.0),
        "egfr":       dict(low_threshold=45.0, low_warning=60.0),
        "hemoglobin": dict(low_threshold=8.0,  low_warning=12.0),
    }
    return classify_value_as_risk_level(value, **thresholds.get(item_key, {}))


def calculate_trend_direction_from_sequential_values(
    current_value: Optional[float],
    prev_value:    Optional[float],
    threshold_pct: float = 5.0,
) -> tuple[str, str]:
    """[SCR-04] 결과 테이블 '추이' 열 — 전일 대비 변화 방향 계산

    현재값과 전날값을 비교해 추이 방향(up/down/stable)과 화살표 문자를 반환한다.
    5% 미만 변화는 '안정(→)'으로 처리해 임상적으로 무의미한 변동을 필터링한다.

    [화면 위치] SCR-04 결과 테이블 마지막 "추이" 열의 ↑ ↓ → - 아이콘

    Args:
        current_value: 오늘 검사 수치
        prev_value:    전날 검사 수치
        threshold_pct: 변화율 기준 (기본 5%)

    Returns:
        (trend: "up"|"down"|"stable"|"unknown", arrow: "↑"|"↓"|"→"|"-")
    """
    if current_value is None or prev_value is None or prev_value == 0:
        return "unknown", "-"
    change_pct = abs((current_value - prev_value) / prev_value) * 100
    if change_pct < threshold_pct:
        return "stable", "→"
    return ("up", "↑") if current_value > prev_value else ("down", "↓")


def derive_egfr_from_creatinine_and_demographics(
    cr_value: float, age: int, gender: str
) -> Optional[float]:
    """[SCR-04] 결과 테이블 세 번째 행 — CKD-EPI 2021로 eGFR 실시간 계산

    DB에 저장된 eGFR이 없을 때 백엔드에서 실시간으로 계산하는 fallback 함수.
    SQL STEP 4(cdss_rule_score_features)의 egfr 계산과 동일한 공식을 사용해야
    화면 표시값과 DB 피처값이 일치한다.

    [화면 위치] SCR-04 결과 테이블 세 번째 행 "eGFR: 38 mL/min"
    [공식] CKD-EPI 2021 (인종 무관 버전)
      여성 κ=0.7: 142 × min(Cr/0.7,1)^-0.241 × max(Cr/0.7,1)^-1.200 × 0.9938^age × 1.012
      남성 κ=0.9: 142 × min(Cr/0.9,1)^-0.302 × max(Cr/0.9,1)^-1.200 × 0.9938^age
    """
    if not cr_value or cr_value <= 0:
        return None
    is_female  = gender.upper() == "F"
    kappa      = 0.7 if is_female else 0.9
    alpha      = -0.241 if is_female else -0.302
    sex_factor = 1.012 if is_female else 1.0
    ratio      = cr_value / kappa
    egfr = 142.0 * (min(ratio,1.0)**alpha) * (max(ratio,1.0)**-1.200) \
                 * (0.9938**age) * sex_factor
    return round(egfr, 1)


def build_aki_monitoring_summary_for_bottom_banner(stay_id: int) -> AKIMonitoringBanner:
    """[SCR-04] 하단 고정 AKI 예측 배너 — 검사 지표 종합 분석 메시지 생성

    크레아티닌·BUN·eGFR·MAP·허혈 지표를 종합해 AKI 위험도와 주요 원인,
    임상 권고 문구를 생성한다. 스크롤에 무관하게 항상 화면 하단에 고정 표시된다.

    [화면 위치] SCR-04 최하단 파란 고정 박스
      첫째 줄: "AKI 예측 — 높음 | 허혈 재관류 손상 의심 | 즉시 모니터링 권고"
      둘째 줄: "크레아티닌 2.1 mg/dL, BUN 42 mg/dL, eGFR 38 mL/min 종합 분석"

    [위험도 판정]
      rule_based_score >= 70 → 높음(빨강), >= 40 → 보통(주황), else → 낮음
    """
    sql = """
        SELECT cr_max, cr_mean, cr_delta, bun_max, bun_cr_ratio,
               egfr_ckdepi, rule_based_score, high_risk_flag,
               flag_cr, flag_bun, flag_egfr, vasopressor_flag, map_below65_hours,
               -- Track D NLP: 방사선 판독 소견 — AKI 배너 원인 분석에 사용
               COALESCE(kw_fluid_overload,     0) AS kw_fluid_overload,
               COALESCE(kw_edema,              0) AS kw_edema,
               COALESCE(kw_pulmonary_edema,    0) AS kw_pulmonary_edema,
               COALESCE(kw_pleural_effusion,   0) AS kw_pleural_effusion,
               COALESCE(kw_hydronephrosis,     0) AS kw_hydronephrosis,
               COALESCE(kw_ascites,            0) AS kw_ascites,
               COALESCE(nlp_fluid_burden_flag, 0) AS nlp_fluid_burden_flag,
               COALESCE(nlp_direct_renal_flag, 0) AS nlp_direct_renal_flag,
               COALESCE(nlp_missing,           1) AS nlp_missing
        FROM aki_project.cdss_master_features
        WHERE stay_id = :stay_id
    """
    rows = execute_query(sql, {"stay_id": stay_id})
    if not rows:
        return AKIMonitoringBanner(
            prediction_level="정보없음",
            risk_level=RiskLevel(level="normal", label_kr="정보없음",
                                 color="#6b7280", border_color="#d1d5db"),
            primary_cause="데이터 없음", recommendation="검사 데이터를 확인해주세요.",
            detail_message=""
        )

    d          = rows[0]
    cr_max     = d.get("cr_max") or d.get("cr_mean") or 0
    bun_max    = d.get("bun_max") or 0
    egfr       = d.get("egfr_ckdepi") or 999
    rule_score = d.get("rule_based_score") or 0
    vaso_flag  = bool(d.get("vasopressor_flag", 0))
    map_hours  = d.get("map_below65_hours") or 0

    # Track D NLP 소견
    nlp_available    = not bool(d.get("nlp_missing", 1))
    nlp_fluid_burden = bool(d.get("nlp_fluid_burden_flag", 0))
    nlp_direct_renal = bool(d.get("nlp_direct_renal_flag", 0))
    kw_hydro         = bool(d.get("kw_hydronephrosis", 0))
    kw_fluid_ol      = bool(d.get("kw_fluid_overload", 0))
    kw_pulm_edema    = bool(d.get("kw_pulmonary_edema", 0))
    kw_pleural       = bool(d.get("kw_pleural_effusion", 0))
    kw_ascites       = bool(d.get("kw_ascites", 0))

    # AKI 원인 분석 — 임상 지표 + NLP 소견 통합
    causes = []
    if vaso_flag and map_hours > 2:
        causes.append("허혈/재관류 손상 의심")
    if d.get("bun_cr_ratio") and d["bun_cr_ratio"] > 20:
        causes.append("신전성 AKI 패턴")
    if egfr < 45:
        causes.append("eGFR 심각 저하")
    if cr_max > 2.0:
        causes.append("크레아티닌 심각 상승")

    # Track D NLP 기반 원인 추가 (방사선 판독 소견)
    if nlp_available:
        if kw_hydro:
            causes.append("방사선 판독: 수신증 (신후성 AKI)")
        if nlp_fluid_burden or (kw_fluid_ol and (kw_pulm_edema or kw_pleural)):
            causes.append("방사선 판독: 체액 과부하 복합 소견")
        elif kw_pulm_edema:
            causes.append("방사선 판독: 폐부종 확인")
        elif kw_pleural:
            causes.append("방사선 판독: 흉수 확인")
        if kw_ascites:
            causes.append("방사선 판독: 복수 (신증후군 의심)")

    primary_cause = " · ".join(causes) if causes else "지속 모니터링 필요"

    if rule_score >= 70:
        level, risk, rec = "높음", \
            RiskLevel(level="high", label_kr="높음", color="#ef4444", border_color="#ef4444"), \
            "즉시 모니터링 권고 — 신장내과 협진 고려"
    elif rule_score >= 40:
        level, risk, rec = "보통", \
            RiskLevel(level="warning", label_kr="보통", color="#f97316", border_color="#f97316"), \
            "집중 모니터링 권고 — 4~6h 내 재검"
    else:
        level, risk, rec = "낮음", \
            RiskLevel(level="normal", label_kr="낮음", color="#22c55e", border_color="#d1d5db"), \
            "정기 모니터링 유지"

    # 상세 설명 (배너 둘째 줄) — NLP 소견 포함
    nlp_note = ""
    if nlp_available and (kw_hydro or nlp_fluid_burden or kw_pulm_edema):
        nlp_parts = []
        if kw_hydro:         nlp_parts.append("수신증")
        if kw_pulm_edema:    nlp_parts.append("폐부종")
        if kw_pleural:       nlp_parts.append("흉수")
        if kw_ascites:       nlp_parts.append("복수")
        nlp_note = f" · 방사선 판독 소견({', '.join(nlp_parts)}) 포함"

    return AKIMonitoringBanner(
        prediction_level = level, risk_level = risk,
        primary_cause    = primary_cause, recommendation = rec,
        detail_message   = (
            f"크레아티닌 {cr_max:.1f} mg/dL, BUN {bun_max:.0f} mg/dL, "
            f"eGFR {egfr:.0f} mL/min 지표를 종합 분석하여 AKI 위험도를 자동 산출합니다.{nlp_note}"
        )
    )


def build_lab_monitoring_screen_response(stay_id: int) -> LabMonitoringResponse:
    """[SCR-04] 검사 결과 & AKI 모니터링 화면 전체 데이터 조립.

    날짜 탭·검사 결과 테이블·AKI 배너를 한 번의 호출로 반환한다.
    main.py의 GET /api/scr04/labs/{stay_id} 엔드포인트가 이 함수를 호출한다.
    """
    patient = execute_query(
        "SELECT age, gender FROM aki_project.cdss_master_features WHERE stay_id=:s",
        {"s": stay_id}
    )
    age    = patient[0]["age"]    if patient else 65
    gender = patient[0]["gender"] if patient else "M"

    lab_rows  = fetch_lab_results_with_date_tabs(stay_id)
    date_grps: dict[str, dict] = defaultdict(dict)
    item_map  = {50912: "creatinine", 51006: "bun", 51222: "hemoglobin"}
    for row in lab_rows:
        key = item_map.get(row["itemid"])
        if key:
            date_grps[str(row["lab_date"])][key] = {
                "value": row["valuenum"], "prev_value": row["prev_valuenum"]
            }

    META = {
        "creatinine": ("크레아티닌",   "mg/dL",  "< 1.0"),
        "bun":        ("BUN",          "mg/dL",  "7 – 20"),
        "egfr":       ("eGFR",         "mL/min", "> 60"),
        "hemoglobin": ("헤모글로빈",   "g/dL",   "12 – 16"),
    }

    sorted_dates = sorted(date_grps.keys(), reverse=True)[:3]
    today_str    = str(datetime.date.today())
    date_tabs    = []

    for date_str in sorted_dates:
        items   = date_grps[date_str]
        cr_val  = items.get("creatinine", {}).get("value")
        egfr_v  = derive_egfr_from_creatinine_and_demographics(cr_val, age, gender) if cr_val else None
        items["egfr"] = {"value": egfr_v, "prev_value": None}

        rows_out = []
        for key in ["creatinine", "bun", "egfr", "hemoglobin"]:
            v       = items.get(key, {}).get("value")
            prev_v  = items.get(key, {}).get("prev_value")
            name, unit, normal = META[key]
            status  = classify_lab_result_status_by_normal_range(key, v or 0)
            trend, arrow = calculate_trend_direction_from_sequential_values(v, prev_v)
            rows_out.append(LabResultRow(
                item_name=name, item_key=key, value=v, unit=unit,
                normal_range=normal, status=status,
                trend=trend, trend_arrow=arrow, prev_value=prev_v
            ))

        display_date = date_str[5:].replace("-", ".")
        is_today = (date_str == today_str) or (date_str == sorted_dates[0])
        date_tabs.append(DateTabData(date_str=display_date, is_today=is_today, lab_results=rows_out))

    return LabMonitoringResponse(
        stay_id         = stay_id,
        date_tabs       = date_tabs,
        current_results = date_tabs[0].lab_results if date_tabs else [],
        aki_banner      = build_aki_monitoring_summary_for_bottom_banner(stay_id),
    )


# ═══════════════════════════════════════════════════════════════════════════