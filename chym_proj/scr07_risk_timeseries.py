from typing import Optional
from db import execute_query

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# scr07_risk_timeseries.py  —  SCR-07 AKI 위험도 시계열 화면 백엔드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# [담당 화면] SCR-07 AI 기반 AKI 위험도 예측 지수 시계열
#
# [화면 레이아웃]
#   ┌──────────────────────────────────────────────────────────────────┐
#   │ AI 기반 AKI 위험도 예측 지수 (08:00 ~ 18:00)                   │
#   │ 100%┤                                                 ●88%      │
#   │  75%┤ - - - - - - - - - - - - - - - - - - - - ●85%             │
#   │  50%┤                                   ●65%                    │
#   │  25%┤              ●25%  ●42%                                   │
#   │   0%┤  ●15%                                                     │
#   │      08:00  10:00  12:00  14:00  16:00  18:00                   │
#   │ ──────────────────────────────────────────────────────────────── │
#   │ 14:00 이후 위험도 급상승 — 카디오 필터 프로토콜 즉시 적용 권고   │
#   └──────────────────────────────────────────────────────────────────┘
#
# [파일 분리 가이드]
#   ├── scr07_risk_timeseries.py   (현재) 시계열 생성·급상승 감지·배너
#   └── scr07_scheduler.py         1h 주기 스케줄러 분리 (APScheduler 사용 시)
#
# [SQL 변경 사항 반영]
#   - generate_series 기준을 prediction_cutoff → effective_cutoff 로 수정
#     (비AKI 환자의 prediction_cutoff=NULL → generate_series(NULL) = 빈 결과 버그 수정)
# ═══════════════════════════════════════════════════════════════════════════

HIGH_RISK_THRESHOLD = 75   # % 이상이면 빨간 점선 초과 → 배너 활성화
ALERT_ESCALATION_H  = 2    # h 연속 상승이면 "급상승" 판정


def generate_hourly_risk_score_timeseries_from_stored_data(
    stay_id: int, window_hours: int = 12
) -> list[dict]:
    """[SCR-07] 선 차트 데이터 — 1시간 단위 위험도 시계열 원시 데이터 생성

    effective_cutoff 기준으로 window_hours 이전부터 1시간 단위로 구간을 생성하고,
    각 구간에서 누적 크레아티닌·BUN·MAP 데이터를 추출해 위험도 계산의 원천을 반환한다.

    [화면 위치] SCR-07 선 차트 x축(시간) × y축(위험도%) 데이터 포인트
    [Bug Fix] prediction_cutoff → effective_cutoff
      비AKI 환자는 prediction_cutoff=NULL이므로
      DATE_TRUNC('hour', NULL) = NULL → generate_series(NULL, NULL) = 빈 결과.
      effective_cutoff는 항상 채워진 값이므로 이를 기준으로 사용.

    Args:
        stay_id:      ICU 체류 ID
        window_hours: 표시할 시간 범위 (기본 12h → 08:00~18:00)
    """
    sql = """
        WITH hourly_windows AS (
            SELECT generate_series(
                DATE_TRUNC('hour', m.effective_cutoff) - INTERVAL '1 hour' * :wh,
                DATE_TRUNC('hour', m.effective_cutoff),
                INTERVAL '1 hour'
            ) AS window_end
            FROM aki_project.cdss_master_features m
            WHERE m.stay_id = :stay_id
        ),
        hourly_map AS (
            SELECT
                DATE_TRUNC('hour', rv.charttime)           AS hour_ts,
                MIN(rv.map)                                AS map_min_hour,
                COUNT(CASE WHEN rv.map < 65 THEN 1 END)    AS low_map_count,
                COUNT(*)                                   AS total_map_count
            FROM aki_project.cdss_raw_map_values rv
            WHERE rv.stay_id = :stay_id
            GROUP BY DATE_TRUNC('hour', rv.charttime)
        ),
        hourly_labs AS (
            SELECT DISTINCT ON (DATE_TRUNC('hour', rl.charttime), rl.itemid)
                DATE_TRUNC('hour', rl.charttime) AS hour_ts,
                rl.itemid, rl.valuenum
            FROM aki_project.cdss_raw_lab_values rl
            WHERE rl.stay_id = :stay_id
            ORDER BY DATE_TRUNC('hour', rl.charttime), rl.itemid, rl.charttime DESC
        )
        SELECT
            hw.window_end AS timestamp,
            MAX(CASE WHEN hl.itemid = 50912 AND hl.hour_ts <= hw.window_end
                     THEN hl.valuenum END) AS cr_at_hour,
            MAX(CASE WHEN hl.itemid = 51006 AND hl.hour_ts <= hw.window_end
                     THEN hl.valuenum END) AS bun_at_hour,
            hm.map_min_hour,
            COALESCE(hm.low_map_count::FLOAT / NULLIF(hm.total_map_count,0), 0)
                                           AS ischemia_ratio_hour
        FROM hourly_windows  hw
        LEFT JOIN hourly_labs hl ON hl.hour_ts <= hw.window_end
        LEFT JOIN hourly_map  hm ON hm.hour_ts  = hw.window_end
        GROUP BY hw.window_end, hm.map_min_hour, hm.low_map_count, hm.total_map_count
        ORDER BY hw.window_end ASC
    """
    rows = execute_query(sql, {"stay_id": stay_id, "wh": window_hours})

    # Track D NLP 플래그 1회 조회 (정적 피처 — 시간별 변하지 않음)
    # 각 시간 포인트에 동일한 NLP 플래그를 주입해 점수 계산에 반영
    nlp_sql = """
        SELECT
            COALESCE(nlp_direct_renal_flag, 0) AS nlp_direct_renal_flag,
            COALESCE(nlp_fluid_burden_flag, 0) AS nlp_fluid_burden_flag
        FROM aki_project.cdss_master_features
        WHERE stay_id = :stay_id
    """
    nlp_rows = execute_query(nlp_sql, {"stay_id": stay_id})
    nlp_flags = nlp_rows[0] if nlp_rows else {"nlp_direct_renal_flag": 0, "nlp_fluid_burden_flag": 0}

    # 각 행에 NLP 플래그 주입
    for row in rows:
        row.update(nlp_flags)

    return rows



def _compute_hourly_rule_score(row: dict) -> int:
    """[SCR-07] 시계열 차트 데이터 포인트 점수 계산

    매 시간 포인트마다 호출되어 선 차트의 y값(0~100%)을 결정한다.
    Track D NLP 플래그는 정적 피처이므로 generate_hourly_* 에서
    DB 1회 조회 후 row에 주입되어 여기서 사용된다.

    [SCR-07 화면 연결]
    - 08:00~18:00 각 포인트 높이 결정
    - 75% 기준선 초과 여부 → 점 색상 빨간색 전환
    - NLP 소견이 있으면 점수가 높아져 더 일찍 기준선을 초과할 수 있음

    점수 구성 (합계 최대 100점):
      크레아티닌 > 1.5  → +30점
      BUN > 30          → +20점
      MAP < 65          → +15점
      허혈 비율 ≥ 50%   → +15점
      NLP 직접 신장 이상 → +10점  (kw_hydronephrosis·kw_aki_mention 등)
      NLP 체액 과부하   → + 5점  (kw_fluid_overload·kw_pulmonary_edema 2개 이상)
    """
    score          = 0
    cr_val         = row.get("cr_at_hour")
    bun_val        = row.get("bun_at_hour")
    map_val        = row.get("map_min_hour")
    ischemia_r     = row.get("ischemia_ratio_hour", 0)
    nlp_direct     = row.get("nlp_direct_renal_flag", 0)  # generate_hourly_*에서 주입
    nlp_fluid      = row.get("nlp_fluid_burden_flag", 0)

    if cr_val  and cr_val  > 1.5: score += 30
    if bun_val and bun_val > 30:  score += 20
    if map_val and map_val < 65:  score += 15
    if ischemia_r >= 0.5:         score += 15
    if nlp_direct:                score += 10  # Track D 직접 신장 이상 소견
    if nlp_fluid:                 score +=  5  # Track D 체액 과부하 복합 소견

    return min(score, 100)

def detect_risk_score_escalation_above_alert_threshold(data_points: list[dict]) -> dict:
    """[SCR-07] 하단 배너 활성화 조건 — 75% 초과 + 연속 상승 패턴 감지

    시계열 위험도 리스트에서 75% 기준선 최초 초과 시각과 이후 2h 연속 상승 여부를 확인한다.
    is_escalating=True이면 SCR-07 하단 경고 배너가 자동으로 활성화된다.

    [화면 위치] SCR-07 하단 파란 배너 활성 여부 결정
    [판정 기준]
      ① risk_pct >= 75% 인 최초 시점 탐색
      ② 이후 2h 구간에서 점수가 계속 상승하면 escalation_rate 계산
      ③ is_escalating = (초과 있음) AND (상승률 > 0)
    """
    if not data_points:
        return {"is_escalating": False, "first_exceed_time": None, "peak_risk_pct": 0, "escalation_rate": None}
    risk_values      = [d["risk_pct"] for d in data_points]
    peak             = max(risk_values)
    first_exceed     = None
    escalation_rate  = None
    for i, d in enumerate(data_points):
        if d["risk_pct"] >= HIGH_RISK_THRESHOLD:
            first_exceed = d.get("time_str")
            tail = data_points[i: i + ALERT_ESCALATION_H + 1]
            if len(tail) >= 2 and tail[-1]["risk_pct"] > tail[0]["risk_pct"]:
                escalation_rate = round((tail[-1]["risk_pct"] - tail[0]["risk_pct"]) / len(tail), 1)
            break
    return {
        "is_escalating":    first_exceed is not None and escalation_rate is not None,
        "first_exceed_time": first_exceed,
        "peak_risk_pct":    peak,
        "escalation_rate":  escalation_rate,
    }


def build_timeseries_alert_banner_message(escalation: dict, current_risk: int) -> dict:
    """[SCR-07] 하단 고정 경고 배너 메시지 생성

    급상승 감지 결과와 현재 위험도를 받아 배너 텍스트 2줄과 활성 여부를 반환한다.

    [화면 위치] SCR-07 최하단 파란 고정 박스
      is_active=False → 회색(비활성) / True → 파란·빨간 강조
      첫째 줄: "14:00 이후 위험도 급상승 — 카디오 필터 프로토콜 즉시 적용 권고"
      둘째 줄: "14:00 경과 시점부터 75%를 초과, 현재 88%에 도달하고 있습니다."
    """
    if not escalation.get("is_escalating"):
        return {
            "is_active": False, "trigger_time": None,
            "message":     "현재 위험도가 기준 이내입니다.",
            "sub_message": "지속적인 모니터링을 유지합니다.",
        }
    first_time = escalation.get("first_exceed_time", "해당 시점")
    rate_str   = f"시간당 {escalation['escalation_rate']:.1f}%p 상승" if escalation.get("escalation_rate") else ""
    return {
        "is_active":    True,
        "trigger_time": first_time,
        "message":      f"{first_time} 이후 위험도 급상승 — 카디오 필터 프로토콜 즉시 적용 권고",
        "sub_message":  (
            f"{first_time} 경과 시점부터 위험도가 {HIGH_RISK_THRESHOLD}%를 초과하며, "
            f"현재 {current_risk}%에 도달하고 있습니다. {rate_str}"
        ),
    }

def build_risk_timeseries_screen_response(
    stay_id: int,
    window_hours: int = 12,
) -> dict:
    """[SCR-07] AKI 위험도 시계열 화면 전체 데이터 조립

    SCR-07 화면 렌더링에 필요한 모든 데이터를 한 번에 반환한다.
    main.py의 GET /api/scr07/timeseries/{stay_id} 엔드포인트가 이 함수를 호출한다.

    [실시간 서빙]
    스케줄러가 1시간마다 이 함수를 호출해 Redis 캐시에 저장,
    프론트엔드는 캐시에서 조회해 차트를 갱신한다.

    [반환 구조]
      data_points   → 선 차트 x·y 데이터 (time_str, risk_pct, point_color)
      chart_config  → 차트 설정 (75% 기준선, 색상 등)
      escalation    → 급상승 감지 결과
      bottom_banner → 경고 배너 ("14:00 이후 위험도 급상승")
      current_risk  → 가장 최근 위험도
    """
    raw_series = generate_hourly_risk_score_timeseries_from_stored_data(
        stay_id, window_hours
    )

    data_points = []
    for row in raw_series:
        ts       = row.get("timestamp")
        risk_pct = _compute_hourly_rule_score(row)
        is_above = risk_pct >= HIGH_RISK_THRESHOLD
        time_str = ts.strftime("%H:%M") if hasattr(ts, "strftime") else str(ts)[:16]

        data_points.append({
            "time_str":           time_str,
            "timestamp":          str(ts),
            "risk_pct":           risk_pct,
            "is_above_threshold": is_above,
            "point_color":        "#ef4444" if is_above else "#3b82f6",
        })

    escalation    = detect_risk_score_escalation_above_alert_threshold(data_points)
    current_risk  = data_points[-1]["risk_pct"] if data_points else 0
    bottom_banner = build_timeseries_alert_banner_message(escalation, current_risk)

    return {
        "stay_id":      stay_id,
        "data_points":  data_points,
        "chart_config": {
            "x_axis_labels":   [dp["time_str"] for dp in data_points],
            "y_axis_max":      100,
            "threshold_line":  HIGH_RISK_THRESHOLD,
            "threshold_label": f"고위험 구간 (≥{HIGH_RISK_THRESHOLD}%)",
            "line_color":      "#3b82f6",
            "danger_bg_color": "#fee2e2",
        },
        "escalation":   escalation,
        "bottom_banner":bottom_banner,
        "current_risk": current_risk,
    }