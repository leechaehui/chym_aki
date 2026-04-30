"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
scr03_drug_management.py  —  SCR-03 처방 약물 관리 화면 백엔드
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[담당 화면] SCR-03 처방 약물 관리

[화면 레이아웃]
  ┌────────────────────────────────────────────────────────────────────┐
  │ 현재 처방 N건                                      [+ 처방 추가]   │
  │ ─────────────────────────────────────────────────────────────────  │
  │  약물명         투여경로  용량      상태      비고                 │
  │  Vancomycin     IV        1g q12h  ⚠ 주의   신독성 모니터링 필요  │
  │  Pip/Tazo       IV        4.5g q8h ⚠ 주의   Vanco 병용 시 3.7배↑ │
  │  Furosemide     IV        40mg bid ✓ 유지   이뇨제 탈수 모니터링  │
  │  Tacrolimus     PO        2mg bid  ✓ 유지   면역억제 효과 모니터링│
  │ ─────────────────────────────────────────────────────────────────  │
  │ [AI 신독성 모니터링]  위험도: 높음  |  Vanco+Pip/Tazo 병용 중단  │
  │  크레아티닌 2.1↑ | Vancomycin 72h 투여 중 | 신독성 위험도 자동산출│
  └────────────────────────────────────────────────────────────────────┘

[파일 분리 가이드]
  이 파일 하나로 SCR-03 전체를 담당한다.
  아래 기준으로 추가 분리 가능:
  ├── scr03_drug_management.py     (현재) 조회·집계·메시지 생성 (Pure Python)
  ├── scr03_drug_queries.py        DB 쿼리 문자열만 모아두는 모듈 (선택)
  └── scr03_drug_schemas.py        Pydantic 모델만 분리 (규모 커질 때)

[DB 연결]
  - cdss_nephrotoxic_rx_raw    현재 활성 처방 조회
  - cdss_master_features       burden_score·drug_risk_score 조회
  - cdss_nephrotoxic_combo_risk 위험 조합 플래그 조회
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from typing import Optional
from pydantic import BaseModel
from db import execute_query, classify_value_as_risk_level, RiskLevel


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic 모델  —  SCR-03 응답 스키마
# ─────────────────────────────────────────────────────────────────────────────

class PrescriptionRow(BaseModel):
    """SCR-03 처방 목록 테이블 1행 스키마.

    화면의 가로 컬럼 5개(약물명·경로·용량·상태·비고)에 1:1 대응한다.
    is_nephrotoxic=True이면 좌측 주황 보더와 ⚠ 아이콘을 렌더링한다.
    """
    drug_name:      str
    route:          str
    dose:           Optional[str]
    status:         RiskLevel        # 주의(주황) | 유지(초록)
    note:           str
    is_nephrotoxic: bool


class DrugCombinationAlert(BaseModel):
    """SCR-03 '+ 처방 추가' 팝업의 교차 분석 경고 1건 스키마.

    is_triggered=True인 항목만 팝업 상단 빨간 박스에 강조 표시된다.
    """
    combo_type:   str    # "vanco_piptazo" | "triple_whammy" 등
    description:  str    # 한국어 경고 문구
    severity:     str    # "high" | "medium"
    is_triggered: bool


class AINephrotoxicityAlert(BaseModel):
    """SCR-03 하단 고정 AI 신독성 모니터링 배너 스키마.

    overall_risk 와 risk_level.color 로 배너 배경색을 결정한다.
    recommendation 은 배너 첫 줄 강조 텍스트,
    detail_message 는 두 번째 줄 보조 설명 텍스트로 표시된다.
    """
    overall_risk:    str
    risk_level:      RiskLevel
    burden_score:    int
    drug_risk_score: int
    primary_drug:    Optional[str]
    recommendation:  str
    detail_message:  str


class DrugManagementResponse(BaseModel):
    """SCR-03 전체 화면 단일 응답 스키마."""
    stay_id:                  int
    prescriptions:            list[PrescriptionRow]
    combination_alerts:       list[DrugCombinationAlert]
    ai_alert:                 AINephrotoxicityAlert
    total_prescription_count: int


# ─────────────────────────────────────────────────────────────────────────────
# 약물별 정적 메타데이터  (화면 "상태"·"비고" 열 값 결정)
# ─────────────────────────────────────────────────────────────────────────────
DRUG_METADATA: dict[str, dict] = {
    "vancomycin":   {"nephrotoxic": True,  "note": "신독성 모니터링 필요",       "category": "항생제"},
    "piperacillin": {"nephrotoxic": True,  "note": "Vanco 병용 시 AKI 3.7배↑",  "category": "항생제"},
    "zosyn":        {"nephrotoxic": True,  "note": "Vanco 병용 시 AKI 3.7배↑",  "category": "항생제"},
    "gentamicin":   {"nephrotoxic": True,  "note": "아미노글리코사이드 신독성",   "category": "항생제"},
    "tobramycin":   {"nephrotoxic": True,  "note": "아미노글리코사이드 신독성",   "category": "항생제"},
    "amikacin":     {"nephrotoxic": True,  "note": "아미노글리코사이드 신독성",   "category": "항생제"},
    "amphotericin": {"nephrotoxic": True,  "note": "강력한 신독성 — 철저 모니터링","category": "항진균제"},
    "ketorolac":    {"nephrotoxic": True,  "note": "NSAIDs — 신혈류 감소",        "category": "진통제"},
    "ibuprofen":    {"nephrotoxic": True,  "note": "NSAIDs — 신혈류 감소",        "category": "진통제"},
    "lisinopril":   {"nephrotoxic": True,  "note": "ACEi — GFR 감소 주의",        "category": "혈압약"},
    "losartan":     {"nephrotoxic": True,  "note": "ARB — GFR 감소 주의",         "category": "혈압약"},
    "furosemide":   {"nephrotoxic": False, "note": "이뇨제 — 탈수 모니터링",      "category": "이뇨제"},
    "lasix":        {"nephrotoxic": False, "note": "이뇨제 — 탈수 모니터링",      "category": "이뇨제"},
    "tacrolimus":   {"nephrotoxic": False, "note": "면역억제 효과 모니터링",       "category": "면역억제제"},
    "cyclosporine": {"nephrotoxic": True,  "note": "칼시뉴린억제제 신독성",        "category": "면역억제제"},
    "metformin":    {"nephrotoxic": False, "note": "신기능 저하 시 금기",          "category": "혈당강하제"},
}


# ─────────────────────────────────────────────────────────────────────────────
# 함수 구현
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_current_prescriptions_for_display(stay_id: int) -> list[PrescriptionRow]:
    """[SCR-03] 처방 목록 테이블 — ICU 현재 활성 처방 목록 조회

    현재 이 환자에게 투여 중인 신독성 관련 약물을 DB에서 조회하고,
    각 약물의 신독성 상태(주의/유지)·비고를 부착해 화면 테이블 형식으로 반환한다.

    [화면 위치] SCR-03 중앙 테이블 (약물명·경로·용량·상태·비고 5열)
    [정렬 기준] 신독성 약물을 최상위 행에 배치 (주황 보더로 즉시 식별)
    [활성 처방 조건] starttime <= NOW() AND (stoptime IS NULL OR stoptime > NOW())
      ← effective_cutoff 대신 NOW() 사용. effective_cutoff는 학습용 역사적 컷오프이므로
         현재 CDSS 실시간 화면에서는 "지금 이 순간" 기준을 써야 한다.

    Args:
        stay_id: ICU 체류 ID

    Returns:
        PrescriptionRow 리스트 (신독성 약물 우선 정렬)
    """
    sql = """
        SELECT
            p.drug,
            p.route,
            p.dose_val_rx,
            p.dose_unit_rx,
            p.starttime,
            p.stoptime
        FROM aki_project.cdss_nephrotoxic_rx_raw p
        JOIN aki_project.cdss_cohort_window       c ON p.subject_id = c.subject_id
        WHERE c.stay_id    = :stay_id
          AND p.starttime >= c.icu_intime     -- ICU 입실 이후 처방만
          AND p.starttime <= NOW()            -- 미래 예약 처방 제외
          -- 현재 활성 처방: 아직 종료되지 않은 처방
          -- stoptime=NULL이면 계속 투여 중, stoptime > NOW()이면 아직 진행 중
          AND (p.stoptime IS NULL OR p.stoptime > NOW())
        ORDER BY p.starttime DESC
    """
    rows = execute_query(sql, {"stay_id": stay_id})

    result: list[PrescriptionRow] = []
    for row in rows:
        drug_lower = row["drug"].lower()
        meta = next(
            (v for k, v in DRUG_METADATA.items() if k in drug_lower),
            {"nephrotoxic": False, "note": "정기 모니터링", "category": "기타"}
        )
        status = (
            RiskLevel(level="warning", label_kr="주의",
                      color="#f97316", border_color="#f97316")
            if meta["nephrotoxic"] else
            RiskLevel(level="normal", label_kr="유지",
                      color="#22c55e", border_color="#d1d5db")
        )
        dose_str = (
            f"{row['dose_val_rx']} {row['dose_unit_rx']}"
            if row.get("dose_val_rx") else "용량 미기재"
        )
        result.append(PrescriptionRow(
            drug_name      = row["drug"],
            route          = row.get("route") or "미기재",
            dose           = dose_str,
            status         = status,
            note           = meta["note"],
            is_nephrotoxic = meta["nephrotoxic"],
        ))

    # 신독성 약물을 상단 정렬 (화면 설계: 위험 약물 최상위)
    result.sort(key=lambda x: (not x.is_nephrotoxic, x.drug_name))
    return result


def detect_dangerous_drug_combination_patterns(stay_id: int) -> list[DrugCombinationAlert]:
    """[SCR-03] + 처방 추가 팝업 — 기존 처방과의 위험 조합 감지

    현재 처방 중인 약물 조합이 고위험 패턴(Vanco+Pip, Triple Whammy 등)에 해당하는지
    DB 플래그를 조회해 팝업 경고 목록으로 반환한다.

    [화면 위치] SCR-03 '+ 처방 추가' 클릭 시 팝업 상단 교차 분석 경고 박스
    [표시 방식] is_triggered=True인 항목은 빨간 배경으로 강조,
               False 항목은 회색 배경으로 "해당 없음" 표시

    Args:
        stay_id: ICU 체류 ID

    Returns:
        DrugCombinationAlert 리스트 (발동된 위험 조합 최상위 정렬)
    """
    sql = """
        SELECT
            vanco_piptazo_combo, vanco_aminogly_combo,
            vanco_carbapenem_combo, nsaid_acei_combo,
            triple_whammy, diuretic_overload_flag, metformin_risk_flag,
            nephrotoxic_burden_score, drug_risk_score
        FROM aki_project.cdss_master_features
        WHERE stay_id = :stay_id
    """
    rows = execute_query(sql, {"stay_id": stay_id})
    if not rows:
        return []
    r = rows[0]

    patterns = [
        {
            "combo_type":  "vanco_piptazo",
            "description": "Vancomycin + Piperacillin/Tazobactam 병용 — AKI 3.7배↑ (2019 ASHP 메타분석)",
            "severity":    "high",
            "triggered":   bool(r.get("vanco_piptazo_combo", 0)),
        },
        {
            "combo_type":  "vanco_aminoglycoside",
            "description": "Vancomycin + 아미노글리코사이드 병용 — 신독성 상가 작용",
            "severity":    "high",
            "triggered":   bool(r.get("vanco_aminogly_combo", 0)),
        },
        {
            "combo_type":  "triple_whammy",
            "description": "NSAIDs + ACEi/ARB + 이뇨제 3중 병용 — 신혈류 3중 차단, 최고 위험",
            "severity":    "high",
            "triggered":   bool(r.get("triple_whammy", 0)),
        },
        {
            "combo_type":  "nsaid_acei",
            "description": "NSAIDs + ACEi/ARB 병용 — 신혈류 이중 감소",
            "severity":    "medium",
            "triggered":   bool(r.get("nsaid_acei_combo", 0)),
        },
        {
            "combo_type":  "diuretic_overload",
            "description": "이뇨제 누적 >200mg + 신독성 항생제 — 탈수로 약물 농도↑",
            "severity":    "medium",
            "triggered":   bool(r.get("diuretic_overload_flag", 0)),
        },
    ]

    alerts = [DrugCombinationAlert(
        combo_type   = p["combo_type"],
        description  = p["description"],
        severity     = p["severity"],
        is_triggered = p["triggered"],
    ) for p in patterns]

    # 발동된 고위험 조합을 최상단에 배치
    alerts.sort(key=lambda x: (not x.is_triggered, x.severity != "high"))
    return alerts


def generate_ai_nephrotoxicity_monitoring_message(stay_id: int) -> AINephrotoxicityAlert:
    """[SCR-03] AI 신독성 모니터링 배너 — 위험도 종합 평가 및 권고 메시지 생성

    환자의 신기능 지표(크레아티닌)와 처방 약물 부담을 종합 분석하여
    SCR-03 하단 고정 파란 배너에 표시할 위험도와 권고 문구를 생성한다.

    [화면 위치] SCR-03 최하단 고정 파란 박스 (스크롤에 관계없이 항상 노출)
      첫째 줄: "AI 신독성 모니터링 — 위험도: 높음  |  Vanco+Pip/Tazo 병용 중단 권고"
      둘째 줄: "크레아티닌 2.1↑ | Vancomycin 72h 투여 중 | 신독성 위험도 자동 산출"

    [위험도 판정 기준]
      burden_score >= 3 또는 drug_risk_score >= 2 → 높음(빨강)
      burden_score >= 1 또는 drug_risk_score >= 1 → 보통(주황)
      그 외 → 낮음(초록)

    Args:
        stay_id: ICU 체류 ID

    Returns:
        AINephrotoxicityAlert (위험도 등급 + 권고 문구 + 상세 설명)
    """
    sql = """
        SELECT
            nephrotoxic_burden_score,
            drug_risk_score,
            cr_max, cr_mean,
            vancomycin_exposure_hours,
            vancomycin_rx,
            vanco_piptazo_combo,
            -- Track D NLP: 방사선 판독 소견 (SCR-03 AI 배너 경고 강화에 사용)
            COALESCE(kw_hydronephrosis,     0) AS kw_hydronephrosis,
            COALESCE(kw_rad_hydronephrosis, 0) AS kw_rad_hydronephrosis,
            COALESCE(kw_contrast_agent,     0) AS kw_contrast_agent,
            COALESCE(nlp_direct_renal_flag, 0) AS nlp_direct_renal_flag,
            COALESCE(nlp_missing,           1) AS nlp_missing
        FROM aki_project.cdss_master_features
        WHERE stay_id = :stay_id
    """
    rows = execute_query(sql, {"stay_id": stay_id})
    if not rows:
        return AINephrotoxicityAlert(
            overall_risk="정보없음",
            risk_level=RiskLevel(level="normal", label_kr="정보없음",
                                 color="#6b7280", border_color="#d1d5db"),
            burden_score=0, drug_risk_score=0,
            primary_drug=None,
            recommendation="처방 데이터를 조회할 수 없습니다.",
            detail_message=""
        )

    d               = rows[0]
    burden_score    = int(d.get("nephrotoxic_burden_score") or 0)
    drug_risk_score = int(d.get("drug_risk_score") or 0)
    cr_value        = float(d.get("cr_max") or d.get("cr_mean") or 0)
    vanco_hours     = float(d.get("vancomycin_exposure_hours") or 0)
    vanco_on        = bool(d.get("vancomycin_rx", 0))
    vanco_pip       = bool(d.get("vanco_piptazo_combo", 0))

    # Track D NLP 소견
    kw_hydro        = bool(d.get("kw_hydronephrosis", 0) or d.get("kw_rad_hydronephrosis", 0))
    kw_contrast     = bool(d.get("kw_contrast_agent", 0))
    nlp_renal       = bool(d.get("nlp_direct_renal_flag", 0))
    nlp_available   = not bool(d.get("nlp_missing", 1))

    # 가장 위험한 약물 결정 (배너 첫 줄 강조)
    primary_drug = (
        "Vancomycin + Piperacillin/Tazobactam" if vanco_pip else
        "Vancomycin" if (vanco_on and cr_value > 1.5) else
        None
    )

    # 위험도 등급 결정
    # NLP에서 수신증 또는 직접 신장 이상 언급이 있으면 burden_score 1점 가산
    nlp_bonus = 1 if (kw_hydro or nlp_renal) else 0
    if burden_score + nlp_bonus >= 3 or drug_risk_score >= 2:
        overall_risk = "높음"
        risk_level   = RiskLevel(level="high", label_kr="높음",
                                 color="#ef4444", border_color="#ef4444")
    elif burden_score >= 1 or drug_risk_score >= 1:
        overall_risk = "보통"
        risk_level   = RiskLevel(level="warning", label_kr="보통",
                                 color="#f97316", border_color="#f97316")
    else:
        overall_risk = "낮음"
        risk_level   = RiskLevel(level="normal", label_kr="낮음",
                                 color="#22c55e", border_color="#d1d5db")

    # 권고 문구 (배너 첫 줄 오른쪽)
    recommendation = (
        "Vanco+Pip/Tazo 병용 중단 또는 대체 항생제 검토 권고" if vanco_pip else
        f"Vancomycin 감량 검토 권고 (Cr {cr_value:.1f} ↑)" if (vanco_on and cr_value > 1.5) else
        "신독성 약물 부담 과다 — 불필요 약물 중단 검토" if burden_score >= 3 else
        "현재 즉각 위험 없음 — 지속 모니터링"
    )

    # 상세 설명 (배너 둘째 줄)
    # Track D NLP 소견이 있으면 방사선 판독 결과를 배너 설명에 포함
    parts = [f"크레아티닌 {cr_value:.1f} {'↑' if cr_value > 1.0 else '정상'}"]
    if vanco_on:
        parts.append(f"Vancomycin {vanco_hours:.0f}h 투여 중")
    if nlp_available:
        if kw_hydro:
            parts.append("방사선 판독: 수신증 소견 확인 (신후성 AKI 위험)")
        if kw_contrast:
            parts.append("방사선 판독: 조영제 투여 이력 (CI-AKI 위험)")
    parts.append("신독성 위험도 자동 산출")

    return AINephrotoxicityAlert(
        overall_risk    = overall_risk,
        risk_level      = risk_level,
        burden_score    = burden_score,
        drug_risk_score = drug_risk_score,
        primary_drug    = primary_drug,
        recommendation  = recommendation,
        detail_message  = " | ".join(parts),
    )


def build_drug_management_screen_response(stay_id: int) -> DrugManagementResponse:
    """[SCR-03] 처방 약물 관리 화면 전체 데이터 조립

    SCR-03 화면 렌더링에 필요한 모든 데이터를 한 번에 반환한다.
    main.py의 GET /api/scr03/drug/{stay_id} 엔드포인트가 이 함수를 호출한다.

    [반환 구조]
      prescriptions      → 처방 목록 테이블
      combination_alerts → + 처방 추가 팝업 교차 경고
      ai_alert           → 하단 AI 신독성 모니터링 배너
    """
    return DrugManagementResponse(
        stay_id                  = stay_id,
        prescriptions            = retrieve_current_prescriptions_for_display(stay_id),
        combination_alerts       = detect_dangerous_drug_combination_patterns(stay_id),
        ai_alert                 = generate_ai_nephrotoxicity_monitoring_message(stay_id),
        total_prescription_count = len(retrieve_current_prescriptions_for_display(stay_id)),
    )