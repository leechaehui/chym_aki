"""병리/병상상세/예약/알림 시드 (프론트 mock 과 동일 데이터).

seed() 트랜잭션 내부에서 호출된다(commit 은 호출자 책임).
"""
import json
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from models.base import new_id
from models.bed_detail import BedDetail
from models.consultation import ConsultEvent, Consultation
from models.notification import Notification
from models.pathology import PathologyResult


# ----------------------------------------------------------
# 병리 결과 (mock/pathology.ts)
# ----------------------------------------------------------
def _layers(filled: bool) -> list[dict]:
    c = lambda v: (v if filled else None)  # noqa: E731
    return [
        {"key": "cortex", "label": "Cortex", "color": "rgba(100, 116, 139, 0.35)", "count": c(1), "visible": True},
        {"key": "glomeruli", "label": "Glomeruli", "color": "rgba(34, 197, 94, 0.4)", "count": c(18), "visible": True},
        {"key": "gs", "label": "Globally Sclerotic", "color": "rgba(220, 38, 38, 0.5)", "count": c(3), "visible": True},
        {"key": "tubules", "label": "Tubules", "color": "rgba(37, 99, 235, 0.4)", "count": c(1240), "visible": False},
        {"key": "arteries", "label": "Arteries", "color": "rgba(168, 85, 247, 0.4)", "count": c(12), "visible": False},
        {"key": "ptc", "label": "Peritubular Capillaries", "color": "rgba(20, 184, 166, 0.4)", "count": c(860), "visible": False},
        {"key": "ifta", "label": "IFTA", "color": "rgba(245, 158, 11, 0.4)", "count": c(1), "visible": False},
    ]


def _metrics(filled: bool, gs: int, total: int) -> list[dict]:
    ratio = round(gs / total * 1000) / 10 if filled else None
    return [
        {"key": "glomCount", "label": "사구체 개수", "value": (total if filled else None), "unit": "개"},
        {"key": "gsCount", "label": "경화 사구체 개수", "value": (gs if filled else None), "unit": "개"},
        {"key": "gsRatio", "label": "경화 비율", "value": ratio, "unit": "%"},
        {"key": "tubuleDensity", "label": "세뇨관 밀도", "value": (78 if filled else None), "unit": "/mm²"},
        {"key": "vesselDensity", "label": "혈관 밀도", "value": (9 if filled else None), "unit": "/mm²"},
        {"key": "ptcDensity", "label": "PTC 밀도", "value": (312 if filled else None), "unit": "/mm²"},
        {"key": "iftaRatio", "label": "IFTA 비율", "value": (15 if filled else None), "unit": "%"},
        {"key": "mesangial", "label": "Mesangial Fraction", "value": (0.21 if filled else None), "unit": ""},
        {"key": "tbm", "label": "TBM Thickness", "value": (412 if filled else None), "unit": "nm"},
        {"key": "luminal", "label": "Luminal Fraction", "value": (0.34 if filled else None), "unit": ""},
    ]


def _pathology() -> list[PathologyResult]:
    return [
        PathologyResult(
            consult_id="c-002", stain="PAS", image_url=None,
            layers_json=json.dumps(_layers(True), ensure_ascii=False),
            metrics_json=json.dumps(_metrics(True, 6, 22), ensure_ascii=False),
            report_findings="", report_diagnosis="", report_status="draft", report_updated_at=None,
        ),
        PathologyResult(
            consult_id="c-001", stain="PAS", image_url=None,
            layers_json=json.dumps(_layers(False), ensure_ascii=False),
            metrics_json=json.dumps(_metrics(False, 0, 0), ensure_ascii=False),
            report_findings="", report_diagnosis="", report_status="draft", report_updated_at=None,
        ),
    ]


# ----------------------------------------------------------
# 병상 입실 상세 (mock/beds.ts mockBedDetails)
# ----------------------------------------------------------
def _bed_details() -> list[BedDetail]:
    def d(bed_id, diagnosis, attending, admitted_at, aki_risk, meds, labs,
          aki_stage=None, recent=None, report=None, items=None):
        return BedDetail(
            bed_id=bed_id, diagnosis=diagnosis, attending=attending, admitted_at=admitted_at,
            aki_risk=aki_risk, aki_stage=aki_stage,
            medications_json=json.dumps(meds, ensure_ascii=False),
            labs_json=json.dumps(labs, ensure_ascii=False),
            recent_inputs_json=json.dumps(recent, ensure_ascii=False) if recent else None,
            treatment_report=report,
            treatment_items_json=json.dumps(items, ensure_ascii=False) if items else None,
        )

    return [
        d("ER-8", "패혈증 의증, 급성 신손상(AKI)", "김도윤", "2026-06-14T08:05:00", True,
          [
              {"name": "생리식염수 (0.9% NaCl)", "dose": "1,000 mL", "route": "IV", "status": "투여중"},
              {"name": "Piperacillin/Tazobactam", "dose": "4.5 g q8h", "route": "IV", "status": "투여중"},
              {"name": "Furosemide", "dose": "40 mg", "route": "IV", "status": "1회 투여 (반응 없음)"},
          ],
          [
              {"label": "Creatinine", "value": "3.8", "unit": "mg/dL", "flag": "high"},
              {"label": "eGFR", "value": "14", "unit": "mL/min", "flag": "low"},
              {"label": "BUN", "value": "62", "unit": "mg/dL", "flag": "high"},
              {"label": "Potassium", "value": "6.4", "unit": "mmol/L", "flag": "high"},
              {"label": "pH", "value": "7.21", "unit": "", "flag": "low"},
              {"label": "Lactate", "value": "4.2", "unit": "mmol/L", "flag": "high"},
          ],
          aki_stage="AKI Stage 3 (KDIGO)",
          recent=[
              {"label": "혈압", "value": "168/95 mmHg"},
              {"label": "맥박", "value": "112 bpm"},
              {"label": "체온", "value": "39.4 ℃"},
              {"label": "SpO₂", "value": "94 %"},
              {"label": "소변량(6h)", "value": "15 mL (무뇨)"},
              {"label": "투여 수액", "value": "생리식염수 1L (누적 2.5L)"},
          ],
          report="Cr 3.8 mg/dL·eGFR 14·K⁺ 6.4 mmol/L, 6시간 무뇨 및 대사성 산증(pH 7.21) 동반으로 KDIGO AKI Stage 3에 해당합니다. 이뇨제 무반응 + 고칼륨혈증으로 응급 혈액투석(HD)이 필요합니다.",
          items=[
              "응급 혈액투석(HD) 준비 및 신장내과 협진",
              "고칼륨혈증 교정: 칼슘글루코네이트 + 인슐린/포도당",
              "수액 반응 재평가 (과부하 주의)",
              "시간당 소변량·전해질 1시간 간격 모니터링",
          ]),
        d("ICU-2", "패혈성 쇼크, 핍뇨성 급성 신손상", "이하람", "2026-06-13T22:40:00", True,
          [
              {"name": "Norepinephrine", "dose": "0.18 mcg/kg/min", "route": "IV", "status": "투여중"},
              {"name": "Meropenem", "dose": "1 g q8h", "route": "IV", "status": "투여중"},
              {"name": "생리식염수", "dose": "유지 80 mL/hr", "route": "IV", "status": "투여중"},
          ],
          [
              {"label": "Creatinine", "value": "4.6", "unit": "mg/dL", "flag": "high"},
              {"label": "eGFR", "value": "11", "unit": "mL/min", "flag": "low"},
              {"label": "Potassium", "value": "5.9", "unit": "mmol/L", "flag": "high"},
              {"label": "Lactate", "value": "5.8", "unit": "mmol/L", "flag": "high"},
              {"label": "HCO₃⁻", "value": "14", "unit": "mmol/L", "flag": "low"},
          ],
          aki_stage="AKI Stage 3 (KDIGO)",
          recent=[
              {"label": "혈압(MAP)", "value": "62 mmHg (승압 중)"},
              {"label": "맥박", "value": "124 bpm"},
              {"label": "소변량(6h)", "value": "40 mL (핍뇨)"},
              {"label": "CVP", "value": "12 mmHg"},
              {"label": "투여 수액", "value": "누적 4.0L"},
          ],
          report="승압제 의존 패혈성 쇼크에 핍뇨성 AKI(Cr 4.6·eGFR 11)와 젖산산증(Lactate 5.8) 동반. 보존적 치료로 교정되지 않아 지속적 신대체요법(CRRT) 적용을 고려해야 합니다.",
          items=[
              "지속적 신대체요법(CRRT) 적용 검토 — 신장내과 협진",
              "고칼륨혈증·산증 교정",
              "승압제 유지 하 용적 상태 재평가",
          ]),
        d("W7-7", "말기신부전, 유지 혈액투석 중", "박현우", "2026-06-12T10:00:00", True,
          [
              {"name": "Sevelamer", "dose": "800 mg tid", "route": "PO", "status": "투여중"},
              {"name": "Calcium polystyrene sulfonate", "dose": "15 g", "route": "PO", "status": "1회 투여"},
              {"name": "EPO (Darbepoetin)", "dose": "40 mcg/week", "route": "SC", "status": "투여중"},
          ],
          [
              {"label": "Creatinine", "value": "8.9", "unit": "mg/dL", "flag": "high"},
              {"label": "Potassium", "value": "6.1", "unit": "mmol/L", "flag": "high"},
              {"label": "Phosphorus", "value": "6.8", "unit": "mg/dL", "flag": "high"},
              {"label": "Hb", "value": "8.9", "unit": "g/dL", "flag": "low"},
          ],
          aki_stage="ESRD on HD — 투석 간 고칼륨",
          recent=[
              {"label": "혈압", "value": "152/88 mmHg"},
              {"label": "체중", "value": "건체중 +2.4 kg"},
              {"label": "마지막 투석", "value": "2일 전"},
              {"label": "소변량(24h)", "value": "200 mL"},
          ],
          report="유지 혈액투석 환자로 투석 간 고칼륨혈증(K⁺ 6.1)과 용적 과부하(+2.4kg) 소견. 예정 외 추가 투석 일정 조정이 필요합니다.",
          items=["추가 혈액투석 일정 조정 — 신장내과 협진", "고칼륨혈증 응급 교정", "수분/염분 제한 교육"]),
        d("ER-2", "우하복부 통증, 의증 충수염", "김도윤", "2026-06-14T07:48:00", False,
          [
              {"name": "Hartmann 용액", "dose": "1,000 mL", "route": "IV", "status": "투여중"},
              {"name": "Ketorolac", "dose": "30 mg", "route": "IV", "status": "1회 투여"},
              {"name": "Ceftriaxone", "dose": "2 g", "route": "IV", "status": "투여중"},
          ],
          [
              {"label": "WBC", "value": "14.2", "unit": "10³/µL", "flag": "high"},
              {"label": "CRP", "value": "8.6", "unit": "mg/dL", "flag": "high"},
              {"label": "Creatinine", "value": "0.8", "unit": "mg/dL", "flag": "normal"},
          ]),
        d("ICU-1", "급성 신우신염, 회복기", "이하람", "2026-06-13T15:20:00", False,
          [
              {"name": "Ciprofloxacin", "dose": "400 mg q12h", "route": "IV", "status": "투여중"},
              {"name": "생리식염수", "dose": "유지 60 mL/hr", "route": "IV", "status": "투여중"},
          ],
          [
              {"label": "Creatinine", "value": "1.2", "unit": "mg/dL", "flag": "normal"},
              {"label": "WBC", "value": "9.8", "unit": "10³/µL", "flag": "normal"},
              {"label": "체온", "value": "37.1", "unit": "℃", "flag": "normal"},
          ]),
    ]




# ----------------------------------------------------------
# 알림 시드 (mock/notifications.ts seedNotifications)
# ----------------------------------------------------------
def _notifications() -> list[Notification]:
    rows = [
        ("ACTION_REQUIRED", "nephrology", "협진 회신 확인", "정민호(AKI-100231) 병리 협진 회신이 등록되었습니다.", True, "/nephrology?patient=AKI-100231"),
        ("ACTION_REQUIRED", "pathology", "신규 협진 요청", "오태윤(AKI-100258) 협진 요청이 대기 중입니다.", True, "/pathology?consult=c-002"),
        ("ACTION_REQUIRED", "admin", "새 가입 요청", "박하은(신장내과) 외 1건의 계정 승인 요청이 있습니다.", False, "/admin"),
    ]
    return [
        Notification(
            id=new_id("noti"), severity=s, department=dept, title=t, message=m, read=r, link=link,
        )
        for s, dept, t, m, r, link in rows
    ]


# ----------------------------------------------------------
# 병리 협진 (병리 결과 c-001~c-002 과 1:1 연결 — 병리과 판독 워크리스트)
# ----------------------------------------------------------
def _consultations() -> list[Consultation]:
    now = datetime.now(timezone.utc)

    def evt(stage, label, actor, at):
        return ConsultEvent(id=new_id("t"), stage=stage, label=label, at=at.isoformat(), actor=actor)

    # 실제 로스터 환자(chym.patients)로 판독 워크리스트를 구성한다. MRN 은 실제 subject_id 기준
    # (AKI-{subject_id})이라, 데모 Setup 후 canonical_patient_name 이 현재 표시명으로 해소한다.
    # c-001: 신규 요청(미분석) — 급성 악화 트리거 주인공(오준현 · subject 10218191)
    c1 = Consultation(
        id="c-001", kind="pathology",
        patient_mrn="AKI-10218191", patient_name="오준현",
        diagnosis="급성 세뇨관괴사 의증, 고칼륨혈증 동반", key_labs="Cr 3.6 · K 6.4 · eGFR 14",
        reason="신생검 조직 판독 요청 — 급성 세뇨관 손상 정도 감별",
        urgency="urgent", status="requested",
        requested_by="홍민준", requested_by_user_id="u-neph", requested_at=now - timedelta(hours=2),
    )
    c1.timeline.append(evt("requested", "협진 요청", "홍민준 (신장내과)", now - timedelta(hours=2)))

    # c-002: 분석 완료, 보고서 작성 중(진행) — 홍별철(subject 10081273)
    c2 = Consultation(
        id="c-002", kind="pathology",
        patient_mrn="AKI-10081273", patient_name="홍별철",
        diagnosis="만성 신질환 급성 악화 (CKD on AKI)", key_labs="Cr 4.3 · K 6.1 · eGFR 12",
        reason="만성 변화 정도 및 급성 병변 동반 여부 판독 요청",
        urgency="urgent", status="in_progress",
        requested_by="홍민준", requested_by_user_id="u-neph", requested_at=now - timedelta(hours=6),
    )
    c2.timeline.append(evt("requested", "협진 요청", "홍민준 (신장내과)", now - timedelta(hours=6)))
    c2.timeline.append(evt("received", "병리과 접수", "이수진 (병리과)", now - timedelta(hours=5)))

    return [c1, c2]


def seed_clinical(db: Session) -> None:
    """병리/병상상세/예약/알림 시드 적재(commit 은 호출자)."""
    for c in _consultations():
        db.add(c)
    for p in _pathology():
        db.add(p)
    for bd in _bed_details():
        db.add(bd)
    for n in _notifications():
        db.add(n)
