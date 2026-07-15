"""이벤트 인제스트 + Alert/Audit API (지시서 §8, §11).

- POST /events/lab            : LAB_EVENT 발행(파이프라인 구동) → 생성 alert 반환
- POST /events/lab/simulate-icu: ICU 스트림 시뮬(데모) — 다양한 케이스 LAB_EVENT 다발 발행
- GET  /alerts                : 영속 alert(우선순위순)
- POST /alerts/{id}/audit     : alert 액션 기록(상태 갱신 + AUDIT_EVENT)
- GET  /events/log            : 이벤트 트레이스(감사 데모)
"""
from __future__ import annotations

import re
import time

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from core.deps import get_current_user, get_db, require_roles
from core.event_bus import ALERT_EVENT, AUDIT_EVENT, LAB_EVENT, event_bus
from core.exceptions import NotFoundError
from core.query_optimizer import Page
from models.base import new_id
from models.user import User
from repositories import mimic_repository
from repositories.alert_repository import AlertRepository
from repositories.event_log_repository import EventLogRepository
from schemas.events import (
    AlertAuditInput,
    AlertOut,
    EventLogOut,
    IngestResult,
    LabEvent,
)
from services import alert_service

router = APIRouter(prefix="/events", tags=["events"])
alerts_router = APIRouter(prefix="/alerts", tags=["alerts"])


def _alert_out(a) -> AlertOut:
    return AlertOut(
        id=a.id, patient_id=a.patient_id, type=a.type, severity=a.severity,
        priority=a.priority, aki_stage=a.aki_stage, aki_score=a.aki_score,
        subject_id=a.subject_id, dedup_key=a.dedup_key, title=a.title, message=a.message,
        source_event_id=a.source_event_id, department=a.department, status=a.status,
        created_at=a.created_at.isoformat() if a.created_at else "",
    )


def _publish_lab(patient_id: str, data: dict, *, age: int | None = None, priors: list[float] | None = None, subject_id: int | None = None) -> None:
    event = {
        "eventType": LAB_EVENT,
        "eventId": new_id("labevt"),
        "patientId": patient_id,
        "data": data,
        "age": age,
        "priorCreatinines": priors or [],
        "timestamp": int(time.time()),
    }
    if subject_id is not None:
        event["subjectId"] = subject_id
    event_bus.publish(LAB_EVENT, event)


@router.post("/lab", response_model=IngestResult)
def ingest_lab(
    body: LabEvent,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """LAB_EVENT 1건 인제스트 → 파이프라인 동기 처리 → 해당 환자 활성 alert 반환."""
    _publish_lab(
        body.patient_id, body.data.model_dump(by_alias=True),
        age=body.age, priors=body.prior_creatinines,
        subject_id=body.subject_id,
    )
    repo = AlertRepository(db)
    alerts = [a for a in repo.list_filtered(Page.of(20, 0), status="active")
              if a.patient_id == body.patient_id]
    return IngestResult(accepted=1, alerts=[_alert_out(a) for a in alerts])


# 데모 케이스 — Safety 파이프라인 전 구간(KDIGO/sensitive/trend/guardrail/baseline)을 자극.
_DEMO_CASES = [
    # (suffix, data, age, priors) → 기대 결정
    ("conf", {"creatinine": 3.2, "egfr": 18, "urineOutput": 0.2}, 72, [1.0, 1.0]),                       # CONFIRMED (KDIGO S3)
    ("susp", {"creatinine": 2.8, "egfr": 30, "urineOutput": 0.7, "symptom": "Cr 상승"}, 65, [1.0]),       # SUSPECTED (sensitive≥th, baseline 저신뢰로 KDIGO 보류)
    ("pre", {"creatinine": 1.25, "egfr": 75, "urineOutput": 1.0}, 60, [1.0, 1.0]),                        # PRE_AKI (Cr 1.25× 약신호)
    ("trend", {"creatinine": 1.3, "egfr": 68, "urineOutput": 1.1}, 58, [1.0, 1.1, 1.2]),                  # PRE_AKI (상승 추세 감지)
    ("none", {"creatinine": 0.9, "egfr": 95, "urineOutput": 1.3}, 45, [0.9, 1.0]),                        # NONE
    ("fp", {"creatinine": 2.2, "egfr": 35, "urineOutput": 0.9, "dehydration": True}, 70, [1.0, 1.0]),     # CONFIRMED(KDIGO S2)+guardrail 설명
    ("ckd", {"creatinine": 1.6, "egfr": 45, "urineOutput": 1.0, "ckdRisk": True, "sex": "M"}, 68, []),    # CKD baseline 상향 → 과진단 억제
]


@router.post("/lab/simulate-icu", response_model=IngestResult)
def simulate_icu(
    n: int = 6,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("nephrology", "admin")),
):
    """ICU 스트림 시뮬레이션 — 다양한 임상 케이스를 LAB_EVENT 로 발행해 파이프라인 시연.

    dedup 시연을 위해 동일 케이스를 반복해도 중복 alert 는 억제된다.
    """
    n = max(1, min(n, 30))
    patient_ids: list[str] = []
    for i in range(n):
        suffix, data, age, priors = _DEMO_CASES[i % len(_DEMO_CASES)]
        # 케이스 묶음마다 환자 id 를 바꿔(주기) dedup 과 신규 alert 를 모두 보여준다.
        pid = f"ICU-{(i // len(_DEMO_CASES)) + 1:02d}-{suffix}"
        subject_id = 10000000 + i
        patient_ids.append(pid)
        _publish_lab(
            pid, dict(data), age=age, priors=list(priors),
            subject_id=subject_id,
        )

    repo = AlertRepository(db)
    alerts = [a for a in repo.list_filtered(Page.of(50, 0)) if a.patient_id in set(patient_ids)]
    return IngestResult(accepted=n, alerts=[_alert_out(a) for a in alerts])


@router.post("/lab/from-stay/{stay_id}", response_model=IngestResult)
def ingest_from_stay(
    stay_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("nephrology", "admin")),
):
    """실제 MIMIC-IV ICU stay 의 시계열을 LAB_EVENT 로 인제스트(라이브).

    chym.cr_timeseries(과거→최신 Cr) + urine_rate(최근 UO) + cohort(나이/성별) 을 읽어
    Safety 엔진(baseline/KDIGO/sensitive/ROC)을 구동한다. 진짜 환자 궤적 기반 실시간 alert.
    """
    series = mimic_repository.stay_creatinine_series(stay_id)
    if not series:
        raise NotFoundError("해당 stay 의 creatinine 시계열이 없습니다.")
    demo = mimic_repository.stay_demographics(stay_id) or {}
    uo = mimic_repository.stay_latest_urine_rate(stay_id)

    # 현재 Cr = 시계열 최대값(악화 시점 포착, recall-first). baseline 은 공식
    # chym.baseline_creatinine(입실 전 기준)을 우선 사용해 시계열 median 인플레이션을 피한다.
    current_cr = max(series)
    official = mimic_repository.stay_baseline(stay_id)
    if official:
        # 공식 baseline 을 high-confidence(EMR median) 로 인식시키기 위해 2점 prior 로 전달.
        priors = [official["baseline_cr"], official["baseline_cr"]]
    else:
        priors = series[:-1]  # 폴백: 과거 Cr
    data = {"creatinine": current_cr, "urineOutput": uo, "sex": demo.get("gender")}
    cohort = mimic_repository.cohort_record(stay_id)
    subject_id = int(cohort["subject_id"]) if cohort and cohort.get("subject_id") is not None else None
    _publish_lab(
        f"stay-{stay_id}", data, age=demo.get("age"), priors=priors,
        subject_id=subject_id,
    )

    repo = AlertRepository(db)
    alerts = [a for a in repo.list_filtered(Page.of(20, 0), status="active")
              if a.patient_id == f"stay-{stay_id}"]
    return IngestResult(accepted=1, alerts=[_alert_out(a) for a in alerts])


@alerts_router.get("", response_model=list[AlertOut])
def list_alerts(
    department: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """영속 alert 목록(우선순위 desc). department 미지정 시 현재 사용자 부서."""
    dept = department or current_user.role
    repo = AlertRepository(db)
    return [_alert_out(a) for a in repo.list_filtered(Page.of(limit, offset), department=dept, status=status)]


_ACTION_STATUS = {
    "VIEWED": "viewed",
    "DISMISSED": "dismissed",
    "ACKNOWLEDGED": "acknowledged",
    "ESCALATED": "escalated",
}

# 에스컬레이션 상급 대상 부서(= role). 신장내과 당직 → 병원 총괄(admin) 재알림.
_ESCALATION_TARGET_DEPT = "admin"


def _escalate_to_admin(db: Session, alert) -> None:
    """에스컬레이션 시 상급 부서로 실제 재알림을 만든다.

    - chym.notifications 영속(상급 부서 드로어/히스토리에 남김)
    - ALERT_EVENT 를 대상 부서로 발행 → WS 로 해당 부서 세션에 라이브 모달 전달
    재에스컬레이션 루프 방지를 위해 라이브 이벤트에는 alertId 를 싣지 않는다(모달에 '바로가기/확인'만 노출).
    """
    # 커밋 전에 필요한 원본 alert 필드를 지역 변수로 확보(commit 후 만료 재조회 회피).
    sid = alert.subject_id
    mrn = alert.patient_id
    src_title = alert.title
    aki_stage = alert.aki_stage
    aki_score = alert.aki_score
    src_id = alert.id

    name = None
    if sid is not None:
        row = db.execute(
            text("SELECT name FROM chym.patients WHERE mimic_subject_id = :sid"), {"sid": sid}
        ).first()
        name = row[0] if row else None
    if name is None:
        row = db.execute(
            text("SELECT name FROM chym.patients WHERE mrn = :mrn"), {"mrn": mrn}
        ).first()
        name = row[0] if row else mrn

    link = f"/nephrology/icu-aki?patient={mrn}"
    title = f"[에스컬레이션] {name} 환자 상급 검토 요청"
    message = f"신장내과에서 '{src_title}' 건을 에스컬레이션했습니다. 즉시 확인이 필요합니다."

    # 1) 영속 notification (상급 부서 히스토리)
    db.execute(text("""
        INSERT INTO chym.notifications (id, department, severity, title, message, link, read)
        VALUES (:id, :dept, 'ACTION_REQUIRED', :title, :message, :link, false)
    """), {
        "id": new_id("noti-esc"), "dept": _ESCALATION_TARGET_DEPT,
        "title": title, "message": message, "link": link,
    })
    db.commit()

    # 2) 라이브 WS 푸시 (대상 부서 모달). push_alert_to_ws 가 department 로 라우팅.
    event_bus.publish(ALERT_EVENT, {
        "eventType": ALERT_EVENT,
        "eventId": new_id("alertevt"),
        "alertId": None,
        "patientId": mrn,
        "subjectId": sid,
        "type": "ESCALATION",
        "severity": "critical",
        "priority": 100,
        "dedupKey": f"esc-{src_id}",
        "title": title,
        "message": message,
        "link": link,
        "akiStage": aki_stage,
        "akiScore": aki_score,
        "department": _ESCALATION_TARGET_DEPT,
        "status": "active",
        "sourceEventId": src_id,
        "createdAt": "",
    })


@alerts_router.post("/{alert_id}/audit", response_model=AlertOut)
def audit_alert(
    alert_id: str,
    body: AlertAuditInput,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """alert 액션 기록 — 상태 갱신 + AUDIT_EVENT 발행(감사 추적)."""
    repo = AlertRepository(db)
    alert = repo.get(alert_id)
    if not alert:
        raise NotFoundError("알림을 찾을 수 없습니다.")
    # VIEWED 는 미해소 상태에서만 갱신(해소 상태를 되돌리지 않음).
    new_status = _ACTION_STATUS[body.action]
    if not (body.action == "VIEWED" and alert.status not in ("active",)):
        alert.status = new_status
    db.commit()

    # 에스컬레이션 → 상급 부서로 실제 재알림(영속 notification + 라이브 WS).
    if body.action == "ESCALATED":
        _escalate_to_admin(db, alert)

    event_bus.publish(AUDIT_EVENT, {
        "eventType": AUDIT_EVENT,
        "eventId": new_id("auditevt"),
        "alertId": alert_id,
        "patientId": alert.patient_id,
        "action": body.action,
        "userId": current_user.id,
        "role": body.role or current_user.role,
        "timestamp": int(time.time()),
    })
    return _alert_out(alert)


@router.get("/log", response_model=list[EventLogOut])
def list_event_log(
    patient_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    _: User = Depends(require_roles("nephrology", "admin")),
):
    """이벤트 트레이스(full trace) — 감사/디버그용."""
    import json

    repo = EventLogRepository(db)
    rows = repo.list_recent(Page.of(limit, offset), patient_id=patient_id)
    out: list[EventLogOut] = []
    for r in rows:
        try:
            payload = json.loads(r.payload) if r.payload else None
        except json.JSONDecodeError:
            payload = None
        out.append(EventLogOut(
            id=r.id, event_type=r.event_type, patient_id=r.patient_id,
            payload=payload, created_at=r.created_at.isoformat() if r.created_at else "",
        ))
    return out
