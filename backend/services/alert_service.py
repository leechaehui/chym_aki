"""Alert Service — AKI_EVENT consumer (지시서 §5).

AKI_EVENT → ALERT_EVENT 변환의 핵심:
  5.1 Deduplication (dedupKey = patientId + type)  — 알림 폭주/중복 제거
  5.2 Priority Queue (CONFIRMED 100 > SUSPECTED 70 > PRE_AKI 40 > SYSTEM 10)
  5.3 Dispatch — 우선순위 높은 순으로 하나씩 영속 + ALERT_EVENT 발행

notification_service(부서 알림 히스토리)와는 별개의 임상 alert 파이프라인이다.
구독 핸들러는 자체 SessionLocal 단명 세션으로 alerts 행을 영속한 뒤 ALERT_EVENT 를 publish 한다
(event_log 적재와 WS 푸시는 ALERT_EVENT 구독자가 담당 → 관심사 분리).
"""
from __future__ import annotations

import heapq
import itertools
import time

from core.database import SessionLocal
from core.event_bus import ALERT_EVENT, event_bus
from core.logging import get_logger
from models.alert import Alert
from models.base import new_id
from repositories.alert_repository import AlertRepository

log = get_logger("chym.alert_service")

# AKI stage → (alert type, severity, priority). NONE 은 alert 미생성.
_STAGE_MAP = {
    "CONFIRMED": ("AKI_CONFIRMED", "critical", 100),
    "SUSPECTED": ("AKI_SUSPECTED", "warning", 70),
    "PRE_AKI": ("PRE_AKI", "info", 40),
}
# AKI alert 의 대상 부서(데모: 신장내과 모니터링).
_DEPARTMENT = "nephrology"

# dedup 인메모리 캐시(TTL). 같은 키 재알림 억제(DB 조회 보완 — 빠른 경로).
_DEDUP_TTL_SEC = 300
_dedup_cache: dict[str, float] = {}

# 우선순위 큐(heapq: (-priority, seq, candidate)). seq 로 동일 우선순위 FIFO 보장.
_counter = itertools.count()
_queue: list[tuple[int, int, dict]] = []


def _seen_recently(dedup_key: str) -> bool:
    now = time.time()
    ts = _dedup_cache.get(dedup_key)
    if ts is not None and (now - ts) < _DEDUP_TTL_SEC:
        return True
    _dedup_cache[dedup_key] = now
    return False


def _build_candidate(event: dict) -> dict | None:
    stage = event.get("stage")
    mapping = _STAGE_MAP.get(stage)
    if not mapping:
        return None  # NONE → 알림 없음
    alert_type, severity, priority = mapping
    patient_id = event.get("patientId") or "unknown"
    score = event.get("aki_score")
    subject_id = event.get("subjectId") if event.get("subjectId") is not None else event.get("subject_id")
    if subject_id is None:
        log.debug("skip alert candidate: missing subject_id patient=%s stage=%s", patient_id, stage)
        return None
        
    # DB에서 실시간 환자 이름 조회하여 이름 표출
    db = SessionLocal()
    patient_name = "환자"
    try:
        row = db.execute(
            text("SELECT name FROM chym.patients WHERE id = :pid OR mimic_subject_id = :sid LIMIT 1"),
            {"pid": patient_id, "sid": subject_id}
        ).first()
        if row:
            patient_name = row[0]
    except Exception:
        pass
    finally:
        db.close()

    rationale = event.get("rationale") or []
    msg = "; ".join(rationale[:3]) if rationale else f"AKI {stage} (score {score})"
    raw_title = {
        "AKI_CONFIRMED": "AKI 확정 위험",
        "AKI_SUSPECTED": "AKI 의심",
        "PRE_AKI": "Pre-AKI 조기 신호",
    }[alert_type]
    title = f"{patient_name}: {raw_title}"
    
    candidate = {
        "id": new_id("alert"),
        "patient_id": patient_id,
        "type": alert_type,
        "severity": severity,
        "priority": priority,
        "aki_stage": event.get("modelStage") or stage,
        "aki_score": score,
        "subject_id": subject_id,
        "dedup_key": f"{patient_id}-{alert_type}",
        "title": title,
        "message": msg,
        "source_event_id": event.get("eventId"),
        "department": _DEPARTMENT,
        "status": "active",
    }
    return candidate


def _persist_and_emit(candidate: dict) -> None:
    """alerts 행 영속(트랜잭션) 후 ALERT_EVENT 발행."""
    db = SessionLocal()
    try:
        repo = AlertRepository(db)
        # DB 차원 dedup(프로세스 재시작/다중워커 대비) — 활성 동일 키 있으면 skip (데모 환자는 바이패스).
        is_demo = candidate["patient_id"].startswith("AKI-")
        if not is_demo and repo.find_active_by_dedup(candidate["dedup_key"]):
            log.debug("dedup(DB) skip %s", candidate["dedup_key"])
            return
        alert = Alert(**candidate)
        repo.add(alert)
        db.commit()
        created_at = alert.created_at.isoformat() if alert.created_at else ""
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    # ALERT_EVENT 발행 → event_log writer + WS pusher 가 소비.
    alert_event = {
        "eventType": ALERT_EVENT,
        "eventId": new_id("alertevt"),
        "alertId": candidate["id"],
        "patientId": candidate["patient_id"],
        # 바로가기 딥링크 — ICU AKI 모니터링에서 해당 환자 검증 리포트 자동 오픈
        # (ClinicalPatientList 의 ?patient= 핸들러).
        "link": f"/nephrology/icu-aki?patient={candidate['patient_id']}",
        "type": candidate["type"],
        "severity": candidate["severity"],
        "priority": candidate["priority"],
        "dedupKey": candidate["dedup_key"],
        "subjectId": candidate.get("subject_id"),
        "title": candidate["title"],
        "message": candidate["message"],
        "akiStage": candidate["aki_stage"],
        "akiScore": candidate["aki_score"],
        "department": candidate["department"],
        "status": candidate["status"],
        "sourceEventId": candidate["source_event_id"],
        "createdAt": created_at,
    }
    event_bus.publish(ALERT_EVENT, alert_event)
    log.info("ALERT_EVENT %s p=%d patient=%s", candidate["type"], candidate["priority"], candidate["patient_id"])


def _drain() -> None:
    """우선순위 큐를 높은 순으로 비우며 하나씩 디스패치(지시서 5.3)."""
    while _queue:
        _, _, candidate = heapq.heappop(_queue)
        _persist_and_emit(candidate)


def process_aki_event(event: dict) -> None:
    """AKI_EVENT 구독 핸들러 — dedup → 우선순위 큐 적재 → 디스패치."""
    candidate = _build_candidate(event)
    if candidate is None:
        return
    is_demo = candidate["patient_id"].startswith("AKI-")
    if not is_demo and _seen_recently(candidate["dedup_key"]):
        log.debug("dedup(cache) skip %s", candidate["dedup_key"])
        return
    heapq.heappush(_queue, (-candidate["priority"], next(_counter), candidate))
    _drain()


def reset_dedup() -> None:
    """테스트/데모 리셋용 — dedup 캐시·큐 비우기."""
    _dedup_cache.clear()
    _queue.clear()
