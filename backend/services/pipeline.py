"""파이프라인 부팅 — 모든 소비자를 이벤트 버스에 등록 (지시서 §11).

LAB_EVENT → AKI_ENGINE → AKI_EVENT → ALERT_SERVICE → ALERT_EVENT → WS push
모든 토픽 → EventLog writer / AUDIT_EVENT → Audit consumer.

init_pipeline() 은 부팅 시 1회 호출(EventBus.subscribe 는 중복 등록을 무시 → 안전).
"""
from __future__ import annotations

import asyncio

from api.ws import push_alert_to_ws, ws_manager
from core.event_bus import (
    AKI_EVENT,
    ALERT_EVENT,
    ALL_TOPICS,
    AUDIT_EVENT,
    LAB_EVENT,
    event_bus,
)
from core.logging import get_logger
from services.aki_engine import process_lab_event
from services.alert_service import process_aki_event
from services.event_consumers import process_audit_event, write_event_log

log = get_logger("chym.pipeline")


def init_pipeline(loop: asyncio.AbstractEventLoop | None = None) -> None:
    """소비자 구독 등록 + WS 매니저에 이벤트 루프 주입."""
    if loop is not None:
        ws_manager.set_loop(loop)

    event_bus.subscribe(LAB_EVENT, process_lab_event)        # §4 AKI 엔진
    event_bus.subscribe(AKI_EVENT, process_aki_event)        # §5 Alert 서비스
    event_bus.subscribe(ALERT_EVENT, push_alert_to_ws)       # §6 WS 푸시

    # full trace — 모든 토픽을 event_log 에 적재(불변).
    for topic in ALL_TOPICS:
        event_bus.subscribe(topic, write_event_log)

    event_bus.subscribe(AUDIT_EVENT, process_audit_event)    # §8 감사 기록
    log.info("event pipeline initialized")
