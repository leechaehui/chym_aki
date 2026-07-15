"""In-process 이벤트 버스 (Kafka-style 토픽).

지시서 §2/§3 의 EVENT BUS 를 단일 프로세스에서 재현한다.
- 실제 Kafka 대신 동기 pub/sub 으로 토픽(LAB/AKI/ALERT/AUDIT)을 제공한다.
- publish() 는 발행 스레드(요청 스레드) 안에서 구독자를 **동기 호출**한다
  → 한 요청이 LAB_EVENT 를 넣으면 AKI 엔진→Alert 서비스까지 같은 흐름에서 처리된다.
- 구독자(소비자)는 요청 scope DB 세션이 없으므로 각자 SessionLocal() 단명 세션을 연다.
- 한 구독자의 예외가 파이프라인 전체/다른 구독자를 죽이지 않도록 격리한다(로깅 후 계속).

설계 원칙(지시서 §12): Event-driven only. 상태 기반 트리거 금지.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from core.logging import get_logger

log = get_logger("chym.eventbus")

# 토픽 상수(문자열) — 지시서 §3 의 eventType 과 동일.
LAB_EVENT = "LAB_EVENT"
AKI_EVENT = "AKI_EVENT"
ALERT_EVENT = "ALERT_EVENT"
AUDIT_EVENT = "AUDIT_EVENT"

ALL_TOPICS = (LAB_EVENT, AKI_EVENT, ALERT_EVENT, AUDIT_EVENT)

Handler = Callable[[dict[str, Any]], None]


class EventBus:
    """토픽별 구독자 목록을 보유하는 동기 pub/sub 버스."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Handler]] = defaultdict(list)

    def subscribe(self, topic: str, handler: Handler) -> None:
        """토픽 구독 등록. 같은 핸들러 중복 등록은 무시(부팅 재호출 안전)."""
        if handler not in self._subscribers[topic]:
            self._subscribers[topic].append(handler)

    def publish(self, topic: str, event: dict[str, Any]) -> None:
        """이벤트 발행 — 해당 토픽 구독자를 등록 순서대로 동기 호출.

        구독자 예외는 격리한다(다른 구독자/발행자를 중단시키지 않음).
        """
        handlers = list(self._subscribers.get(topic, ()))
        log.debug("publish %s -> %d subscriber(s)", topic, len(handlers))
        for handler in handlers:
            try:
                handler(event)
            except Exception:  # noqa: BLE001 — 파이프라인 격리(의료 안전성)
                log.exception("event subscriber failed: topic=%s handler=%s", topic, handler)

    def reset(self) -> None:
        """모든 구독 해제(테스트/재부팅용)."""
        self._subscribers.clear()


# 프로세스 전역 단일 버스.
event_bus = EventBus()
