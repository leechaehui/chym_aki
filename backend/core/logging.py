"""구조적 로깅 + request_id 기반 트레이싱 (작업지시서 3.3).

책임: 단 하나 — 요청 단위로 추적 가능한 로깅 인프라를 제공한다.
- request_id 를 ContextVar 로 보관해, 같은 요청에서 발생한 모든 로그에 자동 부착한다.
- 로그 포맷에 request_id 를 포함시켜 분산 추적/감사와 연결한다.
- 비즈니스 로직 없음(설정/포맷터/필터만).

사용
  from core.logging import get_logger
  log = get_logger(__name__)
  log.info("bed assigned", extra={"bed_id": bed_id})   # → request_id 자동 포함
"""
from __future__ import annotations

import logging
import sys
from contextvars import ContextVar

# 요청 수명 동안 유지되는 추적 ID. 미들웨어가 set, 핸들러/서비스가 참조.
_request_id: ContextVar[str] = ContextVar("request_id", default="-")


def set_request_id(value: str) -> None:
    _request_id.set(value)


def get_request_id() -> str:
    return _request_id.get()


class _RequestIdFilter(logging.Filter):
    """모든 LogRecord 에 현재 request_id 를 주입한다."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


_FORMAT = "%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s: %(message)s"
_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    """루트 로거를 1회 구성(stdout 핸들러 + request_id 필터)."""
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT))
    handler.addFilter(_RequestIdFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """request_id 필터가 부착된 로거 반환(구성 보장)."""
    if not _configured:
        configure_logging()
    return logging.getLogger(name)
