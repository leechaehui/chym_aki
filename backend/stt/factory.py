"""STT 전략 팩토리 (Factory Pattern).

설정값(STT_STRATEGY)에 따라 전략을 생성한다.
whisper 요청 시 의존성/로드 실패하면 passthrough 로 안전하게 폴백한다.
"""
from core.config import settings
from stt.base import SttStrategy
from stt.passthrough_strategy import PassthroughSttStrategy
from stt.whisper_strategy import WhisperSttStrategy


def create_stt_strategy(name: str | None = None) -> SttStrategy:
    """STT 전략 생성. 기본값은 설정의 stt_strategy."""
    chosen = (name or settings.stt_strategy or "passthrough").lower()
    if chosen == "whisper":
        try:
            strategy = WhisperSttStrategy(settings.whisper_model_size)
            strategy._ensure_model()  # 즉시 가용성 확인(실패 시 폴백)
            return strategy
        except RuntimeError:
            # faster-whisper 미설치 등 → passthrough 폴백.
            return PassthroughSttStrategy()
    return PassthroughSttStrategy()
