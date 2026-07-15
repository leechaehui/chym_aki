"""STT 전략 인터페이스 (Strategy Pattern).

음성 → 텍스트 전사의 계약. passthrough(텍스트 직접) / whisper(로컬 추론) 등
구현을 교체 가능하게 한다.
"""
from abc import ABC, abstractmethod


class SttStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def transcribe(self, audio_bytes: bytes | None, fallback_text: str = "") -> str:
        """오디오 바이트를 전사. 오디오가 없으면 fallback_text 를 사용한다."""
        raise NotImplementedError
