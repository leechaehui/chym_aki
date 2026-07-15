"""STT 서비스 — 오디오 → 전사(transcript).

작업지시서 3: Whisper 기반(미설치 시 passthrough 폴백). 출력은 {"transcript": ...}.
STT 전략은 Factory 로 주입(DIP) — whisper/passthrough 교체 가능.
이 서비스는 전사만 책임지며 SOAP/CDSS 로직을 갖지 않는다(SRP).
"""
from __future__ import annotations

from stt.factory import create_stt_strategy


class SttService:
    def __init__(self, strategy_name: str | None = None):
        self.engine = create_stt_strategy(strategy_name)

    def transcribe(self, audio_bytes: bytes | None, fallback_text: str = "") -> dict:
        """오디오 → 전사. 오디오 없으면 fallback_text(passthrough)."""
        transcript = self.engine.transcribe(audio_bytes, fallback_text)
        return {"transcript": transcript or "", "engine": self.engine.name}
