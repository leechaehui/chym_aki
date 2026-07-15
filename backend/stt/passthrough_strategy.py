"""Passthrough STT 전략 (기본값).

음성 인식 엔진 없이 클라이언트가 이미 가진/입력한 전사 텍스트를 그대로 통과시킨다.
유료 API·무거운 모델 의존 없이 즉시 동작한다(데모/개발 기본 경로).
"""
from stt.base import SttStrategy


class PassthroughSttStrategy(SttStrategy):
    name = "passthrough"

    def transcribe(self, audio_bytes: bytes | None, fallback_text: str = "") -> str:
        # 오디오가 있으면 테스트용 더미 텍스트 반환
        if audio_bytes and len(audio_bytes) > 0:
            return "어제부터 소변량이 줄고 다리가 많이 붓는 것 같습니다. 숨도 좀 차고요."
        return fallback_text or ""
