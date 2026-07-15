"""Whisper STT 전략 (선택, 로컬 오프라인 추론).

faster-whisper 가 설치된 경우에만 동작한다(유료 API 아님).
미설치/로드 실패 시 RuntimeError 를 던져 팩토리가 passthrough 로 폴백하게 한다.
모델은 최초 1회만 로드한다(lazy singleton).
"""
import os
import tempfile

from stt.base import SttStrategy


# 모델은 프로세스당 1회만 로드(전략 인스턴스가 요청마다 생성되어도 재로드 방지).
_MODEL_CACHE: dict[str, object] = {}


class WhisperSttStrategy(SttStrategy):
    name = "whisper"

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        cached = _MODEL_CACHE.get(self.model_size)
        if cached is not None:
            self._model = cached
            return
        try:
            from faster_whisper import WhisperModel  # 선택 의존성
        except ImportError as exc:  # pragma: no cover - 환경 의존
            raise RuntimeError(
                "faster-whisper 미설치 — passthrough 전략을 사용하세요."
            ) from exc
        # CPU int8 로 메모리/속도 균형. GPU 환경이면 device='cuda' 로 조정 가능.
        self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        _MODEL_CACHE[self.model_size] = self._model

    def transcribe(self, audio_bytes: bytes | None, fallback_text: str = "") -> str:
        if not audio_bytes:
            return fallback_text or ""
        self._ensure_model()
        # faster-whisper(av) 는 파일 경로를 직접 연다 → Windows 에서는 핸들을 먼저 닫아야
        # 잠금(PermissionError)을 피한다. delete=False 로 만들고 수동 정리한다.
        # 컨테이너/코덱은 av 가 내용으로 자동감지하므로 확장자는 무관(webm/opus 등 OK).
        fd, path = tempfile.mkstemp(suffix=".audio")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(audio_bytes)
            segments, _info = self._model.transcribe(path, language="ko")
            return " ".join(seg.text.strip() for seg in segments).strip()
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
