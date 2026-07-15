"""NLP 전략 인터페이스 (Strategy Pattern).

전사(transcript) → 구조화 증상 추출의 계약을 정의한다.
처리 방식(규칙 기반 / 추후 모델 기반)을 교체 가능하게 한다(OCP/LSP).
"""
from abc import ABC, abstractmethod


class NlpStrategy(ABC):
    """증상 추출 전략."""

    name: str = "base"

    @abstractmethod
    def extract_symptoms(self, transcript: str) -> dict:
        """전사 텍스트에서 증상/소견을 구조화하여 반환.

        반환 예: {"symptoms": ["부종","핍뇨"], "negations": [...], "raw": "..."}
        """
        raise NotImplementedError
