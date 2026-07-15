"""규칙 기반 증상 추출 전략.

신장내과 도메인 키워드 사전으로 한국어 전사에서 증상을 식별한다.
부정 표현(없다/아니다)이 동반되면 negations 로 분리한다.
실제 LLM/의료 NLP 모델로 교체할 때 이 클래스만 갈아끼우면 된다(OCP).
"""
import re

from nlp.base import NlpStrategy


# 증상 키워드 → 표준 라벨 사전(신장/전해질/감염 관련 주요 호소).
SYMPTOM_LEXICON: dict[str, list[str]] = {
    "부종": ["부종", "붓", "edema", "다리가 붓"],
    "핍뇨": ["소변량 감소", "소변이 줄", "소변량이 줄", "핍뇨", "oliguria"],
    "무뇨": ["소변이 안", "소변이 거의", "무뇨", "anuria"],
    "오심/구토": ["메스꺼", "구역", "구토", "토할", "nausea", "vomit"],
    "전신쇠약": ["기운이 없", "무기력", "쇠약", "weak", "피로"],
    "호흡곤란": ["숨이 차", "호흡곤란", "숨쉬기", "dyspnea"],
    "발열": ["열이", "고열", "발열", "fever"],
    "의식저하": ["처지", "의식", "졸리", "confus"],
    "흉통": ["가슴", "흉통", "chest pain"],
    "혈뇨": ["피", "혈뇨", "붉은 소변", "hematuria"],
}

# 부정 표현 — 키워드 직후/직전에 등장하면 negation 으로 처리.
NEGATION_HINTS = ["없", "아니", "안 ", "않", "no ", "denies"]


class RuleBasedNlpStrategy(NlpStrategy):
    name = "rule-based"

    def extract_symptoms(self, transcript: str) -> dict:
        raw = transcript or ""
        text = raw.lower()
        found: list[str] = []
        negated: list[str] = []
        # 증상 라벨 → 원문 증거 스니펫(SOAP A/P evidence 의 근거).
        evidence: dict[str, str] = {}

        for label, keywords in SYMPTOM_LEXICON.items():
            for kw in keywords:
                idx = text.find(kw.lower())
                if idx == -1:
                    continue
                # 키워드 주변 윈도우에서 부정 표현 탐색.
                window = text[max(0, idx - 6) : idx + len(kw) + 6]
                if any(neg in window for neg in NEGATION_HINTS):
                    if label not in negated:
                        negated.append(label)
                else:
                    if label not in found:
                        found.append(label)
                        evidence[label] = _snippet(raw, idx, len(kw))
                break

        # 숫자+단위 패턴(예: "39.4도", "소변 15ml")을 측정치 후보로 수집.
        measurements = re.findall(r"\d+(?:\.\d+)?\s?(?:도|℃|mg|ml|mmol|bpm|%)", text)

        return {
            "symptoms": found,
            "negations": negated,
            "measurements": measurements,
            "evidence": evidence,
            "raw": raw,
        }


def _snippet(raw: str, idx: int, kw_len: int, window: int = 18) -> str:
    """원문에서 키워드 주변 구절을 잘라 증거 스니펫으로 반환(원문 그대로)."""
    start = max(0, idx - window)
    end = min(len(raw), idx + kw_len + window)
    snippet = raw[start:end].strip()
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(raw) else ""
    return f"{prefix}{snippet}{suffix}"
