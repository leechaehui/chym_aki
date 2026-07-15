"""NLP 전략 팩토리 (Factory Pattern).

설정(전략명)에 따라 적절한 NlpStrategy 구현을 생성한다.
호출부는 구체 클래스를 알 필요가 없다(DIP).
"""
from nlp.base import NlpStrategy
from nlp.rule_based_strategy import RuleBasedNlpStrategy

_REGISTRY: dict[str, type[NlpStrategy]] = {
    "rule-based": RuleBasedNlpStrategy,
}


def create_nlp_strategy(name: str = "rule-based") -> NlpStrategy:
    """전략명으로 NLP 전략 인스턴스 생성. 미등록 시 규칙 기반으로 폴백."""
    strategy_cls = _REGISTRY.get(name, RuleBasedNlpStrategy)
    return strategy_cls()
